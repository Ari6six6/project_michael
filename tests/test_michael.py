"""Tests for michael's internals: slugs, projects, paths, staging, trash, replay."""
from __future__ import annotations

import pathlib
import shutil

import pytest

import main as m


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Patch all michael path globals to live under tmp_path/.michael."""
    state = tmp_path / ".michael"
    monkeypatch.setattr(m, "STATE_DIR", state)
    monkeypatch.setattr(m, "GLOBAL_CONFIG_PATH", state / "config.json")
    monkeypatch.setattr(m, "GLOBAL_EVENTS_PATH", state / "events.jsonl")
    monkeypatch.setattr(m, "STATE_FILE_PATH", state / "state.json")
    monkeypatch.setattr(m, "PROJECTS_DIR", state / "projects")
    monkeypatch.setattr(m, "REPL_HISTORY_PATH", state / "repl_history")
    state.mkdir()
    return state


@pytest.fixture
def workspace(tmp_path):
    """Fresh project workspace with a couple of files plus dotted+skip dirs."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "src").mkdir()
    (ws / "src" / "foo.py").write_text("x = 1\n")
    (ws / "README.md").write_text("# hi\n")
    (ws / ".git").mkdir()
    (ws / ".git" / "HEAD").write_text("ref: x\n")
    (ws / "node_modules").mkdir()
    (ws / "node_modules" / "junk.js").write_text("// no\n")
    return ws


# ---- slugify -------------------------------------------------------------

def test_slugify_basic():
    assert m.slugify("Hello World") == "hello-world"


def test_slugify_strips_specials():
    assert m.slugify("foo!@# bar/baz") == "foo-bar-baz"


def test_slugify_empty_falls_back():
    assert m.slugify("") == "project"
    assert m.slugify("///") == "project"


def test_slugify_truncates_to_64():
    s = m.slugify("a" * 200)
    assert len(s) <= 64


# ---- project model -------------------------------------------------------

def test_create_project_round_trip(home, workspace):
    p = m.create_project("my proj", workspace)
    assert p.slug == "my-proj"
    assert p.name == "my proj"
    assert pathlib.Path(p.path).resolve() == workspace.resolve()
    loaded = m.Project.load("my-proj")
    assert loaded == p


def test_create_project_collision_appends_suffix(home, workspace, tmp_path):
    m.create_project("foo", workspace)
    ws2 = tmp_path / "ws2"
    ws2.mkdir()
    p2 = m.create_project("foo", ws2)
    assert p2.slug == "foo-2"


def test_list_projects_sorted(home, tmp_path):
    for n in ("zeta", "alpha", "mike"):
        ws = tmp_path / n
        ws.mkdir()
        m.create_project(n, ws)
    slugs = [p.slug for p in m.list_projects()]
    assert slugs == sorted(slugs)


# ---- path-escape guard ---------------------------------------------------

def test_resolve_in_project_ok(home, workspace):
    p = m.create_project("x", workspace)
    r = m._resolve_in_project(p, "src/foo.py")
    assert r == (workspace / "src" / "foo.py").resolve()


def test_resolve_in_project_refuses_escape(home, workspace):
    p = m.create_project("x", workspace)
    with pytest.raises(m.MichaelError):
        m._resolve_in_project(p, "../escape.txt")


def test_resolve_in_project_refuses_absolute(home, workspace):
    p = m.create_project("x", workspace)
    with pytest.raises(m.MichaelError):
        m._resolve_in_project(p, "/etc/passwd")


# ---- file hashes & diff --------------------------------------------------

def test_file_hashes_skips_dotted_and_skipdirs(home, workspace):
    h = m._file_hashes(workspace)
    assert ".git/HEAD" not in h
    assert "node_modules/junk.js" not in h
    assert "src/foo.py" in h
    assert "README.md" in h


def test_diff_hashes_classifies_correctly():
    before = {"a": "1", "b": "2", "c": "3"}
    after = {"a": "1", "b": "9", "d": "4"}
    d = m._diff_hashes(before, after)
    assert d["added"] == ["d"]
    assert d["removed"] == ["c"]
    assert d["modified"] == ["b"]


# ---- check_expected ------------------------------------------------------

def test_check_expected_match():
    delta = {"added": ["a"], "modified": ["b"], "removed": []}
    assert m._check_expected(["a", "b"], delta) == ""


def test_check_expected_extra():
    delta = {"added": ["a", "c"], "modified": [], "removed": []}
    msg = m._check_expected(["a"], delta)
    assert "extra" in msg and "c" in msg


def test_check_expected_missing():
    delta = {"added": [], "modified": [], "removed": []}
    msg = m._check_expected(["a"], delta)
    assert "missing" in msg


# ---- staging -------------------------------------------------------------

def test_stage_project_skips_dotted_and_skipdirs(home, workspace):
    p = m.create_project("x", workspace)
    stage = m._stage_project(p)
    try:
        assert (stage / "src" / "foo.py").read_text() == "x = 1\n"
        assert not (stage / ".git").exists()
        assert not (stage / "node_modules").exists()
    finally:
        shutil.rmtree(stage.parent, ignore_errors=True)


def test_apply_in_staging_write_file_does_not_touch_real(home, workspace):
    p = m.create_project("x", workspace)
    stage = m._stage_project(p)
    try:
        m._apply_in_staging(
            "write_file",
            {"path": "src/bar.py", "content": "y = 2\n"},
            stage,
        )
        assert (stage / "src" / "bar.py").read_text() == "y = 2\n"
        assert not (workspace / "src" / "bar.py").exists()
    finally:
        shutil.rmtree(stage.parent, ignore_errors=True)


def test_apply_in_staging_refuses_escape(home, workspace):
    p = m.create_project("x", workspace)
    stage = m._stage_project(p)
    try:
        with pytest.raises(m.MichaelError):
            m._apply_in_staging(
                "write_file",
                {"path": "../escape.txt", "content": "x"},
                stage,
            )
    finally:
        shutil.rmtree(stage.parent, ignore_errors=True)


# ---- trash + undo --------------------------------------------------------

def test_save_trash_and_undo_modified(home, workspace):
    p = m.create_project("x", workspace)
    real = workspace.resolve()
    delta = {"added": [], "modified": ["src/foo.py"], "removed": []}
    m._save_trash(p, "write_file",
                  {"path": "src/foo.py", "content": "x = 99\n"},
                  delta, real, verify_rc=None)
    (real / "src" / "foo.py").write_text("x = 99\n")
    m._undo_one(p)
    assert (real / "src" / "foo.py").read_text() == "x = 1\n"


def test_undo_added_deletes_file(home, workspace):
    p = m.create_project("x", workspace)
    real = workspace.resolve()
    delta = {"added": ["src/new.py"], "modified": [], "removed": []}
    m._save_trash(p, "write_file",
                  {"path": "src/new.py", "content": "z = 3\n"},
                  delta, real, verify_rc=None)
    (real / "src" / "new.py").write_text("z = 3\n")
    m._undo_one(p)
    assert not (real / "src" / "new.py").exists()


def test_undo_with_no_trash_errors(home, workspace):
    p = m.create_project("x", workspace)
    with pytest.raises(m.MichaelError):
        m._undo_one(p)


# ---- sync_to_real --------------------------------------------------------

def test_sync_to_real_applies_added_and_modified(home, workspace, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "src").mkdir()
    (stage / "src" / "foo.py").write_text("modified\n")
    (stage / "src" / "new.py").write_text("added\n")
    delta = {
        "added": ["src/new.py"],
        "modified": ["src/foo.py"],
        "removed": [],
    }
    m._sync_to_real(stage, workspace, delta)
    assert (workspace / "src" / "foo.py").read_text() == "modified\n"
    assert (workspace / "src" / "new.py").read_text() == "added\n"


def test_sync_to_real_handles_removed(home, workspace, tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    delta = {"added": [], "modified": [], "removed": ["src/foo.py"]}
    m._sync_to_real(stage, workspace, delta)
    assert not (workspace / "src" / "foo.py").exists()


# ---- event log + replay --------------------------------------------------

def test_replay_instance_lifecycle(home):
    m.append_event("instance.start_requested", {"id": "1", "model": "coder"})
    m.append_event("instance.started", {"id": "1", "model": "coder"})
    state = m.replay_global()
    assert state["models"]["coder"]["instance_state"] == "running"


def test_iter_events_skips_garbage(home):
    log = home / "events.jsonl"
    log.write_text(
        '{"seq": 1, "ts": "x", "type": "test.ok", "payload": {}}\n'
        "this is not json\n"
        '{"seq": 2, "ts": "x", "type": "test.ok", "payload": {}}\n'
    )
    events = m.iter_events(log)
    assert len(events) == 2
    assert events[0]["seq"] == 1
    assert events[1]["seq"] == 2


# ---- filesystem_snapshot shape -------------------------------------------

def test_filesystem_snapshot_lists_files_and_skips_junk(home, workspace):
    snap = m.filesystem_snapshot(workspace)
    assert "src/foo.py" in snap
    assert "README.md" in snap
    assert ".git" not in snap
    assert "node_modules" not in snap


# ---- _config_is_unset ----------------------------------------------------

def test_config_is_unset_when_missing(home):
    assert m._config_is_unset() is True


def test_config_is_unset_when_blank(home):
    cfg = m.Config()
    cfg.save()
    assert m._config_is_unset() is True


def test_config_is_unset_false_when_keys_set(home):
    cfg = m.make_stub_config()
    cfg.vast_api_key = "x"
    cfg.models["coder"].vast_instance_id = "12345"
    cfg.save()
    assert m._config_is_unset() is False


# ---- mode helpers --------------------------------------------------------

def test_tools_for_mode_discussion_is_read_only():
    tools = m._tools_for_mode("discussion")
    names = {t["function"]["name"] for t in tools}
    assert names == m.AUTO_EXEC_TOOLS
    assert "write_file" not in names
    assert "run_shell" not in names


def test_tools_for_mode_code_and_nitro_are_full():
    code_names = {t["function"]["name"] for t in m._tools_for_mode("code")}
    nitro_names = {t["function"]["name"] for t in m._tools_for_mode("nitro")}
    full = {t["function"]["name"] for t in m.TOOLS}
    assert code_names == full
    assert nitro_names == full


def test_resolve_nitro_prefers_nitro_then_big_then_errors(home):
    cfg = m.Config(models={"coder": m.ModelProfile()})
    with pytest.raises(m.MichaelError):
        m._resolve_nitro_model(cfg, None)

    cfg = m.Config(models={"coder": m.ModelProfile(), "big": m.ModelProfile()})
    name, _ = m._resolve_nitro_model(cfg, None)
    assert name == "big"

    cfg = m.Config(models={
        "coder": m.ModelProfile(),
        "big": m.ModelProfile(),
        "nitro": m.ModelProfile(),
    })
    name, _ = m._resolve_nitro_model(cfg, None)
    assert name == "nitro"


def test_resolve_nitro_explicit_model_wins(home):
    cfg = m.Config(models={
        "coder": m.ModelProfile(),
        "nitro": m.ModelProfile(),
    })
    name, _ = m._resolve_nitro_model(cfg, "coder")
    assert name == "coder"


# ---- tool schema: expected_changes is required --------------------------

def _tool(name):
    return next(t for t in m.TOOLS if t["function"]["name"] == name)


def test_write_file_schema_requires_expected_changes():
    schema = _tool("write_file")
    required = schema["function"]["parameters"]["required"]
    assert "expected_changes" in required


def test_apply_patch_schema_requires_expected_changes():
    schema = _tool("apply_patch")
    required = schema["function"]["parameters"]["required"]
    assert "expected_changes" in required


# ---- predicted-delta gate (review reporter, not auto-reject) ------------

def test_execute_with_staging_missing_expected_returns_error_to_llm(home, workspace):
    p = m.create_project("x", workspace)
    cfg = m.Config()
    pending = m.PendingChanges()
    result = m.execute_with_staging(
        "write_file",
        {"path": "src/bar.py", "content": "y = 2\n"},
        p, cfg, pending,
    )
    assert result.startswith("error: expected_changes is required")
    assert not (workspace / "src" / "bar.py").exists()
    assert pending.stage_root is None
    events = m.iter_events(p.events_path)
    types = [e.get("type") for e in events]
    assert "tool.delta_missing" in types


def test_execute_with_staging_mismatch_is_review_data_not_rejection(home, workspace):
    p = m.create_project("x", workspace)
    cfg = m.Config()
    pending = m.PendingChanges()
    # LLM predicts a different file than what the write actually changes.
    result = m.execute_with_staging(
        "write_file",
        {
            "path": "src/bar.py",
            "content": "y = 2\n",
            "expected_changes": ["src/wrong.py"],
        },
        p, cfg, pending,
    )
    # Mismatch is information, not auto-rejection — no error: prefix.
    assert not result.startswith("error:")
    assert "predicted:" in result and "actual:" in result
    assert "src/bar.py" in result
    # Real workspace untouched; staging holds the change for the LLM to review.
    assert not (workspace / "src" / "bar.py").exists()
    assert pending.stage_root is not None
    assert len(pending.change_log) == 1
    events = m.iter_events(p.events_path)
    types = [e.get("type") for e in events]
    assert "tool.delta_mismatch" in types
    assert "tool.staged" in types


def test_execute_with_staging_review_returns_diff_without_prompt(home, workspace, monkeypatch):
    """A clean prediction match returns review data and stages the change.
    The user is NOT prompted; commit is deferred to the Ja gate."""
    p = m.create_project("x", workspace)
    cfg = m.Config()
    pending = m.PendingChanges()
    # If anything tries to prompt the user, fail loudly.
    monkeypatch.setattr(
        m.typer, "prompt",
        lambda *a, **k: pytest.fail("user must not be prompted in review mode"),
    )
    result = m.execute_with_staging(
        "write_file",
        {
            "path": "src/bar.py",
            "content": "y = 2\n",
            "expected_changes": ["src/bar.py"],
        },
        p, cfg, pending,
    )
    assert "predicted:" in result and "actual:" in result
    assert "match" in result.lower()
    # Stage holds the file; real workspace does not.
    assert (pending.stage_root / "src" / "bar.py").read_text() == "y = 2\n"
    assert not (workspace / "src" / "bar.py").exists()
    assert len(pending.change_log) == 1


def test_pending_changes_accumulates_across_calls(home, workspace):
    p = m.create_project("x", workspace)
    cfg = m.Config()
    pending = m.PendingChanges()
    r1 = m.execute_with_staging(
        "write_file",
        {"path": "src/a.py", "content": "a = 1\n", "expected_changes": ["src/a.py"]},
        p, cfg, pending,
    )
    r2 = m.execute_with_staging(
        "write_file",
        {"path": "src/b.py", "content": "b = 2\n", "expected_changes": ["src/b.py"]},
        p, cfg, pending,
    )
    assert "predicted:" in r1 and "predicted:" in r2
    # One persistent stage_root reused; two entries in the change log.
    assert pending.stage_root is not None
    assert len(pending.change_log) == 2
    # Both files exist in stage; neither in real.
    assert (pending.stage_root / "src" / "a.py").is_file()
    assert (pending.stage_root / "src" / "b.py").is_file()
    assert not (workspace / "src" / "a.py").exists()
    assert not (workspace / "src" / "b.py").exists()


def test_commit_pending_syncs_all_entries_then_discards(home, workspace):
    p = m.create_project("x", workspace)
    cfg = m.Config()
    pending = m.PendingChanges()
    m.execute_with_staging(
        "write_file",
        {"path": "src/a.py", "content": "a = 1\n", "expected_changes": ["src/a.py"]},
        p, cfg, pending,
    )
    m.execute_with_staging(
        "write_file",
        {"path": "src/b.py", "content": "b = 2\n", "expected_changes": ["src/b.py"]},
        p, cfg, pending,
    )
    summaries = m.commit_pending(p, pending)
    assert len(summaries) == 2
    assert (workspace / "src" / "a.py").read_text() == "a = 1\n"
    assert (workspace / "src" / "b.py").read_text() == "b = 2\n"
    # Stage is discarded after commit.
    assert pending.stage_root is None
    assert pending.change_log == []
    events = m.iter_events(p.events_path)
    types = [e.get("type") for e in events]
    assert types.count("tool.executed") == 2


# ---- Ja passcode and detector -------------------------------------------

def test_ja_passphrase_constant():
    assert m.JA_PASSPHRASE == "Ja"


def test_ja_detector_recognises_end_of_message():
    assert m._message_ends_with_ja("thoughts.\nJa")
    assert m._message_ends_with_ja("thoughts.\nJa\n")
    assert m._message_ends_with_ja("done with the work. Ja")
    assert m._message_ends_with_ja("done. Ja.")
    assert m._message_ends_with_ja("Ja")


def test_ja_detector_rejects_mid_sentence_and_other_languages():
    assert not m._message_ends_with_ja("")
    assert not m._message_ends_with_ja("Yes")
    assert not m._message_ends_with_ja("Ja, das ist gut")
    assert not m._message_ends_with_ja("Ja im Anfang")
    assert not m._message_ends_with_ja("ja")  # case-sensitive


# ---- Header 4 / build_protocol ------------------------------------------

def test_build_protocol_lists_four_headers():
    text = m.build_protocol("code")
    for h in ("H1", "H2", "H3", "H4"):
        assert h in text


def test_build_protocol_mentions_ja_passcode_and_no_hands():
    text = m.build_protocol("code")
    assert "Ja" in text
    assert "passcode" in text.lower()
    assert "no hands" in text.lower() or "NO HANDS" in text


def test_build_protocol_mode_addendum_changes():
    code = m.build_protocol("code")
    discussion = m.build_protocol("discussion")
    nitro = m.build_protocol("nitro")
    assert "MODE: code" in code
    assert "MODE: discussion" in discussion
    assert "MODE: nitro" in nitro


def test_build_header_includes_protocol(home, workspace):
    p = m.create_project("x", workspace)
    pkg = m.build_header(p, "system stub", mode="code")
    assert "H4: Protocol" in pkg
    assert "Ja" in pkg
    # H1/H2/H3 markers are present too.
    assert "H1:" in pkg and "H2:" in pkg and "H3:" in pkg


# ---- REPL surface --------------------------------------------------------

def test_repl_commands_include_nitro_and_new_subcommands():
    assert "nitro" in m.REPL_COMMANDS
    assert "new" in m.REPL_COMMANDS
    assert set(m.NEW_SUBCOMMANDS) == {"project", "code", "discussion"}


# ---- tier-specific model resolution: code and discussion ----------------

def test_resolve_code_prefers_coder_then_default(home):
    # Only default_model configured — falls back cleanly
    cfg = m.Config(models={"instruct": m.ModelProfile()}, default_model="instruct")
    name, _ = m._resolve_code_model(cfg, None)
    assert name == "instruct"

    # "coder" profile present — should be picked over default
    cfg = m.Config(models={"instruct": m.ModelProfile(), "coder": m.ModelProfile()},
                   default_model="instruct")
    name, _ = m._resolve_code_model(cfg, None)
    assert name == "coder"


def test_resolve_code_explicit_model_wins(home):
    cfg = m.Config(models={"coder": m.ModelProfile(), "instruct": m.ModelProfile()},
                   default_model="coder")
    name, _ = m._resolve_code_model(cfg, "instruct")
    assert name == "instruct"


def test_resolve_discussion_prefers_instruct_then_default(home):
    # Only default_model configured — falls back cleanly
    cfg = m.Config(models={"coder": m.ModelProfile()}, default_model="coder")
    name, _ = m._resolve_discussion_model(cfg, None)
    assert name == "coder"

    # "instruct" profile present — should be picked over default
    cfg = m.Config(models={"coder": m.ModelProfile(), "instruct": m.ModelProfile()},
                   default_model="coder")
    name, _ = m._resolve_discussion_model(cfg, None)
    assert name == "instruct"


def test_resolve_discussion_explicit_model_wins(home):
    cfg = m.Config(models={"coder": m.ModelProfile(), "instruct": m.ModelProfile()},
                   default_model="instruct")
    name, _ = m._resolve_discussion_model(cfg, "coder")
    assert name == "coder"


def test_stub_config_has_three_tiers(home):
    cfg = m.make_stub_config()
    assert "coder" in cfg.models
    assert "instruct" in cfg.models
    assert "nitro" in cfg.models
