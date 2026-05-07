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


def test_protocol_for_mode_branches_distinct():
    code = m._protocol_for_mode("code")
    discussion = m._protocol_for_mode("discussion")
    nitro = m._protocol_for_mode("nitro")
    assert "code" in code.lower() and "expected_changes" in code
    assert "read-only" in discussion or "read_file" in discussion
    assert "nitro" in nitro.lower() and "heavy" in nitro.lower()


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


# ---- predicted-delta gate ------------------------------------------------

def test_execute_with_staging_missing_expected_returns_error_to_llm(home, workspace):
    p = m.create_project("x", workspace)
    cfg = m.Config()
    result = m.execute_with_staging(
        "write_file",
        {"path": "src/bar.py", "content": "y = 2\n"},
        p, cfg,
    )
    assert result.startswith("error: expected_changes is required")
    # The user's workspace must be untouched.
    assert not (workspace / "src" / "bar.py").exists()
    # An audit event should have been logged.
    events = m.iter_events(p.events_path)
    types = [e.get("type") for e in events]
    assert "tool.delta_missing" in types


def test_execute_with_staging_mismatch_returns_error_to_llm(home, workspace):
    p = m.create_project("x", workspace)
    cfg = m.Config()
    # LLM predicts a different file than what the write actually changes.
    result = m.execute_with_staging(
        "write_file",
        {
            "path": "src/bar.py",
            "content": "y = 2\n",
            "expected_changes": ["src/wrong.py"],
        },
        p, cfg,
    )
    assert result.startswith("error: predicted-delta mismatch")
    assert "src/bar.py" in result  # actual delta surfaced
    assert not (workspace / "src" / "bar.py").exists()
    events = m.iter_events(p.events_path)
    types = [e.get("type") for e in events]
    assert "tool.delta_mismatch" in types


def test_execute_with_staging_match_prompts_user_and_applies(home, workspace, monkeypatch):
    p = m.create_project("x", workspace)
    cfg = m.Config()
    # Force the user-confirm prompt to "yes".
    monkeypatch.setattr(m.typer, "prompt", lambda *a, **k: "y")
    result = m.execute_with_staging(
        "write_file",
        {
            "path": "src/bar.py",
            "content": "y = 2\n",
            "expected_changes": ["src/bar.py"],
        },
        p, cfg,
    )
    assert result.startswith("applied;")
    assert (workspace / "src" / "bar.py").read_text() == "y = 2\n"


# ---- REPL surface --------------------------------------------------------

def test_repl_commands_include_nitro_and_new_subcommands():
    assert "nitro" in m.REPL_COMMANDS
    assert "new" in m.REPL_COMMANDS
    assert set(m.NEW_SUBCOMMANDS) == {"project", "code", "discussion"}
