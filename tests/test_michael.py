"""Tests for michael's internals: slugs, projects, paths, staging, trash, replay."""
from __future__ import annotations

import pathlib
import shutil

import pytest

import main as m
import michael.globals as michael_globals


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Patch all michael path globals to live under tmp_path/.michael."""
    state = tmp_path / ".michael"
    monkeypatch.setattr(michael_globals, "STATE_DIR", state)
    monkeypatch.setattr(michael_globals, "GLOBAL_CONFIG_PATH", state / "config.json")
    monkeypatch.setattr(michael_globals, "GLOBAL_EVENTS_PATH", state / "events.jsonl")
    monkeypatch.setattr(michael_globals, "STATE_FILE_PATH", state / "state.json")
    monkeypatch.setattr(michael_globals, "PROJECTS_DIR", state / "projects")
    monkeypatch.setattr(michael_globals, "REPL_HISTORY_PATH", state / "repl_history")
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


def test_slugify_empty_raises():
    with pytest.raises(michael_globals.MichaelError):
        m.slugify("")
    with pytest.raises(michael_globals.MichaelError):
        m.slugify("///")


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
    real_root = workspace.resolve()
    ext_root = stage.parent / "_ext"
    ext_root.mkdir(exist_ok=True)
    try:
        m._apply_in_staging(
            "write_file",
            {"path": "src/bar.py", "content": "y = 2\n"},
            stage,
            real_root,
            ext_root,
        )
        assert (stage / "src" / "bar.py").read_text() == "y = 2\n"
        assert not (workspace / "src" / "bar.py").exists()
    finally:
        shutil.rmtree(stage.parent, ignore_errors=True)


def test_apply_in_staging_refuses_central_fs(home, workspace):
    """Writing to ~/.michael/ must be blocked regardless of staging."""
    p = m.create_project("x", workspace)
    stage = m._stage_project(p)
    real_root = workspace.resolve()
    ext_root = stage.parent / "_ext"
    ext_root.mkdir(exist_ok=True)
    central = str(michael_globals.STATE_DIR / "evil.txt")
    try:
        with pytest.raises(m.MichaelError, match="Central FS violation"):
            m._apply_in_staging(
                "write_file",
                {"path": central, "content": "x"},
                stage,
                real_root,
                ext_root,
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
    cfg.models["god"].vast_instance_id = "12345"
    cfg.save()
    assert m._config_is_unset() is False


# ---- single-model helpers -----------------------------------------------

def test_full_toolset_always_available():
    full = {t["function"]["name"] for t in m.TOOLS}
    assert "write_file" in full
    assert "run_shell" in full
    assert "read_file" in full


def test_stub_config_has_single_god_profile():
    cfg = m.make_stub_config()
    assert "god" in cfg.models
    assert cfg.default_model == "god"
    assert len(cfg.models) == 1


def test_get_model_returns_god_by_default(home):
    cfg = m.make_stub_config()
    cfg.save()
    loaded = m.Config.load()
    name, profile = loaded.get_model()
    assert name == "god"


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


def test_execute_with_staging_mismatch_rolls_back_and_errors(home, workspace):
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
    # Mismatch is an error: rolled back, LLM told to re-propose.
    assert result.startswith("mismatch:")
    assert "predicted:" in result and "actual:" in result
    # Change was rolled back — not in change_log.
    assert len(pending.change_log) == 0
    # Real workspace untouched.
    assert not (workspace / "src" / "bar.py").exists()
    events = m.iter_events(p.events_path)
    types = [e.get("type") for e in events]
    assert "tool.delta_mismatch" in types
    assert "tool.staged" not in types


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


# ---- Header 4 / build_protocol ------------------------------------------

def test_build_protocol_lists_four_headers():
    text = m.build_protocol()
    for h in ("H1", "H2", "H3", "H4"):
        assert h in text



def test_build_header_includes_protocol(home, workspace):
    p = m.create_project("x", workspace)
    pkg = m.build_header(p, "system stub")
    assert "H4: Protocol" in pkg
    assert "H1:" in pkg and "H2:" in pkg and "H3:" in pkg


# ---- REPL surface --------------------------------------------------------

def test_repl_commands_include_core_commands():
    assert "run" in m.REPL_COMMANDS
    assert "new" in m.REPL_COMMANDS
    assert "up" in m.REPL_COMMANDS
    assert "down" in m.REPL_COMMANDS


# ---- workbench -----------------------------------------------------------

import michael.workbench as wb


def test_workbench_project_root_outside_context_raises():
    with pytest.raises(RuntimeError, match="no active project"):
        wb.project_root()


def test_workbench_project_root_inside_context(home, workspace):
    p = m.create_project("wb-test", workspace)
    token = wb._set_context(p)
    try:
        assert wb.project_root() == pathlib.Path(p.path)
        assert wb.project_slug() == p.slug
    finally:
        wb._reset_context(token)


def test_workbench_context_cleared_after_reset(home, workspace):
    p = m.create_project("wb-reset", workspace)
    token = wb._set_context(p)
    wb._reset_context(token)
    with pytest.raises(RuntimeError, match="no active project"):
        wb.project_root()


def test_workbench_read_file_blocks_central_fs(home, workspace):
    p = m.create_project("wb-perm", workspace)
    token = wb._set_context(p)
    try:
        central_path = str(michael_globals.STATE_DIR / "secret.txt")
        with pytest.raises(m.MichaelError, match="Central FS violation"):
            wb.read_file(central_path)
    finally:
        wb._reset_context(token)


def test_workbench_run_shell_blocks_central_fs_reference(home, workspace):
    p = m.create_project("wb-shell", workspace)
    token = wb._set_context(p)
    try:
        with pytest.raises(m.MichaelError):
            wb.run_shell("cat ~/.michael/config.json")
    finally:
        wb._reset_context(token)


# ---- appmodel ------------------------------------------------------------

import michael.appmodel as am


def test_appmodel_save_and_load(home, workspace):
    p = m.create_project("am-test", workspace)
    model = am.make_model(
        "testapp", "v1",
        base_url="https://api.example.com",
        auth={"type": "bearer"},
        notes="test model",
    )
    am.save_model(p, model)
    loaded = am.load_model(p, "testapp", "v1")
    assert loaded.name == "testapp"
    assert loaded.version == "v1"
    assert loaded.base_url == "https://api.example.com"
    assert loaded.auth == {"type": "bearer"}
    assert loaded.notes == "test model"


def test_appmodel_list_returns_all(home, workspace):
    p = m.create_project("am-list", workspace)
    am.save_model(p, am.make_model("app-a", "1.0"))
    am.save_model(p, am.make_model("app-b", "2.0"))
    models = am.list_models(p)
    assert {mo.name for mo in models} == {"app-a", "app-b"}


def test_appmodel_missing_raises(home, workspace):
    p = m.create_project("am-miss", workspace)
    with pytest.raises(m.MichaelError, match="no model"):
        am.load_model(p, "ghost", "v0")
