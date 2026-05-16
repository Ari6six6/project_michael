"""Project model, active-project state, and the event log."""
from __future__ import annotations

import fcntl
import json
import os
import pathlib
import re
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import michael.globals as G


# ---------------------------------------------------------------------------
# Event types (append_event and iter_events)
# ---------------------------------------------------------------------------
#
# Known event types (non-exhaustive):
#
# Project lifecycle:
#   project.created — project initialized
#   project.activated — project set as active
#
# Agent loop:
#   prompt.sent — user entered a prompt
#   assistant.message — LLM generated a response (full text if log_responses=true)
#   assistant.ja — LLM signaled completion with "Ja" passcode
#
# Tool execution:
#   tool.staged — tool call staged (write_file, apply_patch)
#   tool.executed — staged changes committed to real filesystem
#   tool.rejected — user rejected staged changes
#   tool.verify_failed — verification script failed in staging
#   tool.delta_mismatch — actual delta didn't match expected_changes prediction
#
# Kantian machine (stateful):
#   scripture.loaded — scripture files loaded
#   scripture.interpreted — LLM interpreted scripture (Turn 1)
#   target.formulated — LLM formulated target/goal/constraints (Turn 2)
#   kantian.iteration — LLM iterated through Kantian questions (Turn 3+)
#
# Instance management:
#   instance.start_requested, instance.started, instance.stop_requested, instance.stopped
#   instance.poll — periodic polling of Vast.ai instance status


# ---------------------------------------------------------------------------
# Project model
# ---------------------------------------------------------------------------


@dataclass
class Project:
    slug: str
    name: str
    path: str
    created_at: str

    @classmethod
    def load(cls, slug: str) -> "Project":
        cfg = G.PROJECTS_DIR / slug / "config.json"
        if not cfg.is_file():
            raise G.MichaelError(f"unknown project: {slug}")
        data = json.loads(cfg.read_text())
        try:
            return cls(**{k: data[k] for k in cls.__dataclass_fields__})
        except KeyError as e:
            raise G.MichaelError(f"project {slug} config missing key: {e}") from e

    def save(self) -> None:
        d = G.PROJECTS_DIR / self.slug
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        (d / "config.json").write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True)
        )

    @property
    def events_path(self) -> pathlib.Path:
        return G.PROJECTS_DIR / self.slug / "events.jsonl"


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    s = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not s:
        raise G.MichaelError("project name must contain at least one letter or digit")
    return s[:64]


def list_projects() -> list[Project]:
    if not G.PROJECTS_DIR.is_dir():
        return []
    out: list[Project] = []
    for d in sorted(G.PROJECTS_DIR.iterdir()):
        cfg = d / "config.json"
        if not cfg.is_file():
            continue
        try:
            data = json.loads(cfg.read_text())
            out.append(Project(**{k: data[k] for k in Project.__dataclass_fields__}))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return out


def create_project(name: str, path: pathlib.Path) -> Project:
    base = slugify(name)
    slug = base
    n = 2
    existing = {p.slug for p in list_projects()}
    while slug in existing:
        slug = f"{base}-{n}"
        n += 1
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    proj = Project(
        slug=slug,
        name=name,
        path=str(path),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    proj.save()
    append_event("project.created", {"slug": slug, "name": name, "path": str(path)})
    return proj


# ---------------------------------------------------------------------------
# Active-project state
# ---------------------------------------------------------------------------


def get_active_slug() -> Optional[str]:
    if not G.STATE_FILE_PATH.is_file():
        return None
    try:
        return json.loads(G.STATE_FILE_PATH.read_text()).get("active_project")
    except json.JSONDecodeError:
        return None


def set_active_slug(slug: Optional[str]) -> None:
    G.STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    G.STATE_FILE_PATH.write_text(json.dumps({"active_project": slug}, indent=2))
    os.chmod(G.STATE_FILE_PATH, 0o600)


def get_active_project() -> Optional[Project]:
    s = get_active_slug()
    if not s:
        return None
    try:
        return Project.load(s)
    except G.MichaelError:
        return None


def require_active_project() -> Project:
    p = get_active_project()
    if not p:
        raise G.MichaelError("no active project — run `new project` or `use <slug>`")
    return p


# ---------------------------------------------------------------------------
# Event log (global + per-project)
# ---------------------------------------------------------------------------


def _last_seq(path: pathlib.Path) -> int:
    if not path.is_file():
        return 0
    last: Optional[str] = None
    with path.open("rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        tail = f.readline().decode("utf-8", errors="replace").strip()
        if tail:
            last = tail
    if not last:
        return 0
    try:
        return int(json.loads(last).get("seq", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def _append(path: pathlib.Path, type_: str, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.exists():
        path.touch(mode=0o600)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            seq = _last_seq(path) + 1
            event = {
                "seq": seq,
                "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "type": type_,
                "payload": payload,
            }
            line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return event


def append_event(
    type_: str,
    payload: dict[str, Any],
    *,
    project: Optional[Project] = None,
) -> dict[str, Any]:
    if project is not None:
        path = project.events_path
        scope = f"project:{project.slug}"
    else:
        path = G.GLOBAL_EVENTS_PATH
        scope = "global"
    ev = _append(path, type_, payload)
    style = "red" if type_ == "error" else "dim"
    G.console.print(f"[{style}]· {ev['seq']:>4} {type_} ({scope})[/]", highlight=False)
    return ev


def iter_events(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


# ---------------------------------------------------------------------------
# Tool catalog — global registry of delivered tools
# ---------------------------------------------------------------------------


def load_catalog() -> dict[str, Any]:
    """Load the global tools catalog from disk."""
    if not G.TOOLS_CATALOG_PATH.is_file():
        return {}
    try:
        return json.loads(G.TOOLS_CATALOG_PATH.read_text())
    except json.JSONDecodeError:
        return {}


def save_catalog(catalog: dict[str, Any]) -> None:
    G.STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    G.TOOLS_CATALOG_PATH.write_text(json.dumps(catalog, indent=2, sort_keys=True))


def detect_deliverable(project: "Project") -> Optional[tuple[str, str]]:
    """Return (rel_path, run_cmd) for the best deliverable in the project, or None."""
    root = pathlib.Path(project.path)
    if not root.is_dir():
        return None

    for name in ("main.py", "app.py", "cli.py", "tool.py", "__main__.py"):
        f = root / name
        if f.is_file():
            return name, f"python {f}"

    for f in sorted(root.glob("*.py")):
        try:
            text = f.read_text(errors="replace")
        except OSError:
            continue
        if any(tok in text for tok in ("typer", "click", "argparse", "__main__")):
            return f.name, f"python {f}"

    for f in sorted(root.glob("*.sh")):
        return f.name, f"bash {f}"

    for f in sorted(root.iterdir()):
        if f.is_file() and os.access(f, os.X_OK) and not f.name.startswith("."):
            return f.name, str(f)

    return None


def register_deliverable(project: "Project", deliverable: str, run_cmd: str) -> None:
    """Register the deliverable in the catalog and auto-install a wrapper into workbench/bin."""
    G.MICHAEL_BIN_DIR.mkdir(parents=True, exist_ok=True)
    wrapper = G.MICHAEL_BIN_DIR / project.slug
    wrapper.write_text(f"#!/bin/bash\nexec {run_cmd} \"$@\"\n")
    wrapper.chmod(0o755)

    catalog = load_catalog()
    catalog[project.slug] = {
        "slug": project.slug,
        "name": project.name,
        "deliverable": deliverable,
        "run_cmd": run_cmd,
        "installed_as": str(wrapper),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_catalog(catalog)


def replay_global() -> dict[str, Any]:
    """Pure fold over the global event log → per-profile state."""
    state: dict[str, Any] = {
        "models": {},
        "errors": 0,
    }

    def m(name: str) -> dict[str, Any]:
        return state["models"].setdefault(
            name,
            {"instance_state": "unknown", "endpoint": None, "last_poll_ts": None},
        )

    for ev in iter_events(G.GLOBAL_EVENTS_PATH):
        t = ev.get("type", "")
        p = ev.get("payload", {}) or {}
        model = p.get("model", "")
        if t == "instance.start_requested" and model:
            m(model)["instance_state"] = "starting"
        elif t == "instance.started" and model:
            m(model)["instance_state"] = "running"
            if p.get("endpoint"):
                m(model)["endpoint"] = p["endpoint"]
        elif t == "instance.stop_requested" and model:
            m(model)["instance_state"] = "stopping"
        elif t == "instance.stopped" and model:
            m(model)["instance_state"] = "stopped"
        elif t == "instance.poll" and model:
            m(model)["last_poll_ts"] = ev.get("ts")
            if "actual_status" in p:
                m(model)["instance_state"] = p["actual_status"]
        elif t == "endpoint.discovered" and model:
            m(model)["endpoint"] = p.get("endpoint")
        elif t == "error":
            state["errors"] += 1
    return state
