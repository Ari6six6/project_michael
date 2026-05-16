"""Utility SDK for LLM-written dynamic tools.

Import in a tool file to get safe, context-aware access to the active project:

    from michael.workbench import project_root, project_slug, read_file, list_dir, run_shell
"""
from __future__ import annotations

import contextvars
import pathlib
import subprocess
from typing import Optional

import michael.globals as G
from michael import permissions
from michael.project import Project

# Context variable set by _dispatch_dynamic_tool before calling tool functions.
_ctx: contextvars.ContextVar[Optional[Project]] = contextvars.ContextVar(
    "michael_workbench_project", default=None
)


# ---------------------------------------------------------------------------
# Internal API — called only by tools.py dispatcher
# ---------------------------------------------------------------------------

def _set_context(project: Project) -> contextvars.Token:
    return _ctx.set(project)


def _reset_context(token: "contextvars.Token[Optional[Project]]") -> None:
    _ctx.reset(token)


# ---------------------------------------------------------------------------
# Public API — for use inside dynamic tool files
# ---------------------------------------------------------------------------

def _require_project() -> Project:
    p = _ctx.get()
    if p is None:
        raise RuntimeError(
            "workbench: no active project — only call workbench functions "
            "from within a dynamic tool invocation"
        )
    return p


def project_root() -> pathlib.Path:
    """Absolute path to the active project's root directory."""
    return pathlib.Path(_require_project().path)


def project_slug() -> str:
    """Slug (identifier) of the active project."""
    return _require_project().slug


def read_file(path: str) -> str:
    """Read a file; path may be relative to project root or absolute.

    Raises MichaelError if the path resolves into the Central FS.
    """
    p = permissions.resolve_any(path, project_root=project_root())
    permissions.assert_not_central(p, "read")
    return p.read_text(errors="replace")


def list_dir(path: str = ".") -> list[str]:
    """List entries of a directory; path relative to project root or absolute.

    Raises MichaelError if the path resolves into the Central FS.
    """
    p = permissions.resolve_any(path, project_root=project_root())
    permissions.assert_not_central(p, "read")
    if not p.is_dir():
        raise FileNotFoundError(f"not a directory: {p}")
    return sorted(entry.name for entry in p.iterdir())


def run_shell(cmd: str, timeout: int = 60) -> str:
    """Run a shell command in the project root. Returns combined stdout+stderr.

    Raises MichaelError if the command references the Central FS.
    """
    err = permissions.check_shell_cmd(cmd)
    if err:
        raise G.MichaelError(err)
    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(project_root()),
    )
    return ((result.stdout or "") + (result.stderr or "")).strip()
