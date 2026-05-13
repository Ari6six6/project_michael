"""
Dual filesystem permission model for Project Michael.

Central FS  (~/.michael/) — read-only to LLM tool calls.
                            Michael's own application code writes here freely.
Work FS     (everything else) — unrestricted read/write for LLM tool calls.

Enforcement is at the Python tool-dispatch layer — no I/O to the Central FS
path ever happens through an LLM tool call.
"""
from __future__ import annotations

import re
from pathlib import Path

from michael import globals as G

# Matches shell shorthand forms that could appear in a run_shell command.
# The absolute resolved path is checked dynamically (see check_shell_cmd).
_SHELL_SHORTHAND_RE = re.compile(r"(?:~|\$\{?HOME\}?)/\.michael\b")


def _central_fs_root() -> Path:
    """Return the current Central FS root (resolved).

    Evaluated on every call so that test monkeypatching of
    ``michael.globals.STATE_DIR`` takes effect correctly.
    """
    return Path(G.STATE_DIR).resolve()


# Keep a module-level alias for code that needs a stable reference
# (e.g. CLAUDE.md docs). The authoritative value at runtime always comes
# from _central_fs_root().
CENTRAL_FS_ROOT: Path = Path(G.STATE_DIR).resolve()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def resolve_any(path_str: str, project_root: Path) -> Path:
    """Resolve *path_str* to an absolute Path.

    Accepts absolute paths, ``~/``-prefixed paths, and paths relative to
    *project_root*.  Does **not** enforce any zone restriction — callers that
    need protection must follow up with :func:`assert_not_central`.
    """
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = project_root / p
    return p.resolve()


def is_project_path(abs_path: Path, project_root: Path) -> bool:
    """Return True iff *abs_path* is inside *project_root*."""
    try:
        abs_path.relative_to(project_root.resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Permission checks
# ---------------------------------------------------------------------------


def assert_not_central(path: Path, op: str = "write") -> None:
    """Raise :class:`~michael.globals.MichaelError` if *path* is inside the
    Central FS root.

    Call this before any LLM-initiated I/O to guarantee the Central FS is
    never touched by tool calls.

    Exception: the global toolbox (``~/.michael/toolbox/``) is explicitly
    writable — it is the one sub-path of the Central FS where the LLM may
    persist tools for cross-project reuse.
    """
    root = _central_fs_root()
    try:
        path.relative_to(root)
    except ValueError:
        return  # outside Central FS — allowed

    # Narrow carve-out: the global toolbox is writable by LLM tool calls.
    global_toolbox = Path(G.GLOBAL_TOOLS_DIR).resolve()
    try:
        path.relative_to(global_toolbox)
        return  # inside the toolbox — allowed
    except ValueError:
        pass

    raise G.MichaelError(
        f"Central FS violation [{op}]: {path} is inside {root}.\n"
        "Michael's internal state directory is read-only to LLM tool calls.\n"
        "Only Michael's own application code may write there.\n"
        f"(Exception: {global_toolbox} is writable for cross-project tools.)"
    )


def check_shell_cmd(cmd: str) -> str | None:
    """Return an error string if *cmd* references the Central FS, else None.

    Used as a heuristic pre-flight check for ``run_shell`` commands.  It
    catches the most common forms (absolute path, ``~/.michael``,
    ``$HOME/.michael``) without attempting to parse arbitrary shell syntax.
    """
    root = _central_fs_root()
    if _SHELL_SHORTHAND_RE.search(cmd) or str(root) in cmd:
        return (
            f"Blocked: command references the central filesystem "
            f"({root}).  LLM shell commands cannot touch that path."
        )
    return None
