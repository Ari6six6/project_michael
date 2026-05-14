"""Shared constants, path globals, and tiny utilities used across all modules.

All path variables (STATE_DIR etc.) are module-level so they can be patched in
tests via monkeypatch.setattr(michael.globals, "STATE_DIR", ...).
"""
from __future__ import annotations

import pathlib

from rich.console import Console

# ---------------------------------------------------------------------------
# State-directory paths — patchable module-level variables
# ---------------------------------------------------------------------------

STATE_DIR = pathlib.Path.home() / ".michael"
GLOBAL_CONFIG_PATH = STATE_DIR / "config.json"
GLOBAL_EVENTS_PATH = STATE_DIR / "events.jsonl"
STATE_FILE_PATH = STATE_DIR / "state.json"
PROJECTS_DIR = STATE_DIR / "projects"
REPL_HISTORY_PATH = STATE_DIR / "repl_history"
GLOBAL_TOOLS_DIR = STATE_DIR / "toolbox"

# ---------------------------------------------------------------------------
# Shared Rich consoles
# ---------------------------------------------------------------------------

console = Console()
err = Console(stderr=True, style="bold red")

# ---------------------------------------------------------------------------
# Filesystem-snapshot tunables
# ---------------------------------------------------------------------------

MAX_FILE_BYTES_INLINE = 50_000
MAX_TOTAL_BYTES_INLINE = 500_000
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".idea", ".vscode", ".next",
    "target", ".cache",
}

# ---------------------------------------------------------------------------
# Tool routing
# ---------------------------------------------------------------------------

AUTO_EXEC_TOOLS = {"read_file", "list_dir", "search_memory"}

# ---------------------------------------------------------------------------
# Domain error
# ---------------------------------------------------------------------------


class MichaelError(RuntimeError):
    """Domain error surfaced to the user with a clean message."""


# ---------------------------------------------------------------------------
# Agent protocol constants
# ---------------------------------------------------------------------------

_GOD_MODE_PROMPT = (
    "Assess the full state of this project. "
    "Burn what is not working. "
    "Let stand what is righteous. "
    "Propose your changes."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a capable assistant connected to the user's machine through "
    "Project Michael. You can run shell commands, write files, and browse "
    "the web.\n\n"
    "Key rules:\n"
    "- Keep code changes small and reviewable.\n"
    "- Prefer editing existing files over creating new ones.\n"
    "- Do not add unrequested comments, error handling, or scaffolding.\n"
    "- run_in_sandbox has NO network — never use it for HTTP/API calls.\n"
    "- run_shell HAS network — use it for curl, wget, and all web requests.\n"
    "- For weather use: curl -s 'https://wttr.in/CITY?format=3' — no API key needed.\n"
    "- Prefer keyless public APIs (wttr.in, ip-api.com, etc.) over paid ones."
)


MAX_AGENT_TURNS = 60
