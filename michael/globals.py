"""Shared constants, path globals, and tiny utilities used across all modules.

All path variables (STATE_DIR etc.) are module-level so they can be patched in
tests via monkeypatch.setattr(michael.globals, "STATE_DIR", ...).
"""
from __future__ import annotations

import pathlib
from typing import Any

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

AUTO_EXEC_TOOLS = {"read_file", "list_dir", "search_memory", "browse_url", "save_concept", "list_concepts"}


def concept_dir(project: "Any") -> pathlib.Path:
    """Return the concept store directory for a project (Central FS, written by tool layer)."""
    return pathlib.Path(PROJECTS_DIR) / project.slug / "concept"

# ---------------------------------------------------------------------------
# Domain error
# ---------------------------------------------------------------------------


class MichaelError(RuntimeError):
    """Domain error surfaced to the user with a clean message."""


# ---------------------------------------------------------------------------
# Agent protocol constants
# ---------------------------------------------------------------------------

JA_PASSPHRASE = "Ja"

_GOD_MODE_PROMPT = (
    "Assess the full state of this project. "
    "Burn what is not working. "
    "Let stand what is righteous. "
    "Propose your changes."
)

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful coding assistant connected to the user's machine "
    "through Project Michael. Keep changes small and reviewable. Prefer "
    "editing existing files over creating new ones. Do not add unrequested "
    "comments, error handling, or scaffolding."
)


def _message_ends_with_ja(text: str) -> bool:
    """True iff the message's trailing token is the bareword JA_PASSPHRASE.

    Catches: 'thoughts.\\nJa', 'thoughts.\\nJa\\n', 'done. Ja.'
    Rejects: '', 'Ja, das ist gut', 'Yes', 'ja' (case-sensitive).
    """
    if not text:
        return False
    stripped = text.rstrip().rstrip(".!?;:")
    if not stripped:
        return False
    last_token = stripped.rsplit(None, 1)[-1]
    return last_token == JA_PASSPHRASE
