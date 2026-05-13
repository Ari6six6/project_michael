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


MAX_AGENT_CYCLES = 9

ROOMS: list[dict] = [
    {
        "name":      "room.epistemics",
        "label":     "ROOM 1 — WHAT CAN I KNOW?",
        "directive": (
            "ROOM 1: EXPLORATION ONLY. Question: What can I know?\n"
            "Available tools: read_file, list_dir, search_memory — no writes.\n"
            "Map the full state: files, prior history, constraints, open questions.\n"
            "Signal Ja when you have complete epistemic clarity."
        ),
        "nudge": "Room 1: keep exploring. No writes yet. Signal Ja when the full picture is clear.",
    },
    {
        "name":      "room.ethics",
        "label":     "ROOM 2 — WHAT SHOULD I DO?",
        "directive": (
            "ROOM 2: BUILDING. Question: What should I do?\n"
            "Full tool access. Implement the smallest correct action. Test before signalling done.\n"
            "You have two toolboxes — check them before building anything new:\n"
            "  • tools/          — tools for this project only\n"
            "  • ~/.michael/toolbox/ — global tools, available in every project\n"
            "If a required capability is missing, write it as a Python file to one of those paths. "
            "Export TOOL_SCHEMA (OpenAI function schema dict) and a callable with the same name. "
            "It loads as a real tool available to every room in the next cycle.\n"
            "Use the global toolbox for general-purpose tools (port scanners, parsers, etc.) "
            "and the local tools/ for project-specific ones.\n"
            "Signal Ja when the implementation is verified."
        ),
        "nudge": (
            "Room 2: build, test, refine. Check tools/ and ~/.michael/toolbox/ before "
            "writing new ones. Signal Ja when done."
        ),
    },
    {
        "name":      "room.teleology",
        "label":     "ROOM 3 — WHAT CAN I HOPE FOR?",
        "directive": (
            "ROOM 3: PLOTTING. Question: What can I hope for?\n"
            "Room 2 has built or extended the toolbox. Now plot the next cycle:\n"
            "Which tools should Room 1 call to gather the remaining unknowns? "
            "What targets, what parameters, what data is still missing before the object of "
            "desire can be enclosed?\n"
            "Be concrete: name the tools, name the targets, name the open questions. "
            "This plot becomes the agenda for the next Room 1.\n"
            "Signal Ja when the next cycle has a clear agenda."
        ),
        "nudge": (
            "Room 3: make the next cycle's agenda concrete — tools, targets, unknowns. "
            "Signal Ja when the plot is set."
        ),
    },
    {
        "name":      "room.completion",
        "label":     "ROOM 4 — IS THE GOAL MET?",
        "directive": (
            "ROOM 4: COMPLETION GATE. Question: Have you enclosed the object of desire?\n"
            "The object of desire is defined in the user's original prompt. "
            "Ask yourself: can I answer that prompt with yes, right now, with certainty?\n"
            "If YES: end your response with Ja.\n"
            "If NO: state exactly what is still unknown or unbuilt. Do NOT say Ja."
        ),
        "nudge": "Room 4: have you enclosed the object of desire? Ja if yes. State what's missing if no.",
    },
]


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
