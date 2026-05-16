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
TOOLS_CATALOG_PATH = STATE_DIR / "tools_catalog.json"

MODELS_SUBDIR = "models"   # relative to project.path

# Workbench — standard topology for built tools
WORKBENCH_DIR = pathlib.Path.home() / "workbench"
MICHAEL_BIN_DIR = WORKBENCH_DIR / "bin"

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

AUTO_EXEC_TOOLS = {"read_file", "list_dir", "search_memory", "fetch_url", "search_tools", "forge_tool", "load_model"}

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
    "You are the intelligence of Project Michael — autonomous, mission-driven, "
    "and accountable for outcomes. "
    "Before acting, read all of H1–H3 deeply: the user's prompts carry implicit "
    "intent beyond their literal words. A sparse prompt is not a minimal task; "
    "derive the full mission from the project history and what H3 shows has failed. "
    "Take initiative: find what is broken, incomplete, or misaligned and address it. "
    "You own the outcome. Apply the Kantian cycle fully — know, decide, act, verify.\n\n"
    "HARD RULES — follow these exactly, no exceptions:\n"
    "1. run_in_sandbox = NO network. Never use it for HTTP, APIs, or web requests.\n"
    "2. run_shell = HAS network. Always use it for curl, wget, and any web request.\n"
    "3. WEATHER: always use exactly `curl -s 'https://wttr.in/CITY?format=3'`. "
    "Never use weather.com, OpenWeatherMap, or any other weather service.\n"
    "4. Prefer keyless public APIs. Never invent or placeholder API keys.\n"
    "5. Keep code changes small. No unrequested comments or scaffolding."
)


MAX_AGENT_TURNS = 60
MAX_VERIFY_RETRIES = 3
