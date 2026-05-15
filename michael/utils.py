"""Filesystem snapshot and context-package builder (headers H1–H4)."""
from __future__ import annotations

import os
import pathlib
import re
from typing import TYPE_CHECKING, Any

import michael.globals as G
from michael.project import iter_events

if TYPE_CHECKING:
    from michael.project import Project


# ---------------------------------------------------------------------------
# Filesystem snapshot
# ---------------------------------------------------------------------------


def _is_text(path: pathlib.Path, sniff: int = 8192) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(sniff)
    except OSError:
        return False
    if b"\x00" in chunk:
        return False
    try:
        chunk.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def filesystem_snapshot(root: pathlib.Path) -> str:
    """Listing of the project tree + inlined contents for small text files."""
    root = root.resolve()
    listing_lines: list[str] = []
    text_files: list[tuple[pathlib.Path, int]] = []

    if not root.is_dir():
        return f"(project root does not exist: {root})"

    for dp, dirs, files in os.walk(root):
        dp_path = pathlib.Path(dp)
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in G.SKIP_DIRS]
        rel_dp = dp_path.relative_to(root)
        for fname in sorted(files):
            f = dp_path / fname
            try:
                size = f.stat().st_size
            except OSError:
                continue
            rel = (rel_dp / fname).as_posix() if str(rel_dp) != "." else fname
            listing_lines.append(f"{rel} ({size}b)")
            if size <= G.MAX_FILE_BYTES_INLINE and _is_text(f):
                text_files.append((f, size))

    parts: list[str] = []
    parts.append("Directory listing (relative to project root):")
    parts.append("\n".join(listing_lines) if listing_lines else "(empty)")
    parts.append("")
    parts.append(
        f"File contents (text only; per-file cap {G.MAX_FILE_BYTES_INLINE}b, "
        f"total cap {G.MAX_TOTAL_BYTES_INLINE}b):"
    )

    text_files.sort(key=lambda x: x[1])
    bodies: list[str] = []
    total = 0
    for f, size in text_files:
        rel = f.relative_to(root).as_posix()
        if total + size > G.MAX_TOTAL_BYTES_INLINE:
            bodies.append(f"==== {rel} (skipped: total cap reached) ====")
            continue
        try:
            content = f.read_text(errors="replace")
        except OSError:
            continue
        bodies.append(f"==== {rel} ({size}b) ====\n{content}")
        total += size

    parts.append("\n\n".join(bodies) if bodies else "(no text files inlined)")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# H1 / H3 history builders
# ---------------------------------------------------------------------------


def _prompt_history_lines(project: "Project") -> list[str]:
    out: list[str] = []
    n = 0
    for ev in iter_events(project.events_path):
        if ev.get("type") == "prompt.sent":
            n += 1
            prompt = (ev.get("payload") or {}).get("prompt", "")
            out.append(f"[{n}] {prompt}")
    return out


def _action_log_lines(project: "Project") -> list[str]:
    out: list[str] = []
    n = 0
    for ev in iter_events(project.events_path):
        t = ev.get("type", "")
        p = ev.get("payload", {}) or {}
        if t == "tool.executed":
            n += 1
            line = f"[{n}] {p.get('summary', t)}"
            brief = p.get("brief_result", "")
            if brief:
                first_lines = "\n    ".join(brief.splitlines()[:4])
                line += f"\n    {first_lines}"
            out.append(line)
        elif t == "tool.rejected":
            n += 1
            out.append(f"[{n}] {p.get('summary', t)}  [REJECTED BY USER]")
        elif t == "tool.verify_failed":
            n += 1
            rc = p.get("verify_rc", "?")
            out.append(
                f"[{n}] {p.get('summary', t)}  [VERIFY FAILED rc={rc}, user not prompted]"
            )
        elif t == "tool.undone":
            n += 1
            out.append(
                f"[{n}] undone: {p.get('tool', '?')} ({p.get('trash_id', '?')})"
            )
    return out


# ---------------------------------------------------------------------------
# H4: Protocol Bible
# ---------------------------------------------------------------------------


def build_protocol() -> str:
    """Header 4 — the protocol."""
    return "\n".join([
        "You are connected to the user's machine through Project Michael.",
        "Michael is event-sourced: every user prompt and every tool call you",
        "execute is logged. You are amnesiac across user prompts; the package",
        "below is your entire memory of this project.",
        "",
        "PACKAGE STRUCTURE (sent on every fresh instance):",
        "  H1 — User's prompts in this project, verbatim and in order. The",
        "       user's formal/technical language is the source of truth; do",
        "       not re-derive intent from your own past output.",
        "  H2 — Filesystem snapshot of the project workspace as of NOW.",
        "  H3 — Every tool call you have executed in this project, with",
        "       outcomes. This is your causal chain.",
        "  H4 — This protocol. The contract you operate under.",
        "",
        "FULL TOOL ACCESS:",
        "You have all tools from the start. Explore and build freely in whatever",
        "order makes sense. There are no phases, no mode restrictions.",
        "",
        "FILESYSTEM ZONES:",
        "Two zones exist on this machine.",
        "",
        "  Central FS (~/.michael/) — READ-ONLY to you. This is Michael's",
        "  internal state: event logs, config, endpoint cache, project",
        "  metadata. You may read_file inside it to diagnose issues, but",
        "  write_file, apply_patch, and any run_shell command referencing",
        "  this path are blocked at the tool layer.",
        "",
        "  Work FS (everything else) — Unrestricted. write_file and",
        "  apply_patch accept any absolute path or project-relative path",
        "  outside ~/.michael/. run_shell has full system access except for",
        "  commands referencing ~/.michael/ which are blocked.",
        "",
        "STAGING:",
        "write_file and apply_patch write to a staging copy — nothing touches",
        "the real workspace until you call commit_changes(). You MUST include",
        "`expected_changes` on every write: your prediction of which paths will",
        "be added, modified, or removed. Michael computes the actual delta and",
        "feeds prediction vs reality back to you. Mismatch is information, not",
        "failure — read it and decide what to do next.",
        "",
        "COMMITTING:",
        "When your work is complete and you are satisfied, call",
        "commit_changes(summary='...') to apply all staged changes to disk.",
        "Do NOT call it until the goal is fully met. If you finish without",
        "staging any changes (e.g. an informational task), just respond — the",
        "loop exits naturally with nothing committed.",
        "",
        "SANDBOX:",
        "Use run_in_sandbox to test code in an isolated podman container before",
        "writing it. run_shell runs in the project workspace without sandboxing.",
        "Both require user confirmation.",
        "",
        "LONG-TERM MEMORY:",
        "Call search_memory(query) to retrieve context from previous sessions —",
        "what you explored, what the sandbox returned, what failed. Use it early",
        "before re-discovering what you already know.",
        "",
        "TOOLBOX STEWARDSHIP:",
        "tools/ (project-local) and ~/.michael/toolbox/ (global) are your growing",
        "capability set. Every run is an opportunity to leave them better than you",
        "found them. This is not optional scaffolding — it is how Michael compounds",
        "capability across sessions.",
        "",
        "The rule: if you reached for something that didn't exist and had to inline",
        "the logic, that logic belongs in a tool. Write it before calling",
        "commit_changes(). Export TOOL_SCHEMA (OpenAI function schema dict) and a",
        "callable with the same name. It auto-loads immediately — no restart needed.",
        "",
        "General-purpose tools go in ~/.michael/toolbox/ — available in every",
        "project. Project-specific tools go in tools/ — local only.",
        "A tool is worth writing if you can imagine calling it again on a different",
        "prompt. If it's truly one-off, inline is fine. Use judgment.",
        "",
        "Tools (full schemas in the API call):",
        "  write_file(path, content, expected_changes)        stages a file write",
        "  apply_patch(path, unified_diff, expected_changes)  stages a patch",
        "  commit_changes(summary)                            applies all staged changes — call when done",
        "  read_file(path)                                    auto-executes",
        "  list_dir(path='.')                                 auto-executes",
        "  search_memory(query)                               auto-executes",
        "  run_in_sandbox(python_code)                        isolated podman, requires confirmation",
        "  run_shell(cmd, timeout_s=60)                       project workspace, requires confirmation",
        "",
        "All paths are relative to the project root. Do not escape with '..'.",
    ])


def load_scripture(scripture_dir: str) -> str:
    """Read all text files from scripture_dir and return concatenated content."""
    p = pathlib.Path(scripture_dir).expanduser()
    if not p.is_dir():
        return ""
    parts: list[str] = []
    for f in sorted(p.iterdir()):
        if f.is_file() and _is_text(f):
            try:
                parts.append(f"--- {f.name} ---\n{f.read_text(errors='replace')}")
            except OSError:
                continue
    return "\n\n".join(parts)


_TOOL_NAME_RE = re.compile(r'"name"\s*:\s*"([^"]+)"')


def _toolbox_listing(project_path: str) -> str:
    """Summarise available dynamic tools across all three toolbox directories."""
    def _scan(d: pathlib.Path) -> list[str]:
        if not d.is_dir():
            return []
        names: list[str] = []
        for f in sorted(d.glob("*.py")):
            try:
                text = f.read_text(errors="replace")
            except OSError:
                continue
            if "TOOL_SCHEMA" not in text:
                continue
            m = _TOOL_NAME_RE.search(text)
            names.append(m.group(1) if m else f.stem)
        return names

    bundled = pathlib.Path(__file__).parent.parent / "toolbox"
    global_box = G.GLOBAL_TOOLS_DIR
    project_box = pathlib.Path(project_path) / "tools"

    lines = ["Toolbox (dynamic tools available to you):"]
    for label, path in [
        ("bundled toolbox/", bundled),
        ("global ~/.michael/toolbox/", global_box),
        ("project tools/", project_box),
    ]:
        names = _scan(path)
        entry = ", ".join(names) if names else "(empty)"
        lines.append(f"  {label}: {entry}")
    lines.append(
        "  Write a .py file to project tools/ or ~/.michael/toolbox/ "
        "exporting TOOL_SCHEMA + a callable to add a new tool."
    )
    return "\n".join(lines)


def build_header(
    project: "Project",
    system_prompt: str,
    scripture: str = "",
) -> str:
    """Pack the four-header context package sent to a fresh LLM instance."""
    prompts = _prompt_history_lines(project)
    actions = _action_log_lines(project)
    snap = filesystem_snapshot(pathlib.Path(project.path))
    protocol = build_protocol()
    toolbox = _toolbox_listing(project.path)

    parts = [
        system_prompt,
        "",
        "=== H4: Protocol ===",
        protocol,
        "",
        "=== Toolbox ===",
        toolbox,
        "",
    ]
    if scripture:
        parts += [
            "=== Scripture ===",
            scripture,
            "",
        ]
    parts += [
        "=== Project ===",
        f"Name: {project.name}",
        f"Slug: {project.slug}",
        f"Root: {project.path}",
        "",
        "=== H1: User's prompts in this project (verbatim, in order) ===",
        "\n".join(prompts) if prompts else "(this is the user's first prompt)",
        "",
        "=== H3: Tool calls executed in this project (in order) ===",
        "\n".join(actions) if actions else "(none yet)",
        "",
        "=== H2: Filesystem snapshot ===",
        snap,
    ]
    return "\n".join(parts)
