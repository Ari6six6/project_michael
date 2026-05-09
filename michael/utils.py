"""Filesystem snapshot and context-package builder (headers H1–H4)."""
from __future__ import annotations

import os
import pathlib
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


_MODE_ADDENDUM: dict[str, str] = {
    "code": (
        "MODE: code. Full toolset. write_file and apply_patch require "
        "`expected_changes`. Predict, propose, sandbox, review, refine. "
        "Surface to the user only with the Ja passcode."
    ),
    "discussion": (
        "MODE: discussion. You have read-only tools (read_file, list_dir). "
        "write_file, apply_patch, run_in_sandbox, and run_shell are NOT "
        "available — for code changes the user will start a `new code` or "
        "`nitro` session. End your message with the Ja passcode when you "
        "are ready for the user to read your reply."
    ),
    "nitro": (
        "MODE: nitro (heavy model). Same contract as code mode. The user "
        "is paying premium GPU time for this turn — be efficient with the "
        "loop, but do not skip estimation or the Ja gate."
    ),
    "god": (
        "MODE: god (heavy model, full authority). No user approval gate: when "
        "you emit the Ja passcode Michael will auto-commit every staged change "
        "immediately, without asking the user. Authority is fully granted. "
        "Assess the project in its entirety. Burn what is not working. Let "
        "stand what is righteous. This is a one-shot session — make it count. "
        "Do not skip estimation; the staging pipeline still runs."
    ),
}


def build_protocol(mode: str = "code") -> str:
    """Header 4 — the protocol Bible."""
    addendum = _MODE_ADDENDUM.get(mode, _MODE_ADDENDUM["code"])
    ja = G.JA_PASSPHRASE
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
        "NO HANDS:",
        "You propose; Michael executes. You cannot directly write to the",
        "user's filesystem or run shell commands on the host. Every tool call",
        "is a proposal that Michael stages, verifies, and reports back to you.",
        "",
        "ESTIMATION MANDATE:",
        "On write_file and apply_patch you MUST include `expected_changes` —",
        "your prediction of which project-relative paths will be added,",
        "modified, or removed. This is non-negotiable. Michael runs the",
        "proposal in staging, computes the actual delta, and feeds prediction",
        "vs reality back to you. Mismatch is information, not failure: read",
        "it and decide what to do next.",
        "",
        "THE BOMB FIELD (sandbox / VPS):",
        "Michael handles and detonates your estimates in the bomb field — a",
        "remote VPS running rootless podman, or local podman if no VPS is",
        "configured. Use run_in_sandbox to test code before proposing a",
        "write_file. The user's real workspace stays untouched until the Ja",
        "gate fires AND the user approves.",
        "",
        "INDEFINITE ITERATION:",
        "You and Michael iterate alone, in private. There is no turn budget.",
        "Propose, stage, sandbox, review, refine — as many rounds as you need.",
        "The user is not watching individual turns. The only ways out of the",
        "loop are the Ja passcode below, or a user-initiated abort (Ctrl-C).",
        "",
        "LONG-TERM MEMORY:",
        "Your past reasoning and tool results are stored. Call search_memory(query)",
        "when you need context from previous sessions — what you explored, what",
        "you concluded, what the sandbox returned, what failed. Use it early",
        "before re-discovering what you already know.",
        "",
        f"THE {ja!r} PASSCODE:",
        f"The user only sees your work when you END a message with the literal",
        f"bareword `{ja}` (case-sensitive, on its own line or as the",
        f"trailing token). That is the ONLY signal Michael reads as 'surface",
        f"this to the user.' Until {ja}, you are talking to Michael,",
        f"not the user.",
        "",
        f"Do NOT use {ja} casually. Do NOT use it mid-thought. Do",
        f"NOT use it as a filler word. {ja} means: 'I am done",
        f"iterating; this is the product I want the user to review.'",
        "",
        f"After {ja}, Michael shows the user the staged delta and",
        f"asks one yes/no question. Yes = the change is committed and this",
        f"prompt cycle ends. No = the staging is discarded; the next user",
        f"prompt re-enters the loop and you will see the rejection in H3.",
        "",
        "THE THREE KANTIAN QUESTIONS:",
        "When tasked with a problem, you iterate through three orthogonal and",
        "exhaustive dimensions of reasoning:",
        "",
        "1. WHAT CAN I KNOW? (Epistemics)",
        "   - What does the filesystem reveal about the codebase structure,",
        "     dependencies, and current state?",
        "   - What tools do I have available and what are their limits?",
        "   - What are the constraints (sandbox limits, network access,",
        "     timeouts, resource caps)?",
        "   - What errors or warnings did previous attempts produce?",
        "",
        "2. WHAT SHOULD I DO? (Ethics / Imperative)",
        "   - What is the user's explicit intent? What is implicit?",
        "   - What follows from the inherent logic of the problem?",
        "   - What is the smallest, most correct, most reversible action?",
        "   - Does my proposal align with the protocol and the system prompt?",
        "",
        "3. WHAT CAN I HOPE FOR? (Teleology)",
        "   - Is the target achievable with available tools, time, and budget?",
        "   - What is the success criterion? How will I verify correctness?",
        "   - What might go wrong? What is the blast radius if this fails?",
        "   - Can this change be rolled back? Is it reversible?",
        "",
        "Iterate through these three questions until you are confident in",
        "your answer. Do not skip any dimension. Then ACT: call tools,",
        "verify, iterate. When the target is ACHIEVED and you are certain,",
        "signal with Ja. Ja is not a hope; it is a judgment that the work",
        "is DONE.",
        "",
        addendum,
        "",
        "Tools (full schemas in the API call):",
        "  write_file(path, content, expected_changes)        expected_changes required",
        "  apply_patch(path, unified_diff, expected_changes)  expected_changes required",
        "  read_file(path)                                    auto-executes",
        "  list_dir(path='.')                                 auto-executes",
        "  search_memory(query)                               auto-executes, searches past reasoning and tool results",
        "  run_in_sandbox(python_code)                        isolated podman, no network",
        "  run_shell(cmd, timeout_s=60)                       runs in the project workspace",
        "",
        "All paths are relative to the project root. Do not escape with '..'.",
    ])


def build_header(
    project: "Project",
    system_prompt: str,
    *,
    mode: str = "code",
) -> str:
    """Pack the four-header context package sent to a fresh LLM instance."""
    prompts = _prompt_history_lines(project)
    actions = _action_log_lines(project)
    snap = filesystem_snapshot(pathlib.Path(project.path))
    protocol = build_protocol(mode)

    return "\n".join([
        system_prompt,
        "",
        "=== H4: Protocol ===",
        protocol,
        "",
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
    ])
