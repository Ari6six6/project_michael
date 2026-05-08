"""Agent loop: _run_agent_loop and its helpers."""
from __future__ import annotations

import json
from typing import Any, Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from rich.panel import Panel
from rich.syntax import Syntax

import michael.globals as G
from michael.backends import (
    LocalPodmanBackend,
    SandboxBackend,
    _require_endpoint,
    _ssh_preflight,
    llm_client,
    make_backend,
)
from michael.config import Config, ModelProfile
from michael.project import Project, append_event
from michael.tools import (
    PendingChanges,
    TOOLS,
    _resolve_in_project,
    _summary_for,
    commit_pending,
    dispatch_tool_call,
    _format_delta,
)
from michael.utils import build_header

_NUDGE_NO_JA = (
    "system reminder: you ended your turn without tool calls and without "
    f"the {G.JA_PASSPHRASE!r} passcode. Either use tools to keep iterating, "
    f"or end your message with `{G.JA_PASSPHRASE}` on its own line to surface "
    "your work to the user. Until then you are talking to Michael, not "
    "the user."
)


def _tools_for_mode(mode: str) -> list[dict[str, Any]]:
    """code/nitro = full toolset; discussion = read-only tools only."""
    if mode == "discussion":
        return [t for t in TOOLS if t["function"]["name"] in G.AUTO_EXEC_TOOLS]
    return TOOLS


def _resolve_nitro_model(cfg: Config, model: Optional[str]) -> tuple[str, ModelProfile]:
    """Pick the heavy model for nitro: explicit --model wins, then 'nitro', then 'big'."""
    if model:
        return cfg.get_model(model)
    for candidate in ("nitro", "big"):
        if candidate in cfg.models:
            return candidate, cfg.models[candidate]
    raise G.MichaelError(
        "nitro requires a 'nitro' or 'big' model profile in config "
        "(or pass --model NAME explicitly)"
    )


def _present_pending_to_user(
    project: Project,
    pending: PendingChanges,
    final_text: str,
) -> bool:
    """Render accumulated pending changes for the user; ask one yes/no. Returns True on apply."""
    if final_text:
        G.console.print(
            Panel(final_text, title="assistant — Ja", border_style="green")
        )
    if not pending.change_log:
        return True

    for i, entry in enumerate(pending.change_log, 1):
        import difflib
        delta = entry["delta"]
        title = (
            f"[{i}/{len(pending.change_log)}] "
            f"{_summary_for(entry['tool'], entry['args'])}  "
            f"+{len(delta['added'])} ~{len(delta['modified'])} "
            f"-{len(delta['removed'])}"
        )
        sections: list[str] = []
        if entry["tool"] == "write_file":
            rel = str(entry["args"].get("path", "?"))
            try:
                real_target = _resolve_in_project(project, rel)
                old = real_target.read_text(errors="replace") if real_target.is_file() else ""
            except G.MichaelError:
                old = ""
            diff = "".join(difflib.unified_diff(
                old.splitlines(keepends=True),
                str(entry["args"].get("content", "")).splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
            )) or "(no changes)"
            sections.append(diff)
        elif entry["tool"] == "apply_patch":
            sections.append(str(entry["args"].get("unified_diff", "")))
        sections.append(_format_delta(delta))
        if entry.get("verify_rc") is not None:
            sections.append(f"verify rc={entry['verify_rc']}")
        if entry.get("mismatch"):
            sections.append(f"prediction mismatch: {entry['mismatch']}")
        G.console.print(
            Panel(
                Syntax("\n\n".join(sections), "diff", theme="ansi_dark", word_wrap=True),
                title=title, border_style="cyan",
            )
        )

    try:
        choice = (typer.prompt(
            f"Apply all {len(pending.change_log)} pending change(s)? [Y]es / [n]o",
            default="y",
        ) or "").strip().lower()
    except (KeyboardInterrupt, typer.Abort):
        choice = "n"
    return choice in ("", "y", "yes")


def _run_agent_loop(
    project: Project,
    cfg: Config,
    name: str,
    profile: ModelProfile,
    mode: str,
    *,
    verb_label: str,
    god_mode: bool = False,
) -> None:
    """Shared agent-loop body for `run`, `new code`, `new discussion`, `nitro`."""
    endpoint = _require_endpoint(profile, name)
    _ssh_preflight(cfg)

    client = llm_client(endpoint, profile.vllm_api_key)
    backend = make_backend(cfg)
    tools = _tools_for_mode(mode)
    base_prompt = cfg.resolved_system_prompt()

    backend_label = (
        "remote-podman (vps)" if cfg.vps_active()
        else ("local-podman" if isinstance(backend, LocalPodmanBackend)
              else "no-sandbox")
    )
    G.console.print(
        f"[bold cyan]michael {verb_label}[/] [dim]project={project.slug}  "
        f"model={name}  mode={mode}  sandbox={backend_label}[/]"
    )
    G.console.print(
        f"[dim]empty line or 'quit' to exit · Ctrl-C aborts an in-flight "
        f"loop · LLM surfaces with the {G.JA_PASSPHRASE!r} passcode[/]"
    )

    session = PromptSession(
        history=FileHistory(str(G.REPL_HISTORY_PATH)),
        auto_suggest=AutoSuggestFromHistory(),
    )

    append_event(
        "agent.started",
        {
            "model": name,
            "served": profile.served_model_name,
            "mode": mode,
            "god": god_mode,
            "sandbox": backend_label,
        },
        project=project,
    )
    while True:
        if god_mode:
            user = G._GOD_MODE_PROMPT
        else:
            try:
                user = session.prompt(">>> ")
            except (EOFError, KeyboardInterrupt):
                break
            user = (user or "").strip()
            if not user or user.lower() in ("quit", "exit"):
                break

        append_event(
            "prompt.sent",
            {
                "prompt": user,
                "model": name,
                "served": profile.served_model_name,
                "mode": mode,
            },
            project=project,
        )

        header = build_header(project, base_prompt, mode=mode)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": header},
            {"role": "user", "content": user},
        ]
        pending = PendingChanges()
        turn = 0
        ja_received = False
        try:
            while True:
                turn += 1
                G.console.print(f"[dim]· turn {turn}: model thinking…[/]")
                try:
                    resp = client.chat.completions.create(
                        model=profile.served_model_name,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        stream=False,
                        timeout=float(profile.request_timeout_s),
                    )
                except Exception as e:
                    G.err.print(f"LLM error: {e}")
                    append_event(
                        "error",
                        {"where": "agent_loop", "msg": str(e), "turn": turn},
                        project=project,
                    )
                    pending.discard()
                    break

                msg = resp.choices[0].message
                content = msg.content or ""
                if content:
                    payload: dict[str, Any] = {
                        "chars": len(content),
                        "model": name,
                        "served": profile.served_model_name,
                        "turn": turn,
                    }
                    if cfg.log_responses:
                        payload["text"] = content
                    append_event("assistant.message", payload, project=project)

                tool_calls = msg.tool_calls or []
                if tool_calls:
                    for tc in tool_calls:
                        G.console.print(f"[dim]· turn {turn}: tool {tc.function.name}[/]")
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": content,
                }
                if tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in tool_calls
                    ]
                messages.append(assistant_msg)

                if tool_calls:
                    for tc in tool_calls:
                        tname = tc.function.name
                        try:
                            targs = json.loads(tc.function.arguments or "{}")
                        except json.JSONDecodeError:
                            targs = {}
                        result = dispatch_tool_call(
                            tname, targs, project, cfg, backend, pending,
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })
                    continue

                if G._message_ends_with_ja(content):
                    ja_received = True
                    append_event(
                        "assistant.ja",
                        {"turn": turn, "pending": len(pending.change_log)},
                        project=project,
                    )
                    break
                G.console.print(
                    f"[yellow]· turn {turn}: no {G.JA_PASSPHRASE} and no tool "
                    f"calls — nudging the model back into the loop[/]"
                )
                messages.append({"role": "system", "content": _NUDGE_NO_JA})
                continue
        except KeyboardInterrupt:
            n_discarded = len(pending.change_log)
            G.err.print(f"\nturn {turn}: aborted by user; pending changes discarded")
            append_event(
                "agent.aborted",
                {"turn": turn, "pending": n_discarded},
                project=project,
            )
            if n_discarded:
                append_event(
                    "pending.discarded",
                    {
                        "turn": turn,
                        "count": n_discarded,
                        "tools": [e["tool"] for e in pending.change_log],
                    },
                    project=project,
                )
            pending.discard()
            if god_mode:
                break
            continue

        if not ja_received:
            pending.discard()
            if god_mode:
                break
            continue

        if god_mode:
            if content:
                G.console.print(
                    Panel(content, title="⚡ god — Ja", border_style="yellow")
                )
            if pending.change_log:
                summaries = commit_pending(project, pending)
                for s in summaries:
                    G.console.print(f"[green]auto-applied[/] {s['summary']}")
            break
        else:
            approved = _present_pending_to_user(project, pending, content)
            if approved:
                summaries = commit_pending(project, pending)
                if summaries:
                    G.console.print(f"[green]applied[/] {len(summaries)} change(s)")
            else:
                for entry in pending.change_log:
                    append_event(
                        "tool.rejected",
                        {
                            "tool": entry["tool"],
                            "args": entry["args"],
                            "summary": _summary_for(entry["tool"], entry["args"]),
                            "delta": entry["delta"],
                        },
                        project=project,
                    )
                pending.discard()
                G.console.print("[yellow]rejected[/] pending changes discarded")

    append_event(
        "agent.ended",
        {"model": name, "mode": mode, "god": god_mode},
        project=project,
    )
