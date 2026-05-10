"""Agent loop: _run_agent_loop and its helpers."""
from __future__ import annotations

import json
from typing import Any

import michael.globals as G
from michael.backends import (
    LocalPodmanBackend,
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
    _summary_for,
    commit_pending,
    dispatch_tool_call,
    _format_delta,
)
from michael.utils import build_header, load_scripture

_NUDGE_NO_JA = (
    "Keep going. You have full tool access — read files, run the sandbox, "
    "explore, verify. The user is not watching yet. Surface your work with "
    f"`{G.JA_PASSPHRASE}` only when you are certain the job is done."
)


def _run_agent_loop(
    project: Project,
    cfg: Config,
    name: str,
    profile: ModelProfile,
    prompt: str,
    *,
    verb_label: str = "run",
) -> None:
    """Run one prompt through the agent loop. Iterates privately until Ja, then
    auto-commits all staged changes and returns. No Y/n gate, no outer loop."""
    endpoint = _require_endpoint(profile, name)
    _ssh_preflight(cfg)

    client = llm_client(endpoint, profile.vllm_api_key)
    backend = make_backend(cfg)
    base_prompt = cfg.resolved_system_prompt()

    backend_label = (
        "remote-podman (vps)" if cfg.vps_active()
        else ("local-podman" if isinstance(backend, LocalPodmanBackend)
              else "no-sandbox")
    )
    G.console.print(
        f"[bold cyan]michael {verb_label}[/] [dim]project={project.slug}  "
        f"model={name}  sandbox={backend_label}[/]"
    )
    G.console.print(
        f"[dim]LLM iterates privately until the {G.JA_PASSPHRASE!r} passcode · "
        f"Ctrl-C aborts[/]"
    )

    append_event(
        "agent.started",
        {"model": name, "served": profile.served_model_name, "sandbox": backend_label},
        project=project,
    )

    append_event(
        "prompt.sent",
        {"prompt": prompt, "model": name, "served": profile.served_model_name},
        project=project,
    )

    scripture = load_scripture(cfg.scripture_dir)
    header = build_header(project, base_prompt, scripture)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": header},
        {"role": "user", "content": prompt},
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
                    tools=TOOLS,
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
        pending.discard()
        append_event("agent.ended", {"model": name, "aborted": True}, project=project)
        return

    if ja_received:
        if content:
            from rich.panel import Panel
            G.console.print(Panel(content, title="⚡ Ja", border_style="yellow"))
        if pending.change_log:
            summaries = commit_pending(project, pending)
            for s in summaries:
                G.console.print(f"[green]applied[/] {s['summary']}")
        else:
            G.console.print("[dim]no file changes staged[/]")
    else:
        pending.discard()

    append_event("agent.ended", {"model": name, "ja": ja_received}, project=project)
