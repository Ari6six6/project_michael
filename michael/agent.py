"""Agent loop: four-room Kantian cycle with dynamic tool loading."""
from __future__ import annotations

import importlib.util
import json
import pathlib
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
    TOOLS_READ_ONLY,
    TOOLS_PLANNING,
    commit_pending,
    dispatch_tool_call,
)
from michael.utils import build_header, load_scripture


# Kept for backwards-compatibility with imports in main.py
_NUDGE_NO_JA = "Keep going. Signal Ja only when the job is done."


def _load_dynamic_tools(project_path: str) -> list[dict[str, Any]]:
    """Load tool schemas from the global toolbox and the project-local tools/ dir.

    Global tools (~/.michael/toolbox/) load first; a project-local tool with
    the same name overrides the global one.
    """
    seen: dict[str, dict[str, Any]] = {}  # name → schema, later entries win
    search_dirs = [
        pathlib.Path(G.GLOBAL_TOOLS_DIR),
        pathlib.Path(project_path) / "tools",
    ]
    for tools_dir in search_dirs:
        if not tools_dir.exists():
            continue
        for py_file in sorted(tools_dir.glob("*.py")):
            try:
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                if hasattr(mod, "TOOL_SCHEMA"):
                    name = mod.TOOL_SCHEMA.get("function", {}).get("name", py_file.stem)
                    seen[name] = mod.TOOL_SCHEMA
            except Exception as exc:
                G.err.print(f"[dim]dynamic tool load failed ({py_file.name}): {exc}[/]")
    return list(seen.values())


def _run_room(
    room: dict[str, Any],
    messages: list[dict[str, Any]],
    available_tools: list[dict[str, Any]],
    project: Project,
    client: Any,
    profile: ModelProfile,
    pending: PendingChanges,
    cfg: Config,
    backend: Any,
) -> tuple[list[dict[str, Any]], str]:
    """Iterate within one Kantian room until Ja. Returns (messages, ja_content)."""
    turn = 0
    while True:
        turn += 1
        G.console.print(f"[dim]· {room['label']} turn {turn}[/]")
        resp = client.chat.completions.create(
            model=profile.served_model_name,
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
            stream=False,
            timeout=float(profile.request_timeout_s),
            enable_thinking=profile.enable_thinking,
        )
        msg = resp.choices[0].message
        content = msg.content or ""

        if content:
            payload: dict[str, Any] = {
                "chars": len(content),
                "turn": turn,
                "room": room["name"],
            }
            if cfg.log_responses:
                payload["text"] = content
            append_event("assistant.message", payload, project=project)

        tool_calls = msg.tool_calls or []
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
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
                G.console.print(f"[dim]· tool {tc.function.name}[/]")
                try:
                    targs = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    targs = {}
                result = dispatch_tool_call(
                    tc.function.name, targs, project, cfg, backend, pending
                )
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
            continue

        if G._message_ends_with_ja(content):
            append_event(room["name"] + ".ja", {"turn": turn}, project=project)
            return messages, content

        G.console.print(f"[yellow]· {room['label']}: no Ja — nudging[/]")
        messages.append({"role": "system", "content": room["nudge"]})


def _run_agent_loop(
    project: Project,
    cfg: Config,
    name: str,
    profile: ModelProfile,
    prompt: str,
    *,
    verb_label: str = "run",
) -> None:
    """Run one prompt through the four-room Kantian cycle. Iterates until Room 4
    confirms the goal is met, then auto-commits all staged changes and returns."""
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
        f"[dim]Four-room Kantian cycle · up to {G.MAX_AGENT_CYCLES} cycles · "
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

    for cycle_num in range(1, G.MAX_AGENT_CYCLES + 1):
        G.console.print(f"\n[bold cyan]══ CYCLE {cycle_num} ══[/]")
        append_event("cycle.started", {"cycle": cycle_num}, project=project)

        dynamic = _load_dynamic_tools(project.path)
        if dynamic:
            names = ", ".join(
                d["function"]["name"] for d in dynamic if "function" in d
            )
            G.console.print(f"[dim]loaded {len(dynamic)} dynamic tool(s): {names}[/]")
        full_tools = TOOLS + dynamic
        room_tool_lists = [
            TOOLS_READ_ONLY + dynamic,   # Room 1: explore + call user-built tools
            full_tools,                  # Room 2: build (full access, unchanged)
            TOOLS_PLANNING + dynamic,    # Room 3: plot + call user-built tools
            TOOLS_READ_ONLY + dynamic,   # Room 4: gate + call user-built tools
        ]

        try:
            for room, room_tools in zip(G.ROOMS[:3], room_tool_lists[:3]):
                G.console.print(f"\n[bold]{room['label']}[/]")
                append_event(
                    room["name"] + ".entered", {"cycle": cycle_num}, project=project
                )
                messages.append({"role": "system", "content": room["directive"]})
                messages, _ = _run_room(
                    room, messages, room_tools,
                    project, client, profile, pending, cfg, backend,
                )
                G.console.print(f"[green]✓ {room['label']}[/]")

            room4 = G.ROOMS[3]
            G.console.print(f"\n[bold]{room4['label']}[/]")
            append_event(
                room4["name"] + ".entered", {"cycle": cycle_num}, project=project
            )
            messages.append({"role": "system", "content": room4["directive"]})
            messages, gate_content = _run_room(
                room4, messages, TOOLS_READ_ONLY + dynamic,
                project, client, profile, pending, cfg, backend,
            )

        except KeyboardInterrupt:
            pending.discard()
            G.err.print("\nturn aborted by user; pending changes discarded")
            append_event(
                "agent.aborted", {"cycle": cycle_num}, project=project
            )
            append_event("agent.ended", {"model": name, "aborted": True}, project=project)
            return
        except Exception as exc:
            G.err.print(f"LLM error: {exc}")
            append_event(
                "error",
                {"where": "agent_loop", "msg": str(exc), "cycle": cycle_num},
                project=project,
            )
            pending.discard()
            append_event("agent.ended", {"model": name, "ja": False}, project=project)
            return

        if G._message_ends_with_ja(gate_content):
            from rich.panel import Panel
            G.console.print(Panel(gate_content, title="⚡ Goal Complete", border_style="green"))
            if pending.change_log:
                summaries = commit_pending(project, pending)
                for s in summaries:
                    G.console.print(f"[green]applied[/] {s['summary']}")
            else:
                G.console.print("[dim]no file changes staged[/]")
            append_event(
                "agent.ended",
                {"model": name, "ja": True, "cycles": cycle_num},
                project=project,
            )
            return

        G.console.print(
            f"[yellow]· cycle {cycle_num}: goal not yet met — "
            f"starting cycle {cycle_num + 1}[/]"
        )
        append_event("cycle.incomplete", {"cycle": cycle_num}, project=project)
        messages.append({
            "role": "system",
            "content": (
                f"CYCLE {cycle_num} COMPLETE — GOAL NOT YET ANSWERED.\n\n"
                f"{gate_content}\n\n"
                f"Starting cycle {cycle_num + 1}. "
                f"Continue working toward the original goal."
            ),
        })

    from rich.panel import Panel
    G.console.print(
        Panel(
            f"{gate_content}\n\n"
            "[dim]Run [bold]michael run '<what you need>'[/bold] to continue "
            "— Michael will pick up exactly where he left off.[/dim]",
            title=f"⏸  Cycle limit reached ({G.MAX_AGENT_CYCLES})",
            border_style="yellow",
        )
    )
    pending.discard()
    append_event(
        "agent.ended",
        {"model": name, "ja": False, "cycles": G.MAX_AGENT_CYCLES, "blocked_on": gate_content},
        project=project,
    )
