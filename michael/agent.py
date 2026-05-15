"""Agent loop: flat tool loop with explicit commit_changes gate."""
from __future__ import annotations

import importlib.util
import json
import pathlib
from typing import Any

import httpx
import michael.globals as G
from michael.backends import (
    LocalPodmanBackend,
    _ensure_tunnel,
    _ping_vllm,
    _require_endpoint,
    _restart_vllm_on_gpu,
    _ssh_preflight,
    llm_client,
    make_backend,
)
from michael.config import Config, ModelProfile
from michael.project import Project, append_event
import subprocess

from michael.tools import (
    PendingChanges,
    TOOLS,
    COMMIT_SENTINEL,
    commit_pending,
    dispatch_tool_call,
)
from michael.utils import build_header, load_scripture


def _load_dynamic_tools(project_path: str) -> list[dict[str, Any]]:
    """Load tool schemas from the global toolbox and the project-local tools/ dir.

    Global tools (~/.michael/toolbox/) load first; a project-local tool with
    the same name overrides the global one.
    """
    seen: dict[str, dict[str, Any]] = {}  # name → schema, later entries win
    search_dirs = [
        pathlib.Path(__file__).parent.parent / "toolbox",  # bundled tools (lowest priority)
        pathlib.Path(G.GLOBAL_TOOLS_DIR),                  # user global (~/.michael/toolbox/)
        pathlib.Path(project_path) / "tools",              # project-local (highest priority)
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


def _probe_deliverable(project: Project, run_cmd: str) -> tuple[bool, str]:
    """Run the deliverable with --help; return (success, output)."""
    try:
        cp = subprocess.run(
            ["bash", "-c", f"{run_cmd} --help"],
            cwd=project.path,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        out = (cp.stdout or "")[:500] + (cp.stderr or "")[:200]
        return cp.returncode == 0, out
    except Exception as exc:
        return False, str(exc)


def _run_agent_loop(
    project: Project,
    cfg: Config,
    name: str,
    profile: ModelProfile,
    prompt: str,
    *,
    verb_label: str = "run",
) -> None:
    """Run one prompt through a flat tool loop. The LLM iterates freely with all
    tools available and calls commit_changes() when done."""
    endpoint = _require_endpoint(profile, name)
    _ssh_preflight(cfg)

    if cfg.gpu.ssh_host:
        _ensure_tunnel(cfg.gpu)
        if not _ping_vllm(endpoint, profile.vllm_api_key):
            G.console.print("[yellow]vLLM unreachable — auto-restarting...[/]")
            _restart_vllm_on_gpu(cfg.gpu)

    client = llm_client(endpoint, profile.vllm_api_key, profile.enable_thinking)
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
    G.console.print(f"[dim]Flat loop · up to {G.MAX_AGENT_TURNS} turns · Ctrl-C aborts[/]")

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
    dynamic = _load_dynamic_tools(project.path)
    if dynamic:
        names = ", ".join(d["function"]["name"] for d in dynamic if "function" in d)
        G.console.print(f"[dim]loaded {len(dynamic)} dynamic tool(s): {names}[/]")
    all_tools = TOOLS + dynamic

    try:
        for turn in range(1, G.MAX_AGENT_TURNS + 1):
            G.console.print(f"[dim]· turn {turn}[/]")
            try:
                resp = client.chat.completions.create(
                    model=profile.served_model_name,
                    messages=messages,
                    tools=all_tools,
                    tool_choice="auto",
                    stream=False,
                    timeout=float(profile.request_timeout_s),
                )
            except httpx.HTTPStatusError as _exc:
                if _exc.response.status_code == 400 and cfg.gpu.ssh_host:
                    G.console.print("[yellow]400 from vLLM — restarting with correct flags...[/]")
                    _restart_vllm_on_gpu(cfg.gpu)
                    client = llm_client(endpoint, profile.vllm_api_key, profile.enable_thinking)
                    resp = client.chat.completions.create(
                        model=profile.served_model_name,
                        messages=messages,
                        tools=all_tools,
                        tool_choice="auto",
                        stream=False,
                        timeout=float(profile.request_timeout_s),
                    )
                else:
                    raise
            choice = resp.choices[0]
            content = choice.content or ""

            if content:
                payload: dict[str, Any] = {"chars": len(content), "turn": turn}
                if cfg.log_responses:
                    payload["text"] = content
                append_event("assistant.message", payload, project=project)

            tool_calls = choice.tool_calls or []
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in tool_calls
                ]
            messages.append(assistant_msg)

            if not tool_calls:
                # LLM responded without calling a tool — natural loop exit.
                if content:
                    G.console.print(content)
                append_event("agent.ended", {"model": name, "turns": turn}, project=project)
                return

            committed = False
            for tc in tool_calls:
                G.console.print(f"[dim]· tool {tc.name}[/]")
                try:
                    targs = json.loads(tc.arguments or "{}")
                except json.JSONDecodeError:
                    targs = {}
                if tc.name in ("write_file", "apply_patch") and "path" in targs:
                    G.console.print(f"[dim]  → {targs['path']}[/]")
                result = dispatch_tool_call(
                    tc.name, targs, project, cfg, backend, pending
                )
                if result == COMMIT_SENTINEL:
                    committed = True
                    # Still append so the message list is well-formed if we continued.
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": "Changes committed."})
                else:
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

            if committed:
                from rich.panel import Panel
                from michael.project import detect_deliverable, register_deliverable

                det = detect_deliverable(project)
                if det:
                    deliverable, run_cmd = det
                    ok, probe_out = _probe_deliverable(project, run_cmd)
                    if ok:
                        register_deliverable(project, deliverable, run_cmd)
                        installed = G.MICHAEL_BIN_DIR / project.slug
                        G.console.print(Panel(
                            f"[bold]{deliverable}[/]\n"
                            f"installed: [cyan]{installed}[/]\n\n"
                            f"[dim]{probe_out[:300]}[/]\n\n"
                            f"[dim]Add to PATH: export PATH=\"{G.MICHAEL_BIN_DIR}:$PATH\"[/]",
                            title="⚡ Committed + Delivered",
                            border_style="green",
                        ))
                        append_event(
                            "tool.executed",
                            {"tool": "deliver", "summary": f"delivered {deliverable}", "run_cmd": run_cmd},
                            project=project,
                        )
                    else:
                        G.console.print(Panel(
                            f"[yellow]{deliverable}[/] — probe failed\n\n[dim]{probe_out[:400]}[/]\n\n"
                            "Run [bold]michael run '<fix the issue>'[/] to repair and re-deliver.",
                            title="⚡ Committed (verify failed)",
                            border_style="yellow",
                        ))
                else:
                    G.console.print(Panel("Done.", title="⚡ Committed", border_style="green"))

                append_event(
                    "agent.ended", {"model": name, "committed": True, "turns": turn},
                    project=project,
                )
                return

    except KeyboardInterrupt:
        pending.discard()
        G.err.print("\nturn aborted by user; pending changes discarded")
        append_event("agent.aborted", {}, project=project)
        append_event("agent.ended", {"model": name, "aborted": True}, project=project)
        return
    except Exception as exc:
        G.err.print(f"LLM error: {exc}")
        append_event("error", {"where": "agent_loop", "msg": str(exc)}, project=project)
        pending.discard()
        append_event("agent.ended", {"model": name, "error": True}, project=project)
        return

    # Reached max turns without commit_changes
    from rich.panel import Panel
    G.console.print(
        Panel(
            "Max turns reached without commit_changes being called.\n\n"
            "[dim]Run [bold]michael run '<what you need>'[/bold] to continue.[/dim]",
            title=f"⏸  Turn limit ({G.MAX_AGENT_TURNS})",
            border_style="yellow",
        )
    )
    pending.discard()
    append_event(
        "agent.ended",
        {"model": name, "turns": G.MAX_AGENT_TURNS, "committed": False},
        project=project,
    )
