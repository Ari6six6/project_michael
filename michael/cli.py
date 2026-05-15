"""CLI commands, Typer bindings, and the interactive REPL."""
from __future__ import annotations

import json
import os
import pathlib
import shlex
import subprocess
import sys
import time
from typing import Any, Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from rich.panel import Panel
from rich.table import Table

import michael.globals as G
from michael.agent import _run_agent_loop
from michael.backends import (
    VastClient,
    _gpu_ssh_run,
    _gpu_ssh_stream,
    _ping_vllm,
    _require_endpoint,
    _ssh_argv,
    _ssh_preflight,
    chat_stream,
    gpu_port_forward_cmd,
    llm_client,
    make_backend,
    parse_vast_ssh_cmd,
)
from michael.config import Config, CONFIG_HELP, GpuConfig, make_stub_config
from michael.project import (
    Project,
    append_event,
    create_project,
    detect_deliverable,
    get_active_project,
    get_active_slug,
    iter_events,
    list_projects,
    load_catalog,
    register_deliverable,
    replay_global,
    require_active_project,
    save_catalog,
    set_active_slug,
    slugify,
)
from michael.tools import TOOLS, _list_trash, _undo_one
from michael.utils import build_header

app = typer.Typer(
    no_args_is_help=False,
    rich_markup_mode="rich",
    help="michael — air-gapped AI control loop",
)

gpu_app = typer.Typer(help="GPU instance management (A100 / vLLM).")
app.add_typer(gpu_app, name="gpu")


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def cmd_init() -> None:
    G.STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    if not G.GLOBAL_CONFIG_PATH.is_file():
        make_stub_config().save()
        G.console.print(f"[green]wrote stub[/] {G.GLOBAL_CONFIG_PATH}")
    else:
        G.console.print(f"[dim]config exists[/] {G.GLOBAL_CONFIG_PATH}")
    append_event("config.loaded", {"path": str(G.GLOBAL_CONFIG_PATH)})
    G.console.print(
        Panel(
            "Edit ~/.michael/config.json — fill in:\n\n"
            "  [bold]vast_api_key[/]              your Vast.ai console API key\n"
            "  [bold]models.god.vast_instance_id[/]  numeric instance id\n"
            "  [bold]models.god.served_model_name[/]  matches --served-model-name on vLLM\n"
            "  [bold]models.god.vllm_api_key[/]       the key vLLM was launched with\n\n"
            "[dim]Optional, for remote sandbox on the VPS:[/]\n"
            "  [bold]vps.host[/]                  VPS public IP/hostname\n"
            "  [bold]vps.user[/]                  ssh user (default: michael)\n"
            "  [bold]vps.ssh_key_path[/]          path to private key\n"
            "  [bold]vps.workspace_dir[/]         /home/michael/workspace\n\n"
            "[dim]Leave vps.host empty to run without sandbox.[/]",
            title="checklist",
            border_style="green",
        )
    )


def cmd_show() -> None:
    projects = list_projects()
    if not projects:
        G.console.print("0")
        return
    active = get_active_slug()
    table = Table(title=f"projects ({len(projects)})", border_style="cyan")
    table.add_column("active", justify="center")
    table.add_column("slug", style="bold")
    table.add_column("name")
    table.add_column("path")
    table.add_column("created")
    for p in projects:
        mark = "*" if p.slug == active else ""
        table.add_row(mark, p.slug, p.name, p.path, p.created_at)
    G.console.print(table)


def cmd_new(name: Optional[str]) -> None:
    if not name:
        name = (typer.prompt("name") or "").strip()
    if not name:
        G.err.print("name is required")
        return
    cfg = Config.load()
    wb_root = pathlib.Path(cfg.workbench_root).expanduser()
    default_path = wb_root / "codebases" / slugify(name)
    path_str = typer.prompt("path", default=str(default_path))
    path = pathlib.Path(path_str).expanduser().resolve()
    proj = create_project(name, path)
    set_active_slug(proj.slug)
    append_event("project.activated", {"slug": proj.slug})
    G.console.print(f"[green]created[/] {proj.slug} at {proj.path}")


def cmd_use(slug: str) -> None:
    proj = Project.load(slug)
    set_active_slug(proj.slug)
    append_event("project.activated", {"slug": proj.slug})
    G.console.print(f"[green]active[/] {proj.slug}")


def cmd_current() -> None:
    p = get_active_project()
    if not p:
        G.console.print("(no active project)")
        return
    G.console.print(f"{p.slug} — {p.name} — {p.path}")


def cmd_config() -> None:
    G.STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    if not G.GLOBAL_CONFIG_PATH.is_file():
        make_stub_config().save()
    help_lines = [f"[bold]{k}[/] — {v}" for k, v in CONFIG_HELP.items()]
    G.console.print(
        Panel(
            "\n".join(help_lines),
            title=f"config: {G.GLOBAL_CONFIG_PATH}",
            border_style="green",
        )
    )
    current_text = G.GLOBAL_CONFIG_PATH.read_text()
    edited = typer.edit(current_text, extension=".json")
    if edited is None or edited == current_text:
        G.console.print("[dim]no changes[/]")
        return
    try:
        json.loads(edited)
    except json.JSONDecodeError as e:
        G.err.print(f"invalid JSON, not saved: {e}")
        return
    G.GLOBAL_CONFIG_PATH.write_text(edited)
    os.chmod(G.GLOBAL_CONFIG_PATH, 0o600)
    G.console.print("[green]config saved[/]")


def cmd_up() -> None:
    cfg = Config.load()
    name, profile = cfg.get_model()
    if not profile.vast_instance_id:
        raise G.MichaelError(f"models.{name}.vast_instance_id is not set (run `config`)")
    vast = VastClient(cfg.vast_api_key)
    try:
        vast.start(profile.vast_instance_id)
        append_event(
            "instance.start_requested",
            {"id": profile.vast_instance_id, "model": name},
        )
        G.console.print(f"[cyan]starting {name} (instance {profile.vast_instance_id})…[/]")
        endpoint: Optional[str] = None
        _max_wait_s = 600
        _poll_s = max(cfg.boot_poll_s, 10)
        _elapsed = 0
        _attempt = 0
        while _elapsed < _max_wait_s:
            time.sleep(_poll_s)
            _elapsed += _poll_s
            _attempt += 1
            try:
                ep = vast.endpoint_for(profile.vast_instance_id, profile.vllm_internal_port)
            except G.MichaelError:
                ep = None
            append_event(
                "instance.poll",
                {"i": _attempt, "model": name, "endpoint_known": bool(ep), "elapsed_s": _elapsed},
            )
            if not ep:
                G.console.print(f"[dim]· poll {_attempt} ({_elapsed}s elapsed): no endpoint yet[/]")
            elif _ping_vllm(ep, profile.vllm_api_key, timeout_s=10.0):
                endpoint = ep
                break
            else:
                G.console.print(f"[dim]· poll {_attempt} ({_elapsed}s elapsed): endpoint {ep} not ready[/]")
            _poll_s = min(_poll_s * 2, 60)
        if not endpoint:
            raise G.MichaelError(f"instance did not become ready within {_max_wait_s}s")
        append_event("endpoint.discovered", {"endpoint": endpoint, "model": name})
        append_event(
            "instance.started",
            {"id": profile.vast_instance_id, "model": name, "endpoint": endpoint},
        )
        profile.endpoint = endpoint
        cfg.models[name] = profile
        cfg.save()
        G.console.print(f"[green]ready[/] {name} @ {endpoint}")
    finally:
        vast.close()


def cmd_down() -> None:
    cfg = Config.load()
    name, profile = cfg.get_model()
    if not profile.vast_instance_id:
        raise G.MichaelError(f"models.{name}.vast_instance_id is not set")
    vast = VastClient(cfg.vast_api_key)
    try:
        vast.stop(profile.vast_instance_id)
        append_event(
            "instance.stop_requested",
            {"id": profile.vast_instance_id, "model": name},
        )
        append_event(
            "instance.stopped",
            {"id": profile.vast_instance_id, "model": name},
        )
        profile.endpoint = None
        cfg.models[name] = profile
        cfg.save()
        G.console.print(f"[yellow]stopped[/] {name} ({profile.vast_instance_id})")
    finally:
        vast.close()


def cmd_gpu_up() -> None:
    cfg = Config.load()
    gpu = cfg.gpu

    # ── First-time setup: read SSH string from Vast.ai console ──
    if not gpu.ssh_host:
        G.console.print(
            "[bold cyan]Paste the SSH command from your Vast.ai console[/] "
            "[dim](e.g. ssh root@1.2.3.4 -p 10022)[/]"
        )
        ssh_str = typer.prompt("SSH command").strip()
        user, host, port = parse_vast_ssh_cmd(ssh_str)
        gpu.ssh_user = user
        gpu.ssh_host = host
        gpu.ssh_port = port

        # Try to auto-discover the instance ID from the Vast.ai API
        if cfg.vast_api_key:
            try:
                vast = VastClient(cfg.vast_api_key)
                for inst in vast.list():
                    if inst.get("ssh_host") == host or inst.get("public_ipaddr") == host:
                        gpu.vast_instance_id = str(inst["id"])
                        G.console.print(
                            f"[dim]auto-detected instance id: {gpu.vast_instance_id}[/]"
                        )
                        break
                vast.close()
            except G.MichaelError:
                pass

        cfg.gpu = gpu
        cfg.save()
        G.console.print(f"[green]GPU config saved[/] ({gpu.ssh_user}@{gpu.ssh_host}:{gpu.ssh_port})")

    # ── Verify connectivity ──
    G.console.print(f"[dim]connecting to {gpu.ssh_user}@{gpu.ssh_host}:{gpu.ssh_port}…[/]")
    cp = _gpu_ssh_run(gpu, "echo ok", timeout=20)
    if cp.returncode != 0:
        raise G.MichaelError(
            f"GPU unreachable: {cp.stderr.strip()[:200]}\n"
            "Check ssh_key_path in config or try ssh manually."
        )

    # ── Install vLLM if missing ──
    cp = _gpu_ssh_run(gpu, "python3 -c 'import vllm' 2>/dev/null && echo installed || echo missing")
    if "missing" in cp.stdout:
        G.console.print("[cyan]Installing vLLM (grab a coffee, this takes a few minutes)…[/]")
        rc = _gpu_ssh_stream(gpu, "pip install vllm --quiet --upgrade", timeout=900)
        if rc != 0:
            raise G.MichaelError("vLLM installation failed — check GPU terminal for details")
        G.console.print("[green]vLLM installed[/]")

    # ── Check if vLLM is already serving ──
    cp = _gpu_ssh_run(
        gpu,
        f"curl -sf http://localhost:{gpu.vllm_port}/v1/models > /dev/null 2>&1 && echo ready || echo down",
    )
    already_ready = "ready" in cp.stdout

    if not already_ready:
        # Kill any stale process first
        _gpu_ssh_run(gpu, "pkill -f 'vllm serve' 2>/dev/null || true", timeout=10)
        time.sleep(2)

        api_key_flag = f"--api-key {gpu.vllm_api_key} " if gpu.vllm_api_key else ""
        vllm_cmd = (
            f"nohup vllm serve {gpu.model_repo} "
            f"--host 0.0.0.0 --port {gpu.vllm_port} "
            f"--dtype auto --gpu-memory-utilization 0.95 "
            f"--max-model-len 8192 "
            f"{api_key_flag}"
            f"> /tmp/vllm.log 2>&1 & echo $!"
        )
        cp = _gpu_ssh_run(gpu, vllm_cmd, timeout=30)
        pid = cp.stdout.strip()
        G.console.print(f"[cyan]vLLM starting[/] (PID {pid}) — model download may take 20–40 min on first boot")
        G.console.print("[dim]tailing /tmp/vllm.log for progress…[/]")
    else:
        G.console.print("[dim]vLLM already serving, skipping start[/]")

    # ── Poll until /v1/models responds ──
    _max_wait_s = 5400  # 90 min
    _poll_s = 30
    _elapsed = 0
    _attempt = 0
    endpoint_ready = already_ready
    while not endpoint_ready and _elapsed < _max_wait_s:
        time.sleep(_poll_s)
        _elapsed += _poll_s
        _attempt += 1
        cp = _gpu_ssh_run(
            gpu,
            f"curl -sf http://localhost:{gpu.vllm_port}/v1/models > /dev/null 2>&1 && echo ready || echo down",
            timeout=15,
        )
        if "ready" in cp.stdout:
            endpoint_ready = True
            break
        # Show last 2 lines of vllm.log for progress
        log_cp = _gpu_ssh_run(gpu, "tail -2 /tmp/vllm.log 2>/dev/null || true", timeout=10)
        tail = log_cp.stdout.strip().replace("\n", " | ")
        G.console.print(f"[dim]· {_elapsed}s — {tail or 'loading…'}[/]")
        append_event("gpu.poll", {"elapsed_s": _elapsed, "attempt": _attempt})

    if not endpoint_ready:
        raise G.MichaelError(
            f"vLLM did not become ready within {_max_wait_s}s. "
            "SSH in and check /tmp/vllm.log"
        )

    # ── Save endpoint into models.god so `michael run` works via port forward ──
    endpoint = f"http://localhost:{gpu.vllm_port}/v1"
    if "god" not in cfg.models:
        from michael.config import ModelProfile
        cfg.models["god"] = ModelProfile()
        cfg.default_model = cfg.default_model or "god"
    cfg.models["god"].endpoint = endpoint
    cfg.models["god"].served_model_name = gpu.model_repo
    cfg.save()
    append_event("gpu.ready", {"host": gpu.ssh_host, "model": gpu.model_repo, "endpoint": endpoint})

    pf_cmd = gpu_port_forward_cmd(gpu)
    G.console.print(
        Panel(
            f"[bold green]vLLM is ready[/] — {gpu.model_repo}\n\n"
            f"[bold]Open a new terminal and run:[/]\n\n"
            f"  {pf_cmd}\n\n"
            f"[dim]Keep that terminal open. Then use:[/]\n"
            f"  michael run <your prompt>",
            title="port forward",
            border_style="green",
        )
    )


def cmd_gpu_down() -> None:
    cfg = Config.load()
    gpu = cfg.gpu
    if not gpu.ssh_host:
        raise G.MichaelError("no GPU configured — run `michael gpu up` first")

    G.console.print(f"[dim]connecting to {gpu.ssh_user}@{gpu.ssh_host}:{gpu.ssh_port}…[/]")
    _gpu_ssh_run(gpu, "pkill -f 'vllm serve' 2>/dev/null || true", timeout=15)
    G.console.print("[yellow]vLLM stopped[/]")

    if gpu.vast_instance_id and cfg.vast_api_key:
        vast = VastClient(cfg.vast_api_key)
        try:
            vast.stop(gpu.vast_instance_id)
            G.console.print(f"[yellow]instance {gpu.vast_instance_id} stopped[/]")
            append_event("gpu.stopped", {"host": gpu.ssh_host, "instance_id": gpu.vast_instance_id})
        finally:
            vast.close()
    else:
        G.console.print("[dim]no vast_instance_id or vast_api_key — skipping API stop[/]")
        append_event("gpu.stopped", {"host": gpu.ssh_host})

    if "god" in cfg.models:
        cfg.models["god"].endpoint = None
    cfg.save()


def cmd_status() -> None:
    cfg = Config.load()
    state = replay_global()
    active = get_active_project()
    table = Table(title="michael status", border_style="cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("active project", active.slug if active else "(none)")
    if cfg.vps_active():
        table.add_row("vps", f"{cfg.vps.user}@{cfg.vps.host}:{cfg.vps.port}")
        table.add_row("vps.workspace", cfg.vps.workspace_dir)
    else:
        table.add_row("vps", "[dim]not configured (no sandbox)[/]")

    table.add_row("default model", cfg.default_model or "[dim](first available)[/]")
    if not cfg.models:
        table.add_row("models", "[dim](none — edit config.json)[/]")
    for mname, profile in cfg.models.items():
        st = state.get("models", {}).get(mname, {})
        table.add_row(
            f"  {mname}",
            f"state={st.get('instance_state', 'unknown')}  "
            f"endpoint={st.get('endpoint') or profile.endpoint or '—'}",
        )

    table.add_row("errors (global)", str(state["errors"]))
    G.console.print(table)


def cmd_ask(prompt: str) -> None:
    cfg = Config.load()
    name, profile = cfg.get_model()
    endpoint = _require_endpoint(profile, name)
    client = llm_client(endpoint, profile.vllm_api_key)
    project = get_active_project()
    if project is not None:
        append_event(
            "prompt.sent",
            {"prompt": prompt, "model": name, "served": profile.served_model_name},
            project=project,
        )
        system_msg = build_header(project, cfg.resolved_system_prompt())
    else:
        append_event(
            "prompt.sent",
            {"prompt": prompt, "model": name, "served": profile.served_model_name},
        )
        system_msg = cfg.resolved_system_prompt()

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": prompt},
    ]
    text, usage = chat_stream(
        client,
        profile.served_model_name,
        messages,
        timeout_s=float(profile.request_timeout_s),
    )
    payload: dict[str, Any] = {
        "chars": len(text),
        "model": name,
        "served": profile.served_model_name,
        "usage": usage,
    }
    if cfg.log_responses:
        payload["text"] = text
    append_event("assistant.message", payload, project=project)


def cmd_run(prompt: str, no_verify: bool = False) -> None:
    project = require_active_project()
    cfg = Config.load()
    name, profile = cfg.get_model()

    current_prompt = prompt
    for attempt in range(G.MAX_VERIFY_RETRIES + 1):
        result = _run_agent_loop(
            project, cfg, name, profile, current_prompt, verb_label="run"
        )
        if no_verify or result is None:
            break
        verify_passed, error_output = result
        if verify_passed:
            break
        remaining = G.MAX_VERIFY_RETRIES - attempt
        if remaining <= 0:
            G.err.print("[red]auto-verify: retry limit reached[/]")
            break
        G.console.print(
            f"[yellow]auto-verify failed — retrying ({remaining} attempt(s) left)[/]"
        )
        current_prompt = (
            f"AUTO-VERIFY FAILED. The deliverable did not run correctly.\n\n"
            f"Error output:\n{error_output[:800]}\n\n"
            "Fix the issue so the tool runs without errors, then signal Ja."
        )


def cmd_log(tail: int) -> None:
    project = get_active_project()
    if project:
        events = iter_events(project.events_path)
        title = f"events (project: {project.slug})"
    else:
        events = iter_events(G.GLOBAL_EVENTS_PATH)
        title = "events (global)"
    if not events:
        G.console.print("[dim](no events)[/]")
        return
    last = events[-tail:] if tail > 0 else events
    table = Table(
        title=f"{title} — last {len(last)} of {len(events)}",
        border_style="cyan",
    )
    table.add_column("seq", style="bold", justify="right")
    table.add_column("ts")
    table.add_column("type")
    table.add_column("payload")
    for ev in last:
        payload = json.dumps(ev.get("payload", {}), ensure_ascii=False, sort_keys=True)
        if len(payload) > 80:
            payload = payload[:77] + "..."
        table.add_row(
            str(ev.get("seq", "?")),
            str(ev.get("ts", "?")),
            str(ev.get("type", "?")),
            payload,
        )
    G.console.print(table)


def cmd_undo(list_only: bool = False, trash_id: Optional[str] = None) -> None:
    project = require_active_project()
    if list_only:
        entries = _list_trash(project)
        if not entries:
            G.console.print("(no trash)")
            return
        table = Table(
            title=f"trash for {project.slug} (newest last)",
            border_style="cyan",
        )
        table.add_column("trash_id", style="bold")
        table.add_column("ts")
        table.add_column("tool")
        table.add_column("delta")
        table.add_column("verify")
        for m in entries:
            d = m.get("delta", {}) or {}
            delta_summary = (
                f"+{len(d.get('added', []))} "
                f"~{len(d.get('modified', []))} "
                f"-{len(d.get('removed', []))}"
            )
            v = m.get("verify_rc")
            v_str = "—" if v is None else f"rc={v}"
            table.add_row(
                str(m.get("trash_id", "?")),
                str(m.get("ts", "?")),
                str(m.get("tool", "?")),
                delta_summary,
                v_str,
            )
        G.console.print(table)
        return
    metadata = _undo_one(project, trash_id)
    append_event(
        "tool.undone",
        {
            "trash_id": metadata.get("trash_id"),
            "tool": metadata.get("tool"),
            "summary": metadata.get("summary", ""),
        },
        project=project,
    )
    G.console.print(
        f"[green]undone[/] {metadata.get('tool')} ({metadata.get('trash_id')})"
    )


def cmd_sandbox(file: pathlib.Path, net: bool = False, timeout: int = 30) -> None:
    cfg = Config.load()
    _ssh_preflight(cfg)
    backend = make_backend(cfg)
    project = get_active_project()
    code = pathlib.Path(file).read_text()
    cp = backend.run(code, network=net, timeout_s=timeout, project=project)
    stdout_tail = "\n".join((cp.stdout or "").splitlines()[-80:])
    stderr_tail = "\n".join((cp.stderr or "").splitlines()[-40:])
    G.console.print(
        Panel(
            stdout_tail or "(empty)",
            title=f"stdout (rc={cp.returncode})",
            border_style="green" if cp.returncode == 0 else "red",
        )
    )
    if stderr_tail:
        G.console.print(Panel(stderr_tail, title="stderr", border_style="red"))


def cmd_tools() -> None:
    catalog = load_catalog()
    if not catalog:
        G.console.print("[dim](tool body is empty — build some tools first)[/]")
        return
    table = Table(title=f"tool body ({len(catalog)} tools)", border_style="cyan")
    table.add_column("slug", style="bold")
    table.add_column("deliverable")
    table.add_column("run command")
    table.add_column("installed at")
    table.add_column("built")
    for slug, entry in sorted(catalog.items()):
        table.add_row(
            slug,
            entry.get("deliverable", "?"),
            entry.get("run_cmd", "?"),
            entry.get("installed_as") or "—",
            (entry.get("built_at") or "?")[:19],
        )
    G.console.print(table)


def cmd_install(slug: str) -> None:
    catalog = load_catalog()
    if slug not in catalog:
        raise G.MichaelError(f"unknown tool: {slug!r} — run `michael tools` to list")
    entry = catalog[slug]
    run_cmd = entry.get("run_cmd", "")
    deliverable = entry.get("deliverable", "")
    if not deliverable:
        raise G.MichaelError(f"no deliverable recorded for {slug!r}")

    # Locate the actual file from the run_cmd
    parts = run_cmd.split()
    target_file = pathlib.Path(parts[-1]).expanduser().resolve() if parts else None
    if not target_file or not target_file.is_file():
        raise G.MichaelError(
            f"deliverable file not found: {target_file}. "
            "The project workspace may have moved."
        )

    G.MICHAEL_BIN_DIR.mkdir(parents=True, exist_ok=True)
    link = G.MICHAEL_BIN_DIR / slug
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target_file)
    os.chmod(target_file, target_file.stat().st_mode | 0o111)

    catalog[slug]["installed_as"] = str(link)
    save_catalog(catalog)

    G.console.print(f"[green]installed[/] {slug} → {link}")
    path_str = str(G.MICHAEL_BIN_DIR)
    G.console.print(
        Panel(
            f"[dim]Add to PATH if not already present:[/]\n"
            f"  export PATH=\"{path_str}:$PATH\"",
            border_style="dim",
        )
    )


def cmd_deliver() -> None:
    project = require_active_project()
    found = detect_deliverable(project)
    if not found:
        G.console.print("[dim]no deliverable detected in project root[/]")
        return
    rel_path, run_cmd = found
    register_deliverable(project, rel_path, run_cmd)
    G.console.print(f"[green]registered[/] {project.slug}: {run_cmd}")


def cmd_ssh_test() -> None:
    cfg = Config.load()
    if not cfg.vps_active():
        raise G.MichaelError("vps.host is not configured")
    t0 = time.monotonic()
    cp = subprocess.run(
        _ssh_argv(cfg.vps) + ["echo ok && podman --version 2>/dev/null || true"],
        capture_output=True, text=True, timeout=15, check=False,
    )
    dt = round(time.monotonic() - t0, 3)
    if cp.returncode != 0:
        append_event("ssh.health", {"host": cfg.vps.host, "ok": False, "stderr": cp.stderr[:200]})
        raise G.MichaelError(f"ssh failed in {dt}s: {cp.stderr.strip()[:200]}")
    append_event("ssh.health", {"host": cfg.vps.host, "ok": True, "duration_s": dt})
    G.console.print(
        Panel(
            cp.stdout.strip() or "(no output)",
            title=f"ssh ok in {dt}s — {cfg.vps.user}@{cfg.vps.host}",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# Typer command bindings
# ---------------------------------------------------------------------------


@app.command(name="init")
def init_cmd() -> None:
    """Write a stub config file if missing. Idempotent."""
    cmd_init()


@app.command(name="show")
def show_cmd() -> None:
    """List projects."""
    cmd_show()


@app.command(name="new")
def new_cmd(
    name: Optional[str] = typer.Argument(None, help="Project name."),
) -> None:
    """Create a new project."""
    cmd_new(name)


@app.command(name="use")
def use_cmd(slug: str = typer.Argument(...)) -> None:
    """Set the active project."""
    cmd_use(slug)


@app.command(name="current")
def current_cmd() -> None:
    """Print the active project."""
    cmd_current()


@app.command(name="config")
def config_cmd() -> None:
    """Open the global config file in $EDITOR (with help panel)."""
    cmd_config()


@app.command(name="up")
def up_cmd() -> None:
    """Resume the GPU instance and wait for vLLM."""
    cmd_up()


@app.command(name="down")
def down_cmd() -> None:
    """Pause the GPU instance."""
    cmd_down()


@gpu_app.command("up")
def gpu_up_cmd() -> None:
    """Install vLLM on the GPU, start Qwen3-72B-AWQ, print the port-forward command."""
    cmd_gpu_up()


@gpu_app.command("down")
def gpu_down_cmd() -> None:
    """Kill vLLM on the GPU and stop the Vast.ai instance."""
    cmd_gpu_down()


@app.command(name="status")
def status_cmd() -> None:
    """Show derived state from the event log."""
    cmd_status()


@app.command(name="ask")
def ask_cmd(
    prompt: str = typer.Argument(..., help="One-shot prompt for the LLM."),
) -> None:
    """One-shot LLM call (uses active project's context if any)."""
    cmd_ask(prompt)


@app.command(name="run", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run_cmd(
    prompt: list[str] = typer.Argument(None, help="Prompt — every word after 'run' is the prompt."),
    no_verify: bool = typer.Option(False, "--no-verify", help="Skip auto-verify after Ja."),
) -> None:
    """Run the agent on a prompt. Everything after 'run' is the prompt.

    Example: michael run fix the auth bug in login.py
    """
    text = " ".join(prompt or []).strip()
    if not text:
        G.err.print("michael run requires a prompt. Example: michael run fix the login bug")
        raise typer.Exit(1)
    cmd_run(text, no_verify=no_verify)


@app.command(name="tools")
def tools_cmd() -> None:
    """List all tools in your personal tool body."""
    cmd_tools()


@app.command(name="install")
def install_cmd(
    slug: str = typer.Argument(..., help="Project slug of the tool to install."),
) -> None:
    """Install a built tool into ~/workbench/bin/ and symlink it."""
    cmd_install(slug)


@app.command(name="deliver")
def deliver_cmd() -> None:
    """Detect and register the deliverable in the active project without rebuilding."""
    cmd_deliver()


@app.command(name="log")
def log_cmd(
    tail: int = typer.Option(20, "--tail", "-n", help="How many events to show."),
) -> None:
    """Show the project event log (or global if no project active)."""
    cmd_log(tail)


@app.command(name="sandbox")
def sandbox_cmd(
    file: pathlib.Path = typer.Argument(..., exists=True, readable=True),
    net: bool = typer.Option(False, "--net", help="Allow bridge networking."),
    timeout: int = typer.Option(30, help="Wall-clock timeout in seconds."),
) -> None:
    """Run a Python file in the sandbox (local or VPS depending on config)."""
    cmd_sandbox(file, net, timeout)


@app.command(name="undo")
def undo_cmd(
    list_only: bool = typer.Option(False, "--list", "-l", help="List trash entries."),
    trash_id: Optional[str] = typer.Argument(None, help="Specific trash id to undo."),
) -> None:
    """Restore the most recent (or named) staged change."""
    cmd_undo(list_only=list_only, trash_id=trash_id)


@app.command(name="ssh-test")
def ssh_test_cmd() -> None:
    """Verify the VPS is reachable and report the SSH handshake time."""
    cmd_ssh_test()


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

REPL_COMMANDS = {
    "project", "new", "run", "up", "down", "config", "init",
    "tools", "install", "deliver",
    "quit", "exit", "help",
}


def _config_is_unset() -> bool:
    if not G.GLOBAL_CONFIG_PATH.is_file():
        return True
    try:
        cfg = Config.load()
    except G.MichaelError:
        return True
    if not cfg.vast_api_key:
        return True
    return not any(p.vast_instance_id for p in cfg.models.values())


class MichaelCompleter(Completer):
    """Tab-completion for the REPL."""

    def get_completions(self, document: Document, complete_event):
        text = document.text_before_cursor
        words = text.split()
        at_boundary = text.endswith(" ") or not text

        if not words or (len(words) == 1 and not at_boundary):
            prefix = words[0] if words else ""
            for cmd in sorted(REPL_COMMANDS):
                if cmd.startswith(prefix):
                    yield Completion(cmd, start_position=-len(prefix))
            return

        head = words[0]
        if head == "project":
            prefix = words[1] if len(words) > 1 and not at_boundary else ""
            for p in list_projects():
                if p.slug.startswith(prefix):
                    yield Completion(p.slug, start_position=-len(prefix))
            return


def repl() -> None:
    G.STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(G.REPL_HISTORY_PATH)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=MichaelCompleter(),
        complete_while_typing=False,
    )
    G.console.print("[bold cyan]michael[/] [dim]— event-sourced LLM loop[/]")
    if _config_is_unset():
        G.console.print(
            "[yellow]setup required[/] [dim]type: config[/]"
        )
    while True:
        try:
            line = session.prompt("michael> ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            continue
        if not line:
            continue
        if line in ("quit", "exit"):
            break
        try:
            dispatch_repl(line)
        except G.MichaelError as e:
            G.err.print(f"michael: {e}")
        except typer.Abort:
            G.err.print("aborted")
        except KeyboardInterrupt:
            G.err.print("interrupted")


def _opt_value(rest: list[str], *flags: str) -> Optional[str]:
    for f in flags:
        if f in rest:
            i = rest.index(f)
            if i + 1 < len(rest):
                return rest[i + 1]
    return None


def dispatch_repl(line: str) -> None:
    try:
        parts = shlex.split(line)
    except ValueError as e:
        G.err.print(f"parse error: {e}")
        return
    if not parts:
        return
    cmd, rest = parts[0], parts[1:]

    if cmd == "help":
        G.console.print(
            "commands:\n"
            "  run <prompt>          run the agent on a prompt\n"
            "  project [slug]        select/list projects\n"
            "  new [name]            create new project (defaults to ~/workbench/codebases/)\n"
            "  tools                 list your tool body (all tools ever built)\n"
            "  install <slug>        install a built tool into ~/workbench/bin/\n"
            "  deliver               register deliverable in active project\n"
            "  up / down             start/stop GPU (legacy — needs config.json)\n"
            "  gpu up / gpu down     install vLLM, start model, print port-forward\n"
            "  config                edit config\n"
            "  init                  initialize config\n"
            "  exit / quit           exit michael"
        )
        return

    if cmd == "init":
        cmd_init()
    elif cmd == "config":
        cmd_config()
    elif cmd == "project":
        if rest:
            cmd_use(rest[0])
        else:
            cmd_show()
    elif cmd == "new":
        name = " ".join(rest) if rest else None
        cmd_new(name)
    elif cmd == "run":
        if not rest:
            G.err.print("run requires a prompt. Example: run fix the auth bug")
            return
        no_verify = "--no-verify" in rest
        words = [w for w in rest if w != "--no-verify"]
        cmd_run(" ".join(words), no_verify=no_verify)
    elif cmd == "tools":
        cmd_tools()
    elif cmd == "install":
        if not rest:
            G.err.print("install requires a slug. Example: install csv-splitter")
            return
        cmd_install(rest[0])
    elif cmd == "deliver":
        cmd_deliver()
    elif cmd == "up":
        cmd_up()
    elif cmd == "down":
        cmd_down()
    elif cmd == "gpu":
        sub = rest[0] if rest else ""
        if sub == "up":
            cmd_gpu_up()
        elif sub == "down":
            cmd_gpu_down()
        else:
            G.err.print("usage: gpu up | gpu down")
    else:
        G.err.print(f"unknown command: {cmd!r}. try 'help'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    G.STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    try:
        if len(sys.argv) == 1:
            repl()
        else:
            app()
    except G.MichaelError as e:
        G.err.print(f"michael: {e}")
        sys.exit(2)
    except subprocess.CalledProcessError as e:
        G.err.print(f"command failed (exit {e.returncode})")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        G.err.print("interrupted")
        sys.exit(130)
