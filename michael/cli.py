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
    _build_vllm_cmd,
    _gpu_ssh_argv,
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
    set_active_slug,
    slugify,
)
from michael.agent import _load_dynamic_tools
from michael.tools import TOOLS, _list_trash, _undo_one, _dispatch_dynamic_tool_from_path
from michael.utils import (
    build_header,
    load_scripture,
    _prompt_history_lines,
    _action_log_lines,
)

app = typer.Typer(
    no_args_is_help=False,
    rich_markup_mode="rich",
    help="michael — air-gapped AI control loop",
)

gpu_app = typer.Typer(help="GPU instance management (A100 / vLLM).")
app.add_typer(gpu_app, name="gpu")

tools_app = typer.Typer(help="Inspect and run dynamic tools.")
app.add_typer(tools_app, name="tools")


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


_SHELL_MARKER = "# michael shell integration"
_SHELL_LINES = (
    "\n{marker}\n"
    "export PATH=\"{bin}:$PATH\"\n"
    "mcd() {{ cd \"$(michael path)\"; }}\n"
)


def _shell_profile() -> Optional[pathlib.Path]:
    shell = os.environ.get("SHELL", "")
    home = pathlib.Path.home()
    if "zsh" in shell:
        return home / ".zshrc"
    if "bash" in shell:
        for name in (".bashrc", ".bash_profile"):
            p = home / name
            if p.is_file():
                return p
        return home / ".bashrc"
    return None


def _inject_shell_integration() -> str:
    profile = _shell_profile()
    if profile is None:
        return "[yellow]unknown shell — add manually:[/]\n  export PATH=\"{bin}:$PATH\"\n  mcd() {{ cd \"$(michael path)\"; }}".format(bin=G.MICHAEL_BIN_DIR)
    text = profile.read_text() if profile.is_file() else ""
    if _SHELL_MARKER in text:
        return f"[dim]shell integration already in {profile}[/]"
    profile.parent.mkdir(parents=True, exist_ok=True)
    with profile.open("a") as f:
        f.write(_SHELL_LINES.format(marker=_SHELL_MARKER, bin=G.MICHAEL_BIN_DIR))
    return f"[green]wrote shell integration → {profile}[/]\n[dim]run: source {profile}[/]"


def cmd_init() -> None:
    G.STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    G.MICHAEL_BIN_DIR.mkdir(parents=True, exist_ok=True)
    if not G.GLOBAL_CONFIG_PATH.is_file():
        make_stub_config().save()
        G.console.print(f"[green]wrote stub[/] {G.GLOBAL_CONFIG_PATH}")
    else:
        G.console.print(f"[dim]config exists[/] {G.GLOBAL_CONFIG_PATH}")
    append_event("config.loaded", {"path": str(G.GLOBAL_CONFIG_PATH)})
    shell_msg = _inject_shell_integration()
    G.console.print(shell_msg)
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


def cmd_upgrade() -> None:
    repo_dir = pathlib.Path(__file__).parent.parent
    if not (repo_dir / ".git").is_dir():
        raise G.MichaelError(f"not a git repo: {repo_dir}")
    G.console.print(f"[dim]pulling {repo_dir}…[/]")
    cp = subprocess.run(
        ["git", "pull", "--ff-only"],
        cwd=str(repo_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise G.MichaelError(f"git pull failed:\n{cp.stderr.strip()}")
    G.console.print(f"[green]{cp.stdout.strip() or 'already up to date'}[/]")
    shell_msg = _inject_shell_integration()
    G.console.print(shell_msg)


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
    try:
        slug_preview = slugify(name)
    except G.MichaelError as e:
        G.err.print(str(e))
        return
    default_path = G.WORKBENCH_DIR / "codebases" / slug_preview
    path_str = typer.prompt("path", default=str(default_path))
    path = pathlib.Path(path_str).expanduser().resolve()
    proj = create_project(name, path)
    set_active_slug(proj.slug)
    append_event("project.activated", {"slug": proj.slug})
    G.console.print(f"[green]created[/] {proj.slug} at {proj.path}")
    G.console.print(f"[dim]workspace is empty — add your code there, then run: michael run <prompt>[/]")


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


def cmd_gpu_new(name: str, instance_id: str) -> None:
    cfg = Config.load()
    cfg.gpus[name] = GpuConfig(vast_instance_id=instance_id)
    cfg.save()
    G.console.print(f"[green]registered GPU[/] [bold]{name}[/] (instance {instance_id})")
    G.console.print(f"[dim]Run: michael gpu up {name}[/]")


def _pick_gpu(cfg: "Config", name: Optional[str]) -> tuple[str, GpuConfig]:
    """Return (name, GpuConfig) — prompts if multiple GPUs registered and name is None."""
    if not cfg.gpus and not cfg.gpu.ssh_host and not cfg.gpu.vast_instance_id:
        raise G.MichaelError("no GPUs registered — run: michael gpu new <name> <instance_id>")

    pool: dict[str, GpuConfig] = dict(cfg.gpus)
    if not pool and (cfg.gpu.ssh_host or cfg.gpu.vast_instance_id):
        pool["(legacy)"] = cfg.gpu

    if name:
        if name not in pool:
            raise G.MichaelError(
                f"unknown GPU {name!r}. Available: {', '.join(pool)}"
            )
        return name, pool[name]

    if len(pool) == 1:
        n, g = next(iter(pool.items()))
        G.console.print(f"[dim]Using GPU:[/] [bold]{n}[/]")
        return n, g

    names = list(pool)
    G.console.print("[bold cyan]Available GPUs:[/]")
    for i, n in enumerate(names, 1):
        g = pool[n]
        iid = g.vast_instance_id or "—"
        G.console.print(f"  {i}. [bold]{n}[/]  [dim](instance {iid})[/]")
    choice = typer.prompt("Which GPU?", default="1")
    try:
        idx = int(choice) - 1
        chosen = names[idx]
    except (ValueError, IndexError):
        raise G.MichaelError(f"invalid choice: {choice!r}")
    return chosen, pool[chosen]


def cmd_gpu_up(name: Optional[str] = None) -> None:
    cfg = Config.load()
    if not cfg.vast_api_key:
        raise G.MichaelError("vast_api_key is not set — run `michael config`")
    gname, gpu = _pick_gpu(cfg, name)
    if not gpu.vast_instance_id:
        raise G.MichaelError(
            f"GPU {gname!r} has no instance ID — run: michael gpu new {gname} <instance_id>"
        )
    G.console.print(f"[bold]Using GPU:[/] {gname}  [dim](instance {gpu.vast_instance_id})[/]")

    # ── Start instance ──
    vast = VastClient(cfg.vast_api_key)
    try:
        vast.start(gpu.vast_instance_id)
    except G.MichaelError as e:
        if "404" in str(e):
            raise G.MichaelError(
                f"Instance {gpu.vast_instance_id} not found — it may have been destroyed.\n"
                f"Register a new one: michael gpu new {gname} <new_instance_id>"
            ) from e
        raise
    finally:
        vast.close()

    G.console.print("[dim]start requested — waiting for SSH…[/]")

    # ── Wait for SSH host/port to appear in Vast API ──
    for _t in range(10, 181, 10):
        time.sleep(10)
        try:
            _vc = VastClient(cfg.vast_api_key)
            inst = _vc.get(gpu.vast_instance_id)
            _vc.close()
        except G.MichaelError:
            G.console.print(f"[dim]· {_t}s — waiting for instance metadata…[/]")
            continue
        status = inst.get("actual_status", "unknown")
        if status == "exited":
            raise G.MichaelError(
                f"Instance {gpu.vast_instance_id} exited before SSH was ready.\n"
                "Check Vast.ai console for the crash reason (OOM, driver error, etc.).\n"
                "You may need to destroy and re-rent, or SSH in manually first to verify."
            )
        fresh_host = inst.get("ssh_host") or inst.get("public_ipaddr") or ""
        fresh_port = int(inst.get("ssh_port") or 0)
        if fresh_host and fresh_port:
            if fresh_host != gpu.ssh_host or fresh_port != gpu.ssh_port:
                gpu.ssh_host = fresh_host
                gpu.ssh_port = fresh_port
                if gname in cfg.gpus:
                    cfg.gpus[gname] = gpu
                cfg.gpu = gpu
                cfg.save()
                G.console.print(f"[green]SSH:[/] {gpu.ssh_user}@{gpu.ssh_host}:{gpu.ssh_port}")
            break
        G.console.print(f"[dim]· {_t}s — status={status}, SSH details not assigned yet…[/]")
    else:
        raise G.MichaelError(
            "Instance did not expose SSH details within 180s — check Vast.ai console."
        )

    # ── Poll SSH until reachable ──
    import subprocess as _sp
    _last_ssh_err = ""
    for _t in range(10, 301, 10):
        time.sleep(10)
        # Also check instance hasn't crashed
        try:
            _vc = VastClient(cfg.vast_api_key)
            _inst = _vc.get(gpu.vast_instance_id)
            _vc.close()
            _status = _inst.get("actual_status", "unknown")
            if _status == "exited":
                raise G.MichaelError(
                    f"Instance crashed while waiting for SSH (status=exited).\n"
                    f"Last SSH error: {_last_ssh_err or '(none yet)'}\n"
                    "Check Vast.ai console for OOM/driver crash."
                )
        except G.MichaelError:
            raise
        except Exception:
            pass

        key = os.path.expanduser(gpu.ssh_key_path)
        _r = _sp.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no",
             "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=12",
             "-i", key, "-p", str(gpu.ssh_port),
             f"{gpu.ssh_user}@{gpu.ssh_host}", "echo ok"],
            capture_output=True, text=True, timeout=15, check=False,
        )
        if _r.returncode == 0:
            break
        _last_ssh_err = (_r.stderr or "").strip().splitlines()[-1] if _r.stderr else ""
        G.console.print(f"[dim]· {_t}s — SSH not ready ({_last_ssh_err or 'timeout'})[/]")
    else:
        raise G.MichaelError(
            f"SSH did not become reachable within 300s at "
            f"{gpu.ssh_user}@{gpu.ssh_host}:{gpu.ssh_port}\n"
            f"Last error: {_last_ssh_err or '(none)'}\n"
            "Verify your SSH key is uploaded in Vast.ai account settings:\n"
            "  https://cloud.vast.ai/account/ → SSH Keys"
        )

    # ── Install vLLM if missing ──
    cp = _gpu_ssh_run(gpu, "pip show vllm > /dev/null 2>&1 && echo installed || echo missing", timeout=15)
    if "missing" in cp.stdout:
        G.console.print("[cyan]Installing vLLM (takes a few minutes)…[/]")
        if _gpu_ssh_stream(gpu, "pip install vllm --quiet --upgrade", timeout=900) != 0:
            raise G.MichaelError("vLLM installation failed — check the instance terminal")
        G.console.print("[green]vLLM installed[/]")

    # ── Start vLLM ──
    already = _gpu_ssh_run(
        gpu,
        f"curl -sf http://localhost:{gpu.vllm_port}/v1/models > /dev/null 2>&1 && echo y || echo n",
    )
    if "y" in already.stdout:
        G.console.print("[dim]vLLM already serving[/]")
    else:
        _gpu_ssh_run(gpu, "pkill -f 'vllm serve' 2>/dev/null || true", timeout=10)
        time.sleep(2)
        pid = _gpu_ssh_run(gpu, _build_vllm_cmd(gpu), timeout=30).stdout.strip()
        G.console.print(f"[cyan]vLLM starting[/] (PID {pid}) — model load can take 20–40 min on first boot")

    # ── Poll until vLLM responds ──
    _max_wait_s = 5400
    _poll_s = 30
    _elapsed = 0
    endpoint: Optional[str] = None

    while _elapsed < _max_wait_s:
        time.sleep(_poll_s)
        _elapsed += _poll_s

        # Check locally via SSH first (works even if external port not mapped)
        local = _gpu_ssh_run(
            gpu,
            f"curl -sf http://localhost:{gpu.vllm_port}/v1/models > /dev/null 2>&1 && echo y || echo n",
            timeout=15,
        )
        if "y" not in local.stdout:
            alive = _gpu_ssh_run(gpu, "pgrep -f 'vllm serve' > /dev/null 2>&1 && echo y || echo n", timeout=10)
            if "n" in alive.stdout:
                tail = _gpu_ssh_run(gpu, "tail -20 /tmp/vllm.log 2>/dev/null", timeout=10).stdout.strip()
                raise G.MichaelError(f"vLLM process died.\n{tail}")
            log = _gpu_ssh_run(gpu, "tail -2 /tmp/vllm.log 2>/dev/null", timeout=10).stdout.strip()
            G.console.print(f"[dim]· {_elapsed}s — {log or 'loading…'}[/]")
            append_event("gpu.poll", {"elapsed_s": _elapsed})
            _poll_s = min(_poll_s * 2, 60)
            continue

        endpoint = f"http://localhost:{gpu.vllm_port}/v1"
        break

    if not endpoint:
        raise G.MichaelError(f"vLLM did not become ready within {_max_wait_s}s")

    # ── Save endpoint ──
    if "god" not in cfg.models:
        from michael.config import ModelProfile
        cfg.models["god"] = ModelProfile()
        cfg.default_model = cfg.default_model or "god"
    cfg.models["god"].endpoint = endpoint
    cfg.models["god"].served_model_name = gpu.model_repo
    if gname in cfg.gpus:
        cfg.gpus[gname] = gpu
    cfg.save()
    append_event("gpu.ready", {"instance": gpu.vast_instance_id, "model": gpu.model_repo, "endpoint": endpoint})

    G.console.print(
        Panel(
            f"[bold green]vLLM is ready[/] — {gpu.model_repo}\n\n"
            f"[bold]Open a new terminal and keep this running:[/]\n\n"
            f"  {gpu_port_forward_cmd(gpu)}\n\n"
            f"[dim]Keep that terminal open. Then:[/]\n"
            f"  michael run <your prompt>",
            title="port forward",
            border_style="green",
        )
    )


def cmd_gpu_down(name: Optional[str] = None) -> None:
    cfg = Config.load()
    gname, gpu = _pick_gpu(cfg, name)
    G.console.print(
        f"[bold]Stopping GPU:[/] {gname}  [dim](instance {gpu.vast_instance_id or '—'})[/]"
    )
    if not gpu.ssh_host:
        raise G.MichaelError(f"GPU {gname!r} has no SSH host — was it ever started?")

    # Kill vLLM via SSH (best-effort — instance may already be off)
    cp = _gpu_ssh_run(gpu, "pkill -f 'vllm serve' 2>/dev/null || true", timeout=15)
    if cp.returncode == 0:
        G.console.print("[yellow]vLLM stopped[/]")
    else:
        G.console.print("[dim]SSH unreachable — skipping vLLM kill (instance likely already off)[/]")

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


def cmd_gpu_debug(name: Optional[str] = None) -> None:
    import subprocess as _sp

    cfg = Config.load()
    gname, gpu = _pick_gpu(cfg, name)

    G.console.print(f"\n[bold cyan]═══ STEP 1: local config for {gname!r} ═══[/]")
    G.console.print(f"  instance_id : {gpu.vast_instance_id or '(not set)'}")
    G.console.print(f"  ssh_host    : {gpu.ssh_host or '(not set)'}")
    G.console.print(f"  ssh_port    : {gpu.ssh_port}")
    G.console.print(f"  ssh_user    : {gpu.ssh_user}")
    G.console.print(f"  ssh_key     : {gpu.ssh_key_path}")
    G.console.print(f"  model_repo  : {gpu.model_repo}")
    G.console.print(f"  vllm_port   : {gpu.vllm_port}")

    G.console.print(f"\n[bold cyan]═══ STEP 2: Vast.ai API response ═══[/]")
    if not cfg.vast_api_key:
        G.console.print("  [red]vast_api_key not set — skipping[/]")
    elif not gpu.vast_instance_id:
        G.console.print("  [red]vast_instance_id not set — skipping[/]")
    else:
        try:
            vast = VastClient(cfg.vast_api_key)
            inst = vast.get(gpu.vast_instance_id)
            vast.close()
            for key in ("actual_status", "ssh_host", "ssh_port", "public_ipaddr", "ports"):
                G.console.print(f"  {key}: {inst.get(key)!r}")
        except G.MichaelError as e:
            G.console.print(f"  [red]API error: {e}[/]")

    G.console.print(f"\n[bold cyan]═══ STEP 3: SSH command ═══[/]")
    argv = _gpu_ssh_argv(gpu)
    G.console.print("  " + " ".join(argv) + " echo ok")

    G.console.print(f"\n[bold cyan]═══ STEP 4: SSH verbose connect ═══[/]")
    key = os.path.expanduser(gpu.ssh_key_path)
    ssh_cmd = [
        "ssh", "-vvv",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=15",
        "-i", key,
        "-p", str(gpu.ssh_port),
        f"{gpu.ssh_user}@{gpu.ssh_host}",
        "echo __SSH_OK__",
    ]
    G.console.print(f"  running: {' '.join(ssh_cmd[:9])} …")
    result = _sp.run(ssh_cmd, capture_output=True, text=True, timeout=25, check=False)
    G.console.print(f"  exit code: {result.returncode}")
    if result.stdout:
        G.console.print("[bold]stdout:[/]\n" + result.stdout)
    if result.stderr:
        G.console.print("[bold]stderr (contains SSH debug):[/]\n" + result.stderr)

    if "__SSH_OK__" not in result.stdout:
        G.console.print("\n[red bold]SSH failed — share the stderr above to diagnose.[/]")
        return

    G.console.print(f"\n[bold cyan]═══ STEP 5: instance state ═══[/]")
    checks = (
        "uname -a",
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 || echo 'no nvidia-smi'",
        "pip show vllm 2>&1 | head -3 || echo 'vllm not installed'",
        "pgrep -fa 'vllm serve' 2>/dev/null || echo 'vllm not running'",
        "ss -tlnp 2>/dev/null | grep 8000 || echo 'nothing listening on 8000'",
        "curl -sf http://localhost:8000/v1/models 2>&1 && echo '\\nvllm HTTP ok' || echo 'vllm HTTP not ready'",
        "tail -30 /tmp/vllm.log 2>/dev/null || echo 'no /tmp/vllm.log'",
    )
    for cmd in checks:
        G.console.print(f"\n[dim]$ {cmd}[/]")
        cp = _gpu_ssh_run(gpu, cmd, timeout=20)
        G.console.print(cp.stdout or cp.stderr or "(no output)")


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
    client = llm_client(endpoint, profile.vllm_api_key, profile.enable_thinking)
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


def cmd_run(prompt: str) -> None:
    project = require_active_project()
    cfg = Config.load()
    name, profile = cfg.get_model()
    _run_agent_loop(project, cfg, name, profile, prompt, verb_label="run")


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


def cmd_inspect() -> None:
    project = require_active_project()
    cfg = Config.load()
    scripture = load_scripture(cfg.scripture_dir)
    header = build_header(project, cfg.resolved_system_prompt(), scripture)
    prompts = _prompt_history_lines(project)
    actions = _action_log_lines(project)
    G.console.print(f"\n[bold cyan]Project:[/] {project.name}  [dim]({project.slug})[/]")
    G.console.print(
        f"[dim]H1 prompts: {len(prompts)} · H3 tool calls: {len(actions)} · "
        f"context size: {len(header):,} chars[/]\n"
    )
    G.console.print(header)


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
# Catalog / install / deliver commands
# ---------------------------------------------------------------------------


def cmd_catalog() -> None:
    catalog = load_catalog()
    if not catalog:
        G.console.print("[dim]catalog is empty — deliver a tool first[/]")
        return
    table = Table(title=f"tool catalog ({len(catalog)} tools)", border_style="cyan")
    table.add_column("slug", style="bold")
    table.add_column("description")
    table.add_column("installed", style="green")
    table.add_column("built_at", style="dim")
    for slug, entry in sorted(catalog.items()):
        table.add_row(
            slug,
            str(entry.get("description", "—"))[:60],
            str(entry.get("installed_as") or "—"),
            str(entry.get("built_at", "—"))[:19],
        )
    G.console.print(table)


def cmd_install(slug: Optional[str]) -> None:
    catalog = load_catalog()
    if not catalog:
        raise G.MichaelError("catalog is empty — no tools to install")
    if slug is None:
        proj = get_active_project()
        if not proj:
            raise G.MichaelError("no active project and no slug given")
        slug = proj.slug
    entry = catalog.get(slug)
    if not entry:
        raise G.MichaelError(f"tool {slug!r} not found in catalog")
    deliverable = entry.get("deliverable", "")
    if not deliverable:
        raise G.MichaelError(f"no deliverable path recorded for {slug!r}")
    src = pathlib.Path(deliverable).expanduser()
    if not src.is_file():
        raise G.MichaelError(f"deliverable not found: {src}")
    G.MICHAEL_BIN_DIR.mkdir(parents=True, exist_ok=True)
    link = G.MICHAEL_BIN_DIR / slug
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(src)
    if not src.stat().st_mode & 0o111:
        src.chmod(src.stat().st_mode | 0o755)
    run_cmd = str(link)
    from michael.project import save_catalog
    catalog[slug]["installed_as"] = str(link)
    catalog[slug]["run_cmd"] = run_cmd
    save_catalog(catalog)
    G.console.print(
        Panel(
            f"[bold green]installed[/] {slug}\n"
            f"  symlink: {link} → {src}\n\n"
            f"Add to PATH:\n  export PATH=\"{G.MICHAEL_BIN_DIR}:$PATH\"",
            title="michael install",
            border_style="green",
        )
    )


def cmd_path() -> None:
    p = get_active_project()
    if not p:
        raise G.MichaelError("no active project")
    G.console.print(p.path)


def cmd_deliver() -> None:
    project = require_active_project()
    det = detect_deliverable(project)
    if not det:
        raise G.MichaelError("no deliverable detected in this project (look for main.py, app.py, *.sh, etc.)")
    deliverable, run_cmd = det
    register_deliverable(project, deliverable, run_cmd)
    G.console.print(
        Panel(
            f"[bold green]delivered[/] {deliverable}\n"
            f"installed: [cyan]{G.MICHAEL_BIN_DIR / project.slug}[/]\n\n"
            f"[dim]Add to PATH: export PATH=\"{G.MICHAEL_BIN_DIR}:$PATH\"[/]",
            title="michael deliver",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# Tools workspace commands
# ---------------------------------------------------------------------------

_TOOL_DIR_LABELS = [
    ("bundled", pathlib.Path(__file__).parent.parent / "toolbox"),
    ("global",  pathlib.Path(G.GLOBAL_TOOLS_DIR)),
]


def _tool_search_dirs(project_path: str | None) -> list[tuple[str, pathlib.Path]]:
    dirs = list(_TOOL_DIR_LABELS)
    if project_path:
        dirs.append(("project", pathlib.Path(project_path) / "tools"))
    return dirs


def _parse_kv_args(tokens: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise typer.BadParameter(f"expected key=value, got {token!r}")
        k, _, v = token.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def _find_tool_file(name: str, project_path: str | None) -> pathlib.Path | None:
    # Project-local takes priority, then global, then bundled.
    search = list(reversed(_tool_search_dirs(project_path)))
    for _label, d in search:
        candidate = d / f"{name}.py"
        if candidate.exists():
            return candidate
    return None


def cmd_tools_list() -> None:
    project_path: str | None = None
    try:
        project_path = require_active_project().path
    except G.MichaelError:
        pass

    import importlib.util as _ilu

    rows: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    # Reverse priority so highest-priority entry wins display
    for label, d in reversed(_tool_search_dirs(project_path)):
        if not d.is_dir():
            continue
        for py_file in sorted(d.glob("*.py")):
            try:
                spec = _ilu.spec_from_file_location(py_file.stem, py_file)
                mod = _ilu.module_from_spec(spec)       # type: ignore[arg-type]
                spec.loader.exec_module(mod)             # type: ignore[union-attr]
                if not hasattr(mod, "TOOL_SCHEMA"):
                    continue
                fn_name = mod.TOOL_SCHEMA.get("function", {}).get("name", py_file.stem)
                if fn_name in seen:
                    continue
                seen.add(fn_name)
                desc = mod.TOOL_SCHEMA.get("function", {}).get("description", "")
                desc = desc.strip().splitlines()[0][:72] if desc else ""
                rows.append((fn_name, desc, label))
            except Exception as exc:
                G.err.print(f"[dim]skipped {py_file.name}: {exc}[/]")

    if not rows:
        G.console.print("[dim]no dynamic tools found[/]")
        return

    t = Table(show_header=True, header_style="bold", box=None, pad_edge=False, min_width=60)
    t.add_column("Name", style="cyan", no_wrap=True)
    t.add_column("Description", no_wrap=False)
    t.add_column("Source", style="dim", no_wrap=True)
    for name, desc, label in sorted(rows, key=lambda r: r[0]):
        t.add_row(name, desc, label)
    G.console.print(t)


def cmd_tools_run(name: str, kv_tokens: list[str]) -> None:
    project: Optional[Any] = None
    try:
        project = require_active_project()
    except G.MichaelError:
        pass

    py_file = _find_tool_file(name, project.path if project else None)
    if py_file is None:
        raise G.MichaelError(f"tool {name!r} not found in any toolbox directory")

    try:
        args = _parse_kv_args(kv_tokens)
    except typer.BadParameter as e:
        raise G.MichaelError(str(e)) from e

    result = _dispatch_dynamic_tool_from_path(name, args, py_file, project)
    G.console.print(result)


def cmd_tools_show(name: str) -> None:
    project_path: str | None = None
    try:
        project_path = require_active_project().path
    except G.MichaelError:
        pass

    py_file = _find_tool_file(name, project_path)
    if py_file is None:
        raise G.MichaelError(f"tool {name!r} not found in any toolbox directory")

    from rich.syntax import Syntax
    G.console.print(f"[dim]{py_file}[/]")
    G.console.print(Syntax(py_file.read_text(), "python", line_numbers=True))


# ---------------------------------------------------------------------------
# Typer command bindings
# ---------------------------------------------------------------------------


@app.command(name="init")
def init_cmd() -> None:
    """Write stub config, create workbench dirs, inject shell integration. Idempotent."""
    cmd_init()


@app.command(name="upgrade")
def upgrade_cmd() -> None:
    """git pull the michael repo and re-run shell integration."""
    cmd_upgrade()


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


@gpu_app.command("new")
def gpu_new_cmd(
    name: str = typer.Argument(..., help="Short name for this GPU (e.g. 'rtx6000')."),
    instance_id: str = typer.Argument(..., help="Vast.ai instance ID."),
) -> None:
    """Register a named GPU instance by Vast.ai instance ID."""
    cmd_gpu_new(name, instance_id)


@gpu_app.command("up")
def gpu_up_cmd(
    name: Optional[str] = typer.Argument(None, help="GPU name (from 'gpu new'). Prompts if omitted."),
) -> None:
    """Start vLLM on a named GPU instance."""
    cmd_gpu_up(name)


@gpu_app.command("down")
def gpu_down_cmd(
    name: Optional[str] = typer.Argument(None, help="GPU name (from 'gpu new'). Prompts if omitted."),
) -> None:
    """Kill vLLM on a named GPU instance and stop it via Vast.ai API."""
    cmd_gpu_down(name)


@gpu_app.command("debug")
def gpu_debug_cmd(
    name: Optional[str] = typer.Argument(None, help="GPU name. Prompts if omitted."),
) -> None:
    """Run full SSH + vLLM diagnostics and print everything."""
    cmd_gpu_debug(name)


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
) -> None:
    """Run the agent on a prompt. Everything after 'run' is the prompt.

    Example: michael run fix the auth bug in login.py
    """
    text = " ".join(prompt or []).strip()
    if not text:
        G.err.print("michael run requires a prompt. Example: michael run fix the login bug")
        raise typer.Exit(1)
    cmd_run(text)


@app.command(name="log")
def log_cmd(
    tail: int = typer.Option(20, "--tail", "-n", help="How many events to show."),
) -> None:
    """Show the project event log (or global if no project active)."""
    cmd_log(tail)


@app.command(name="inspect")
def inspect_cmd() -> None:
    """Print the full H1–H4 context package the model will receive on the next run."""
    cmd_inspect()


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


@app.command(name="catalog")
def catalog_cmd() -> None:
    """List all delivered tools in the global catalog."""
    cmd_catalog()


@app.command(name="path")
def path_cmd() -> None:
    """Print the active project's workspace path (useful for cd $(michael path))."""
    cmd_path()


@app.command(name="deliver")
def deliver_cmd() -> None:
    """Detect, register, and install the active project's deliverable."""
    cmd_deliver()


@app.command(name="install", hidden=True)
def install_cmd(
    slug: Optional[str] = typer.Argument(None, help="Tool slug to reinstall."),
) -> None:
    """Reinstall a delivered tool's wrapper script (repair command)."""
    cmd_install(slug)


@tools_app.command(name="list")
def tools_list_cmd() -> None:
    """List all dynamic tools across bundled, global, and project toolboxes."""
    cmd_tools_list()


@tools_app.command(
    name="run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def tools_run_cmd(
    name: str = typer.Argument(..., help="Tool name to invoke."),
    ctx: typer.Context = typer.Option(None, hidden=True),
) -> None:
    """Run a dynamic tool by name. Pass arguments as key=value pairs."""
    cmd_tools_run(name, ctx.args if ctx else [])


@tools_app.command(name="show")
def tools_show_cmd(
    name: str = typer.Argument(..., help="Tool name to inspect."),
) -> None:
    """Print the source code of a dynamic tool."""
    cmd_tools_show(name)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------

REPL_COMMANDS = {
    "project", "new", "run", "up", "down", "gpu", "config", "init",
    "tools", "quit", "exit", "help",
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
            "  run <prompt>                      run the agent on a prompt\n"
            "  project [slug]                    select/list projects\n"
            "  new [name]                        create new project\n"
            "  up / down                         start/stop GPU (legacy — needs config.json)\n"
            "  gpu up / gpu down                 start instance, install vLLM, serve model\n"
            "  tools list                        list all dynamic tools\n"
            "  tools run <name> [key=value ...]  run a dynamic tool directly\n"
            "  tools show <name>                 print tool source\n"
            "  catalog                           list all delivered tools\n"
            "  path                              print active project workspace path\n"
            "  deliver                           detect + install active project's deliverable\n"
            "  config                            edit config\n"
            "  init                              initialize config + shell integration\n"
            "  upgrade                           git pull + re-apply shell integration\n"
            "  exit / quit                       exit michael"
        )
        return

    if cmd == "init":
        cmd_init()
    elif cmd == "upgrade":
        cmd_upgrade()
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
        cmd_run(" ".join(rest))
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
    elif cmd == "tools":
        sub = rest[0] if rest else "list"
        if sub == "list":
            cmd_tools_list()
        elif sub == "run":
            if len(rest) < 2:
                G.err.print("usage: tools run <name> [key=value ...]")
            else:
                cmd_tools_run(rest[1], rest[2:])
        elif sub == "show":
            if len(rest) < 2:
                G.err.print("usage: tools show <name>")
            else:
                cmd_tools_show(rest[1])
        else:
            G.err.print("usage: tools list | tools run <name> [key=value ...] | tools show <name>")
    elif cmd == "catalog":
        cmd_catalog()
    elif cmd == "path":
        cmd_path()
    elif cmd == "deliver":
        cmd_deliver()
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
