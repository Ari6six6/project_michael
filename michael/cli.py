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
from rich.syntax import Syntax
from rich.table import Table

import michael.globals as G
from michael.agent import _resolve_nitro_model, _resolve_tier, _run_agent_loop, _tools_for_mode
from michael.backends import (
    VastClient,
    _ping_vllm,
    _require_endpoint,
    _ssh_argv,
    _ssh_preflight,
    chat_stream,
    llm_client,
    make_backend,
)
from michael.config import Config, CONFIG_HELP, make_stub_config
from michael.project import (
    Project,
    append_event,
    create_project,
    get_active_project,
    get_active_slug,
    iter_events,
    list_projects,
    replay_global,
    require_active_project,
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
            "  [bold]default_model[/]             which profile to use by default\n"
            "  [bold]models.<name>[/]             one entry per Vast.ai instance:\n"
            "    vast_instance_id              numeric instance id\n"
            "    served_model_name             matches --served-model-name on vLLM\n"
            "    vllm_api_key                  the key vLLM was launched with\n\n"
            "[dim]Optional, for remote sandbox on the VPS:[/]\n"
            "  [bold]vps.host[/]                  VPS public IP/hostname\n"
            "  [bold]vps.user[/]                  ssh user (default: michael)\n"
            "  [bold]vps.ssh_key_path[/]          path to private key\n"
            "  [bold]vps.workspace_dir[/]         /home/michael/workspace\n\n"
            "[dim]Leave vps.host empty to run chat-only (no sandbox).[/]",
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
    default_path = pathlib.Path.cwd() / slugify(name)
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


def cmd_up(model: Optional[str]) -> None:
    cfg = Config.load()
    name, profile = cfg.get_model(model or None)
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


def cmd_down(model: Optional[str]) -> None:
    cfg = Config.load()
    name, profile = cfg.get_model(model or None)
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
        table.add_row("vps", "[dim]not configured (chat-only)[/]")

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


def cmd_ask(prompt: str, model: Optional[str]) -> None:
    cfg = Config.load()
    name, profile = cfg.get_model(model or None)
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


def cmd_run(
    coder: bool = False,
    instruct: bool = False,
    hacker: bool = False,
    model: Optional[str] = None,
    legacy: bool = False,
) -> None:
    project = require_active_project()
    cfg = Config.load()
    if model:
        name, profile = cfg.get_model(model)
        mode, god = "code", False
    elif hacker:
        name, profile, mode, god = _resolve_tier(cfg, "hacker")
    elif instruct:
        name, profile, mode, god = _resolve_tier(cfg, "instruct")
    elif coder:
        name, profile, mode, god = _resolve_tier(cfg, "coder")
    else:
        name, profile = cfg.get_model(None)
        mode, god = "code", False
    use_kantian = not legacy and cfg.use_stateful_kantian
    _run_agent_loop(project, cfg, name, profile, mode=mode, verb_label="run", god_mode=god, use_kantian=use_kantian)


def cmd_new_code(model: Optional[str]) -> None:
    G.console.print("[yellow]'new code' is deprecated — use 'run' or 'run --coder'[/]")
    cmd_run(coder=True, model=model)


def cmd_new_discussion(model: Optional[str]) -> None:
    G.console.print("[yellow]'new discussion' is deprecated — use 'run --instruct'[/]")
    cmd_run(instruct=True, model=model)


def cmd_nitro(model: Optional[str], god: bool = False) -> None:
    flag = "--hacker" if god else "--instruct"
    G.console.print(f"[yellow]'nitro' is deprecated — use 'run {flag}'[/]")
    if god:
        cmd_run(hacker=True, model=model)
    else:
        cmd_run(instruct=True, model=model)


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
    keyword: Optional[str] = typer.Argument(
        None,
        help="'project' (create), 'code' (code agent loop), 'discussion' "
             "(read-only chat), or the project name.",
    ),
    name: Optional[str] = typer.Argument(None, help="Project name (if 'project' was passed)"),
    model: str = typer.Option(
        "", "--model", "-m",
        help="Model profile (only meaningful for 'new code' / 'new discussion').",
    ),
) -> None:
    """Create a new project, or start a fresh `code`/`discussion` agent loop."""
    if keyword == "code":
        cmd_new_code(model or None)
        return
    if keyword == "discussion":
        cmd_new_discussion(model or None)
        return
    if keyword == "project":
        actual_name = name
    elif keyword and name is None:
        actual_name = keyword
    else:
        actual_name = name
    cmd_new(actual_name)


@app.command(name="nitro")
def nitro_cmd(
    model: str = typer.Option(
        "", "--model", "-m",
        help="Override the heavy-model profile (defaults to 'nitro' then 'big').",
    ),
    god: bool = typer.Option(
        False, "--god",
        help="God mode: hardcoded prompt, changes auto-apply on Ja (no approval gate).",
    ),
) -> None:
    """Fresh agent loop on the heavy model (cold-start aware)."""
    cmd_nitro(model or None, god=god)


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
def up_cmd(
    model: str = typer.Option("", "--model", "-m", help="Model profile name."),
) -> None:
    """Resume a Vast.ai instance and wait for vLLM."""
    cmd_up(model or None)


@app.command(name="down")
def down_cmd(
    model: str = typer.Option("", "--model", "-m", help="Model profile name."),
) -> None:
    """Pause a Vast.ai instance."""
    cmd_down(model or None)


@app.command(name="status")
def status_cmd() -> None:
    """Show derived state from the event log."""
    cmd_status()


@app.command(name="ask")
def ask_cmd(
    prompt: str = typer.Argument(..., help="One-shot prompt for the LLM."),
    model: str = typer.Option("", "--model", "-m", help="Model profile name."),
) -> None:
    """One-shot LLM call (uses active project's brief if any)."""
    cmd_ask(prompt, model or None)


@app.command(name="run")
def run_cmd(
    coder:   bool = typer.Option(False, "--coder",   help="Coder tier — Qwen 30B Coder."),
    instruct: bool = typer.Option(False, "--instruct", help="Instruct tier — Qwen 30B Instruct."),
    hacker:  bool = typer.Option(False, "--hacker",  help="Hacker tier — Qwen 235B, god mode."),
    model:   str  = typer.Option("",   "--model", "-m", help="Power-user: exact profile name."),
    legacy:  bool = typer.Option(False, "--legacy", help="Use stateless loop (disable Kantian machine)."),
) -> None:
    """Agent loop. Default tier: coder. Use --coder / --instruct / --hacker to select tier."""
    cmd_run(coder=coder, instruct=instruct, hacker=hacker, model=model or None, legacy=legacy)


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
    "show", "new", "use", "current", "config", "init",
    "up", "down", "status",
    "ask", "run", "nitro", "log", "sandbox", "undo", "ssh-test",
    "quit", "exit", "help",
}

NEW_SUBCOMMANDS = ("project", "code", "discussion")


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

    LOG_FLAGS = ("--tail", "-n")
    UP_FLAGS = ("--model", "-m")
    DOWN_FLAGS = ("--model", "-m")
    NITRO_FLAGS = ("--model", "-m", "--god")
    UNDO_FLAGS = ("--list", "-l")

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
        if head == "use":
            prefix = words[1] if len(words) > 1 and not at_boundary else ""
            for p in list_projects():
                if p.slug.startswith(prefix):
                    yield Completion(p.slug, start_position=-len(prefix))
            return
        if head == "new":
            if len(words) == 1 and at_boundary:
                for sub in NEW_SUBCOMMANDS:
                    yield Completion(sub, start_position=0)
            elif len(words) == 2 and not at_boundary:
                for sub in NEW_SUBCOMMANDS:
                    if sub.startswith(words[1]):
                        yield Completion(sub, start_position=-len(words[1]))
            return
        if head == "log":
            prefix = words[-1] if not at_boundary else ""
            for f in self.LOG_FLAGS:
                if f.startswith(prefix):
                    yield Completion(f, start_position=-len(prefix))
            return
        if head in ("up", "down", "run", "ask"):
            prefix = words[-1] if not at_boundary else ""
            for f in self.UP_FLAGS:
                if f.startswith(prefix):
                    yield Completion(f, start_position=-len(prefix))
            return
        if head == "nitro":
            prefix = words[-1] if not at_boundary else ""
            for f in self.NITRO_FLAGS:
                if f.startswith(prefix):
                    yield Completion(f, start_position=-len(prefix))
            return
        if head == "undo":
            prefix = words[-1] if not at_boundary else ""
            for f in self.UNDO_FLAGS:
                if f.startswith(prefix):
                    yield Completion(f, start_position=-len(prefix))
            return


def repl() -> None:
    G.STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(G.REPL_HISTORY_PATH)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=MichaelCompleter(),
        complete_while_typing=False,
    )
    G.console.print("hey")
    if _config_is_unset():
        G.console.print(
            "[yellow]no config yet — type `config` to set up your vast.ai keys, "
            "vllm key, and model[/]"
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
        G.console.print("commands: " + ", ".join(sorted(REPL_COMMANDS)))
        return
    if cmd == "show":
        cmd_show()
    elif cmd == "init":
        cmd_init()
    elif cmd == "new":
        if rest and rest[0] == "code":
            cmd_new_code(_opt_value(rest[1:], "--model", "-m"))
        elif rest and rest[0] == "discussion":
            cmd_new_discussion(_opt_value(rest[1:], "--model", "-m"))
        else:
            if rest and rest[0] == "project":
                rest = rest[1:]
            name = " ".join(rest) if rest else None
            cmd_new(name)
    elif cmd == "use":
        if not rest:
            G.err.print("usage: use <slug>")
            return
        cmd_use(rest[0])
    elif cmd == "current":
        cmd_current()
    elif cmd == "config":
        cmd_config()
    elif cmd == "up":
        cmd_up(_opt_value(rest, "--model", "-m"))
    elif cmd == "down":
        cmd_down(_opt_value(rest, "--model", "-m"))
    elif cmd == "status":
        cmd_status()
    elif cmd == "ask":
        model = _opt_value(rest, "--model", "-m")
        prompt_parts = []
        skip = 0
        for i, tok in enumerate(rest):
            if skip:
                skip -= 1
                continue
            if tok in ("--model", "-m"):
                skip = 1
                continue
            prompt_parts.append(tok)
        if not prompt_parts:
            G.err.print("usage: ask <prompt> [--model NAME]")
            return
        cmd_ask(" ".join(prompt_parts), model)
    elif cmd == "run":
        cmd_run(
            coder="--coder" in rest,
            instruct="--instruct" in rest,
            hacker="--hacker" in rest,
            model=_opt_value(rest, "--model", "-m"),
        )
    elif cmd == "nitro":
        cmd_nitro(
            _opt_value(rest, "--model", "-m"),
            god="--god" in rest,
        )
    elif cmd == "log":
        n = 20
        if (v := _opt_value(rest, "--tail", "-n")) is not None:
            try:
                n = int(v)
            except ValueError:
                pass
        cmd_log(n)
    elif cmd == "sandbox":
        if not rest:
            G.err.print("usage: sandbox <file>")
            return
        cmd_sandbox(pathlib.Path(rest[0]))
    elif cmd == "undo":
        list_only = "--list" in rest or "-l" in rest
        positional = [r for r in rest if r not in ("--list", "-l")]
        target = positional[0] if positional else None
        cmd_undo(list_only=list_only, trash_id=target)
    elif cmd == "ssh-test":
        cmd_ssh_test()
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
