"""
michael — air-gapped, event-sourced, AI-native control loop.

Runs on a constrained Ubuntu 24.04 LTS VPS (2 vCPUs, 2 GB RAM) and orchestrates
a remote Vast.ai GPU instance that serves an open-source coding LLM via a
vLLM OpenAI-compatible API. LLM-proposed code is executed in ephemeral,
network-isolated Podman/Docker sandboxes after a human Y/n/edit confirmation.

State is never mutated: every transition is appended to a JSONL event log and
the live state is a pure fold over that log. No daemons, no databases, no
phone-home.

Subcommands:
    michael up                  resume the Vast.ai instance, wait for vLLM
    michael down                pause it (preserve disk, stop GPU billing)
    michael status              current state, derived by replaying events
    michael ask "..."           one-shot LLM call
    michael run                 interactive control loop: ask -> diff -> Y/n/e -> sandbox
    michael log [--tail N]      show event log
    michael sandbox <file.py>   run a file in the throwaway sandbox
    michael init                write a stub config file
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import httpx
import typer
from openai import OpenAI
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

STATE_DIR = pathlib.Path.home() / ".michael"
EVENTS_PATH = STATE_DIR / "events.jsonl"
CONFIG_PATH = STATE_DIR / "config.json"

console = Console()
err = Console(stderr=True, style="bold red")

app = typer.Typer(
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="michael — air-gapped AI control loop",
)

RUNTIME = "podman" if shutil.which("podman") else "docker"


class MichaelError(RuntimeError):
    """Domain error surfaced to the user with a clean message."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    vast_api_key: str = ""
    vast_instance_id: str = ""
    vllm_api_key: str = ""
    vllm_internal_port: int = 8000
    model_name: str = "qwen3-coder"
    sandbox_image: str = "michael-sandbox:alpine"
    sandbox_memory_mb: int = 384
    sandbox_cpus: float = 1.5
    sandbox_pids: int = 128
    request_timeout_s: int = 60
    boot_poll_s: int = 10
    endpoint: Optional[str] = None  # cached after `up`

    @classmethod
    def load(cls) -> "Config":
        data: dict[str, Any] = {}
        if CONFIG_PATH.is_file():
            try:
                data = json.loads(CONFIG_PATH.read_text())
            except json.JSONDecodeError as e:
                raise MichaelError(f"config.json is not valid JSON: {e}") from e
        env_overrides = {
            "vast_api_key": os.environ.get("VAST_API_KEY"),
            "vast_instance_id": os.environ.get("VAST_INSTANCE_ID"),
            "vllm_api_key": os.environ.get("VLLM_API_KEY"),
            "model_name": os.environ.get("MICHAEL_MODEL"),
        }
        for k, v in env_overrides.items():
            if v:
                data[k] = v
        valid = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in valid}
        return cls(**clean)

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        os.chmod(CONFIG_PATH, 0o600)


# ---------------------------------------------------------------------------
# Event log — the heart of the system
# ---------------------------------------------------------------------------


def _last_seq() -> int:
    """Return the seq of the last event, or 0 if the log is empty."""
    if not EVENTS_PATH.is_file():
        return 0
    last: Optional[str] = None
    with EVENTS_PATH.open("rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n":
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        tail = f.readline().decode("utf-8", errors="replace").strip()
        if tail:
            last = tail
    if not last:
        return 0
    try:
        return int(json.loads(last).get("seq", 0))
    except (json.JSONDecodeError, ValueError):
        return 0


def append_event(type_: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Append one JSON line to EVENTS_PATH, fsync, and echo a summary."""
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    seq = _last_seq() + 1
    event = {
        "seq": seq,
        "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "type": type_,
        "payload": payload,
    }
    line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
    with EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    style = "red" if type_ == "error" else "dim"
    console.print(f"[{style}]· {seq:>4} {type_}[/]", highlight=False)
    return event


def _iter_events() -> Iterable[dict[str, Any]]:
    if not EVENTS_PATH.is_file():
        return []
    out: list[dict[str, Any]] = []
    with EVENTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def replay() -> dict[str, Any]:
    """Pure fold over the event log into a derived state dict."""
    state: dict[str, Any] = {
        "instance_state": "unknown",
        "endpoint": None,
        "last_prompt": None,
        "last_poll_ts": None,
        "patches_pending": [],
        "sandbox_runs": 0,
        "errors": 0,
    }
    for ev in _iter_events():
        t = ev.get("type", "")
        p = ev.get("payload", {}) or {}
        if t == "instance.start_requested":
            state["instance_state"] = "starting"
        elif t == "instance.started":
            state["instance_state"] = "running"
        elif t == "instance.stop_requested":
            state["instance_state"] = "stopping"
        elif t == "instance.stopped":
            state["instance_state"] = "stopped"
        elif t == "instance.poll":
            state["last_poll_ts"] = ev.get("ts")
            if "actual_status" in p:
                state["instance_state"] = p["actual_status"]
        elif t == "endpoint.discovered":
            state["endpoint"] = p.get("endpoint")
        elif t == "prompt.sent":
            state["last_prompt"] = p.get("prompt")
        elif t == "patch.proposed":
            state["patches_pending"].append(
                {
                    "id": p.get("id"),
                    "target": p.get("target"),
                    "old": p.get("old", ""),
                    "new": p.get("new", ""),
                }
            )
        elif t in ("patch.accepted", "patch.rejected", "patch.edited"):
            pid = p.get("id")
            state["patches_pending"] = [
                x for x in state["patches_pending"] if x.get("id") != pid
            ]
        elif t == "sandbox.run":
            state["sandbox_runs"] += 1
        elif t == "error":
            state["errors"] += 1
    return state


# ---------------------------------------------------------------------------
# Vast.ai client (plain httpx — no `vastai` library)
# ---------------------------------------------------------------------------

# On Vast.ai, rent 1× H100/H200 80 GB and launch the vLLM template with:
#   Image:      vllm/vllm-openai:v0.20.0
#   Env:        -p 8000:8000  -e HF_TOKEN=...  -e VLLM_API_KEY=...
#   Run args:
#     --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
#     --served-model-name qwen3-coder
#     --max-model-len 131072
#     --gpu-memory-utilization 0.92
#     --enable-auto-tool-choice --tool-call-parser qwen3_coder
#     --api-key ${VLLM_API_KEY}
# Budget alternative on 1× RTX 4090 24 GB:
#     --model tclf90/Qwen3-Coder-30B-A3B-Instruct-AWQ
#     --quantization awq_marlin --max-model-len 32768


class VastClient:
    def __init__(self, api_key: str, base: str = "https://console.vast.ai/api/v0") -> None:
        if not api_key:
            raise MichaelError("VAST_API_KEY is not set")
        self.base = base.rstrip("/")
        self._client = httpx.Client(
            timeout=httpx.Timeout(10.0, read=60.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
        )

    def _wrap(self, fn_name: str, request: "callable[..., httpx.Response]") -> Any:
        try:
            r = request()
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            body = (e.response.text or "")[:200]
            msg = f"vast {fn_name}: HTTP {e.response.status_code} — {body}"
            append_event("error", {"where": fn_name, "msg": msg})
            raise MichaelError(msg) from e
        except httpx.HTTPError as e:
            msg = f"vast {fn_name}: {e}"
            append_event("error", {"where": fn_name, "msg": msg})
            raise MichaelError(msg) from e

    def list_instances(self) -> list[dict[str, Any]]:
        data = self._wrap("list_instances", lambda: self._client.get(f"{self.base}/instances/"))
        return data.get("instances", []) or []

    def get(self, inst_id: str | int) -> dict[str, Any]:
        data = self._wrap(
            "get",
            lambda: self._client.get(f"{self.base}/instances/{inst_id}/"),
        )
        return data.get("instances", {}) or {}

    def start(self, inst_id: str | int) -> dict[str, Any]:
        return self._wrap(
            "start",
            lambda: self._client.put(
                f"{self.base}/instances/{inst_id}/", json={"state": "running"}
            ),
        )

    def stop(self, inst_id: str | int) -> dict[str, Any]:
        return self._wrap(
            "stop",
            lambda: self._client.put(
                f"{self.base}/instances/{inst_id}/", json={"state": "stopped"}
            ),
        )

    def endpoint_for(self, inst_id: str | int, internal_port: int = 8000) -> Optional[str]:
        """Return f"http://{ip}:{HostPort}/v1" or None when the mapping isn't ready.

        Defends against both schemas: documented int-array (`ports: [8080,...]`)
        and the actual returned dict (`ports: {"8000/tcp":[{"HostPort":"33526"}]}`).
        """
        inst = self.get(inst_id)
        if not inst:
            return None
        ip = inst.get("public_ipaddr") or inst.get("ssh_host")
        if not ip:
            return None
        ports = inst.get("ports")
        host_port: Optional[int] = None
        if isinstance(ports, dict):
            mappings = ports.get(f"{internal_port}/tcp") or ports.get(str(internal_port))
            if isinstance(mappings, list) and mappings:
                hp = mappings[0].get("HostPort")
                if hp is not None:
                    try:
                        host_port = int(hp)
                    except (TypeError, ValueError):
                        host_port = None
        elif isinstance(ports, list):
            # documented form — assume same external port; some templates expose it directly
            for entry in ports:
                if isinstance(entry, int) and entry == internal_port:
                    host_port = internal_port
                    break
        if host_port is None:
            return None
        return f"http://{ip}:{host_port}/v1"

    def close(self) -> None:
        self._client.close()


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


def llm_client(endpoint: str, api_key: Optional[str]) -> OpenAI:
    """vLLM requires a non-empty key even when launched without auth — use 'EMPTY'."""
    return OpenAI(base_url=endpoint, api_key=api_key or "EMPTY")


def chat(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    *,
    stream: bool = True,
    timeout_s: float = 60.0,
) -> str:
    if not stream:
        resp = client.chat.completions.create(
            model=model, messages=messages, stream=False, timeout=timeout_s
        )
        text = resp.choices[0].message.content or ""
        console.out(text)
        return text
    chunks: list[str] = []
    stream_resp = client.chat.completions.create(
        model=model, messages=messages, stream=True, timeout=timeout_s
    )
    for chunk in stream_resp:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            chunks.append(delta)
            console.out(delta, end="")
    console.out("")
    return "".join(chunks)


# ---------------------------------------------------------------------------
# Sandbox runner
# ---------------------------------------------------------------------------


def run_sandbox(
    code: str,
    *,
    network: bool = False,
    timeout_s: int = 30,
    cfg: Optional[Config] = None,
) -> subprocess.CompletedProcess:
    cfg = cfg or Config.load()
    sbx_id = uuid.uuid4().hex[:12]
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"michael-sbx-{sbx_id}-", dir="/tmp"))
    try:
        (tmp / "main.py").write_text(code)
        os.chmod(tmp, 0o755)
        os.chmod(tmp / "main.py", 0o644)

        mb = int(cfg.sandbox_memory_mb)
        cpus = float(cfg.sandbox_cpus)
        pids = int(cfg.sandbox_pids)

        mount = f"{tmp}:/workspace:rw"
        if RUNTIME == "podman":
            mount += ",Z"

        argv = [
            RUNTIME, "run", "--rm",
            "--name", f"sbx_{sbx_id}",
            "--network", "bridge" if network else "none",
            f"--memory={mb}m",
            f"--memory-swap={mb}m",
            f"--cpus={cpus}",
            f"--pids-limit={pids}",
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--tmpfs", "/home/sandbox:rw,nosuid,size=64m",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user", "1000:1000",
            "-v", mount,
            "-w", "/workspace",
            cfg.sandbox_image,
            "python3", "main.py",
        ]

        append_event(
            "sandbox.run",
            {
                "id": sbx_id,
                "runtime": RUNTIME,
                "network": network,
                "timeout_s": timeout_s,
                "image": cfg.sandbox_image,
                "argv_summary": " ".join(argv[:6]) + " ...",
            },
        )

        t0 = time.monotonic()
        try:
            cp = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout_s, check=False
            )
            duration = time.monotonic() - t0
        except subprocess.TimeoutExpired as e:
            duration = time.monotonic() - t0
            append_event(
                "sandbox.exit",
                {
                    "id": sbx_id,
                    "rc": 124,
                    "duration_s": round(duration, 3),
                    "stdout_truncated": (e.stdout or b"")[-2000:].decode(errors="replace")
                    if isinstance(e.stdout, (bytes, bytearray))
                    else (e.stdout or "")[-2000:],
                    "stderr_truncated": (e.stderr or b"")[-2000:].decode(errors="replace")
                    if isinstance(e.stderr, (bytes, bytearray))
                    else (e.stderr or "")[-2000:],
                    "timed_out": True,
                },
            )
            subprocess.run(
                [RUNTIME, "rm", "-f", f"sbx_{sbx_id}"],
                capture_output=True,
                timeout=10,
                check=False,
            )
            raise MichaelError(f"sandbox timed out after {timeout_s}s") from e

        append_event(
            "sandbox.exit",
            {
                "id": sbx_id,
                "rc": cp.returncode,
                "duration_s": round(duration, 3),
                "stdout_truncated": (cp.stdout or "")[-2000:],
                "stderr_truncated": (cp.stderr or "")[-2000:],
                "timed_out": False,
            },
        )
        return cp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Diff + Y/n/Edit confirmer
# ---------------------------------------------------------------------------


def confirm_change(old: str, new: str, *, title: str) -> Optional[str]:
    patch_id = uuid.uuid4().hex[:12]
    append_event(
        "patch.proposed",
        {"id": patch_id, "target": title, "old": old, "new": new},
    )
    while True:
        diff = list(
            __import__("difflib").unified_diff(
                old.splitlines(keepends=True),
                new.splitlines(keepends=True),
                fromfile="current",
                tofile="proposed",
            )
        )
        rendered = "".join(diff) or "(no changes)"
        console.print(
            Panel(
                Syntax(rendered, "diff", theme="ansi_dark", word_wrap=True),
                title=title,
                border_style="cyan",
            )
        )
        choice = (typer.prompt("Apply? [Y]es / [n]o / [e]dit", default="y") or "").strip().lower()
        if choice in ("", "y", "yes"):
            append_event("patch.accepted", {"id": patch_id, "target": title})
            return new
        if choice in ("n", "no"):
            append_event("patch.rejected", {"id": patch_id, "target": title})
            return None
        if choice in ("e", "edit"):
            edited = typer.edit(new)
            if edited is None:
                err.print("editor returned no content; try again")
                continue
            append_event(
                "patch.edited",
                {"id": patch_id, "target": title, "new": edited},
            )
            return edited
        err.print(f"unknown choice: {choice!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_FENCE_RE = re.compile(r"```python\s*\n(.*?)\n```", re.DOTALL)


def extract_python_block(text: str) -> Optional[str]:
    m = _FENCE_RE.search(text)
    return m.group(1) if m else None


def _require_endpoint(cfg: Config) -> str:
    state = replay()
    endpoint = cfg.endpoint or state.get("endpoint")
    if not endpoint:
        raise MichaelError("no endpoint known — run `michael up` first")
    return endpoint


def _ping_vllm(endpoint: str, api_key: Optional[str], *, timeout_s: float = 10.0) -> bool:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        with httpx.Client(timeout=timeout_s) as c:
            r = c.get(f"{endpoint}/models", headers=headers)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command()
def init() -> None:
    """Write a stub config file if missing. Idempotent."""
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    if not CONFIG_PATH.is_file():
        Config().save()
        console.print(f"[green]wrote stub[/] {CONFIG_PATH}")
    else:
        console.print(f"[dim]config exists[/] {CONFIG_PATH}")
    append_event("config.loaded", {"path": str(CONFIG_PATH)})
    console.print(
        Panel(
            "Required environment variables:\n"
            "  VAST_API_KEY        Vast.ai console API key\n"
            "  VAST_INSTANCE_ID    numeric id of the rented GPU instance\n"
            "  VLLM_API_KEY        the key passed to vLLM at launch (or empty)\n"
            "  MICHAEL_MODEL       served-model-name (default: qwen3-coder)",
            title="checklist",
            border_style="green",
        )
    )


@app.command()
def up() -> None:
    """Resume the Vast.ai instance and wait for vLLM to answer /v1/models."""
    cfg = Config.load()
    if not cfg.vast_instance_id:
        raise MichaelError("VAST_INSTANCE_ID is not set")
    vast = VastClient(cfg.vast_api_key)
    try:
        vast.start(cfg.vast_instance_id)
        append_event("instance.start_requested", {"id": cfg.vast_instance_id})
        console.print(f"[cyan]starting instance {cfg.vast_instance_id}…[/]")
        endpoint: Optional[str] = None
        for i in range(30):
            time.sleep(cfg.boot_poll_s)
            try:
                ep = vast.endpoint_for(cfg.vast_instance_id, cfg.vllm_internal_port)
            except MichaelError:
                ep = None
            append_event(
                "instance.poll",
                {"i": i + 1, "endpoint_known": bool(ep)},
            )
            if not ep:
                console.print(f"[dim]· poll {i + 1}: no endpoint yet[/]")
                continue
            if _ping_vllm(ep, cfg.vllm_api_key, timeout_s=10.0):
                endpoint = ep
                break
            console.print(f"[dim]· poll {i + 1}: endpoint {ep} not ready[/]")
        if not endpoint:
            raise MichaelError("instance did not become ready within poll budget")
        append_event("endpoint.discovered", {"endpoint": endpoint})
        append_event("instance.started", {"id": cfg.vast_instance_id, "endpoint": endpoint})
        cfg.endpoint = endpoint
        cfg.save()
        console.print(f"[green]ready[/] {endpoint}")
    finally:
        vast.close()


@app.command()
def down() -> None:
    """Pause the Vast.ai instance (preserve disk, stop GPU billing)."""
    cfg = Config.load()
    if not cfg.vast_instance_id:
        raise MichaelError("VAST_INSTANCE_ID is not set")
    vast = VastClient(cfg.vast_api_key)
    try:
        vast.stop(cfg.vast_instance_id)
        append_event("instance.stop_requested", {"id": cfg.vast_instance_id})
        append_event("instance.stopped", {"id": cfg.vast_instance_id})
        console.print(f"[yellow]stopped[/] {cfg.vast_instance_id}")
    finally:
        vast.close()


@app.command()
def status() -> None:
    """Show the current derived state (replay of the event log)."""
    state = replay()
    table = Table(title="michael status", border_style="cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("instance state", str(state["instance_state"]))
    table.add_row("endpoint", str(state["endpoint"]))
    table.add_row("last poll ts", str(state["last_poll_ts"]))
    table.add_row("patches pending", str(len(state["patches_pending"])))
    table.add_row("sandbox runs", str(state["sandbox_runs"]))
    table.add_row("errors", str(state["errors"]))
    console.print(table)


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="One-shot prompt for the LLM."),
    model: str = typer.Option("", help="Override served-model-name."),
) -> None:
    """One-shot LLM call against the running vLLM endpoint."""
    cfg = Config.load()
    endpoint = _require_endpoint(cfg)
    model_name = model or cfg.model_name
    client = llm_client(endpoint, cfg.vllm_api_key)
    append_event("prompt.sent", {"prompt": prompt, "model": model_name})
    text = chat(
        client,
        model_name,
        [{"role": "user", "content": prompt}],
        stream=True,
        timeout_s=float(cfg.request_timeout_s),
    )
    append_event("prompt.received", {"chars": len(text)})


@app.command()
def run() -> None:
    """Interactive REPL: ask -> propose patch -> Y/n/edit -> sandbox."""
    cfg = Config.load()
    endpoint = _require_endpoint(cfg)
    client = llm_client(endpoint, cfg.vllm_api_key)
    history: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are a careful coding assistant. When you propose code, return it "
                "in a single ```python fenced block. Keep changes small and reviewable."
            ),
        }
    ]
    console.print("[bold cyan]michael run[/] — empty line or 'quit' to exit")
    while True:
        try:
            user = typer.prompt(">>>", default="", show_default=False)
        except (EOFError, typer.Abort):
            break
        if not user or user.strip().lower() == "quit":
            break
        history.append({"role": "user", "content": user})
        append_event("prompt.sent", {"prompt": user, "model": cfg.model_name})
        text = chat(
            client,
            cfg.model_name,
            history,
            stream=True,
            timeout_s=float(cfg.request_timeout_s),
        )
        history.append({"role": "assistant", "content": text})
        append_event("prompt.received", {"chars": len(text)})

        block = extract_python_block(text)
        if not block:
            continue
        target_str = typer.prompt("target file", default="proposed.py")
        target = pathlib.Path(target_str).expanduser()
        old = target.read_text() if target.is_file() else ""
        accepted = confirm_change(old, block, title=str(target))
        if accepted is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(accepted)
        console.print(f"[green]wrote[/] {target}")
        if typer.confirm("Run in sandbox?", default=True):
            try:
                cp = run_sandbox(accepted, network=False, timeout_s=30, cfg=cfg)
            except MichaelError as e:
                err.print(str(e))
                continue
            stdout_tail = "\n".join((cp.stdout or "").splitlines()[-80:])
            stderr_tail = "\n".join((cp.stderr or "").splitlines()[-40:])
            console.print(
                Panel(stdout_tail or "(empty)", title=f"stdout (rc={cp.returncode})", border_style="green")
            )
            if stderr_tail:
                console.print(Panel(stderr_tail, title="stderr", border_style="red"))


@app.command(name="log")
def log_(tail: int = typer.Option(20, "--tail", "-n", help="How many events to show.")) -> None:
    """Show the last N events as a table."""
    events = list(_iter_events())
    if not events:
        console.print("[dim](no events yet)[/]")
        return
    last = events[-tail:] if tail > 0 else events
    table = Table(title=f"events (last {len(last)} of {len(events)})", border_style="cyan")
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
    console.print(table)


@app.command()
def sandbox(
    file: pathlib.Path = typer.Argument(..., exists=True, readable=True),
    net: bool = typer.Option(False, "--net", help="Allow bridge networking (default off)."),
    timeout: int = typer.Option(30, help="Wall-clock timeout in seconds."),
) -> None:
    """Run a Python file in the throwaway sandbox."""
    cfg = Config.load()
    code = file.read_text()
    cp = run_sandbox(code, network=net, timeout_s=timeout, cfg=cfg)
    stdout_tail = "\n".join((cp.stdout or "").splitlines()[-80:])
    stderr_tail = "\n".join((cp.stderr or "").splitlines()[-40:])
    console.print(
        Panel(stdout_tail or "(empty)", title=f"stdout (rc={cp.returncode})", border_style="green")
    )
    if stderr_tail:
        console.print(Panel(stderr_tail, title="stderr", border_style="red"))
    sys.exit(cp.returncode)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    try:
        app()
    except MichaelError as e:
        err.print(f"michael: {e}")
        sys.exit(2)
    except subprocess.CalledProcessError as e:
        err.print(f"command failed (exit {e.returncode})")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        err.print("interrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
