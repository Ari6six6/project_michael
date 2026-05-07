"""
michael — air-gapped, event-sourced, AI-native control loop.

Runs on Termux (Android) or any Linux host. The phone is the control plane:
event log, config, LLM client. When `vps.host` is configured, sandboxing is
delegated over SSH to a hardened Ubuntu 24.04 VPS that runs rootless Podman.
LLM inference is talked to directly over the network (Vast.ai vLLM
OpenAI-compatible endpoints).

State is never mutated: every transition is appended to a JSONL event log and
the live state is a pure fold over that log. No daemons, no databases, no
phone-home.

Subcommands:
    michael init                       write a stub config file
    michael up [--model PROFILE]       resume a Vast.ai instance, wait for vLLM
    michael down [--model PROFILE]     pause it (preserve disk, stop GPU bill)
    michael status                     current state, derived by replay
    michael ask "..." [--model P]      one-shot LLM call
    michael run [--model PROFILE]      interactive control loop
    michael log [--tail N]             show event log
    michael sandbox <file.py>          run a file in the throwaway sandbox
    michael ssh-test                   verify the VPS is reachable
"""

from __future__ import annotations

import atexit
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
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


class MichaelError(RuntimeError):
    """Domain error surfaced to the user with a clean message."""


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful coding assistant. When you propose code, return it in a "
    "single ```python fenced block. Keep changes small and reviewable. Prefer "
    "editing existing files over creating new ones. Do not add unrequested "
    "comments, error handling, or scaffolding."
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModelProfile:
    """One Vast.ai instance hosting one model behind a vLLM endpoint."""

    vast_instance_id: str = ""
    served_model_name: str = ""
    vllm_internal_port: int = 8000
    vllm_api_key: str = ""
    request_timeout_s: int = 120
    endpoint: Optional[str] = None  # cached after `up`


@dataclass
class VpsConfig:
    """Remote VPS that runs rootless Podman for sandbox execution."""

    host: str = ""
    port: int = 22
    user: str = "michael"
    ssh_key_path: str = "~/.ssh/id_ed25519"
    workspace_dir: str = "/home/michael/workspace"
    control_persist: str = "10m"


@dataclass
class SandboxConfig:
    image: str = "michael-sandbox:alpine"
    memory_mb: int = 384
    cpus: float = 1.5
    pids: int = 128
    timeout_s: int = 30


@dataclass
class Config:
    vast_api_key: str = ""
    models: dict[str, ModelProfile] = field(default_factory=dict)
    default_model: str = ""
    vps: VpsConfig = field(default_factory=VpsConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    system_prompt_file: str = ""  # if non-empty, read prompt from this path
    log_responses: bool = False
    boot_poll_s: int = 10

    @classmethod
    def load(cls) -> "Config":
        data: dict[str, Any] = {}
        if CONFIG_PATH.is_file():
            try:
                data = json.loads(CONFIG_PATH.read_text())
            except json.JSONDecodeError as e:
                raise MichaelError(f"config.json is not valid JSON: {e}") from e

        if v := os.environ.get("VAST_API_KEY"):
            data["vast_api_key"] = v
        if v := os.environ.get("MICHAEL_DEFAULT_MODEL"):
            data["default_model"] = v

        models_raw = data.pop("models", {}) or {}
        models: dict[str, ModelProfile] = {}
        valid_mp = set(ModelProfile.__dataclass_fields__)
        for name, prof in models_raw.items():
            if isinstance(prof, dict):
                models[name] = ModelProfile(**{k: v for k, v in prof.items() if k in valid_mp})

        vps_raw = data.pop("vps", None) or {}
        valid_vps = set(VpsConfig.__dataclass_fields__)
        vps = VpsConfig(**{k: v for k, v in vps_raw.items() if k in valid_vps})

        sb_raw = data.pop("sandbox", None) or {}
        valid_sb = set(SandboxConfig.__dataclass_fields__)
        sandbox = SandboxConfig(**{k: v for k, v in sb_raw.items() if k in valid_sb})

        valid = set(cls.__dataclass_fields__) - {"models", "vps", "sandbox"}
        clean = {k: v for k, v in data.items() if k in valid}
        return cls(models=models, vps=vps, sandbox=sandbox, **clean)

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        os.chmod(CONFIG_PATH, 0o600)

    def get_model(self, name: Optional[str] = None) -> tuple[str, ModelProfile]:
        if not self.models:
            raise MichaelError(
                "no model profiles configured — edit config.json and add a 'models' entry"
            )
        chosen = name or self.default_model or next(iter(self.models))
        if chosen not in self.models:
            raise MichaelError(
                f"unknown model profile: {chosen!r}. Available: {sorted(self.models)}"
            )
        return chosen, self.models[chosen]

    def resolved_system_prompt(self) -> str:
        if self.system_prompt_file:
            p = pathlib.Path(self.system_prompt_file).expanduser()
            if p.is_file():
                return p.read_text()
        return self.system_prompt

    def vps_active(self) -> bool:
        return bool(self.vps and self.vps.host)


def make_stub_config() -> Config:
    return Config(
        vast_api_key="",
        models={
            "coder": ModelProfile(
                vast_instance_id="",
                served_model_name="qwen3-coder",
            ),
            "big": ModelProfile(
                vast_instance_id="",
                served_model_name="qwen3",
            ),
        },
        default_model="coder",
        vps=VpsConfig(),
        sandbox=SandboxConfig(),
    )


# ---------------------------------------------------------------------------
# Event log — the heart of the system
# ---------------------------------------------------------------------------


def _last_seq() -> int:
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
        "models": {},  # profile_name -> {instance_state, endpoint, last_poll_ts}
        "last_prompt": None,
        "patches_pending": [],
        "sandbox_runs": 0,
        "errors": 0,
    }

    def m(name: str) -> dict[str, Any]:
        return state["models"].setdefault(
            name,
            {"instance_state": "unknown", "endpoint": None, "last_poll_ts": None},
        )

    for ev in _iter_events():
        t = ev.get("type", "")
        p = ev.get("payload", {}) or {}
        model = p.get("model", "")
        if t == "instance.start_requested" and model:
            m(model)["instance_state"] = "starting"
        elif t == "instance.started" and model:
            m(model)["instance_state"] = "running"
            if p.get("endpoint"):
                m(model)["endpoint"] = p["endpoint"]
        elif t == "instance.stop_requested" and model:
            m(model)["instance_state"] = "stopping"
        elif t == "instance.stopped" and model:
            m(model)["instance_state"] = "stopped"
        elif t == "instance.poll" and model:
            m(model)["last_poll_ts"] = ev.get("ts")
            if "actual_status" in p:
                m(model)["instance_state"] = p["actual_status"]
        elif t == "endpoint.discovered" and model:
            m(model)["endpoint"] = p.get("endpoint")
        elif t == "prompt.sent":
            state["last_prompt"] = p.get("prompt")
        elif t == "patch.proposed":
            state["patches_pending"].append({
                "id": p.get("id"),
                "target": p.get("target"),
                "old": p.get("old", ""),
                "new": p.get("new", ""),
            })
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
# SSH helpers
# ---------------------------------------------------------------------------


def _ssh_argv(vps: VpsConfig) -> list[str]:
    sock = STATE_DIR / "ssh-%C.sock"
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={sock}",
        "-o", f"ControlPersist={vps.control_persist}",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-i", os.path.expanduser(vps.ssh_key_path),
        "-p", str(vps.port),
        f"{vps.user}@{vps.host}",
    ]


def _ssh_run(
    vps: VpsConfig,
    remote_cmd: str,
    *,
    input_data: Optional[str] = None,
    timeout: int = 15,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        _ssh_argv(vps) + [remote_cmd],
        input=input_data,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _ssh_close_master(vps: VpsConfig) -> None:
    if not vps.host:
        return
    sock = STATE_DIR / "ssh-%C.sock"
    try:
        subprocess.run(
            [
                "ssh", "-O", "exit",
                "-o", f"ControlPath={sock}",
                "-p", str(vps.port),
                f"{vps.user}@{vps.host}",
            ],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass


# ---------------------------------------------------------------------------
# Workspace abstraction
# ---------------------------------------------------------------------------


def _validate_rel_path(path: str) -> None:
    if not path:
        raise MichaelError("empty path")
    if path.startswith("/"):
        raise MichaelError(f"absolute paths not allowed in workspace: {path!r}")
    parts = path.split("/")
    if ".." in parts:
        raise MichaelError(f"unsafe path: {path!r}")


class Workspace(ABC):
    @abstractmethod
    def root(self) -> str: ...
    @abstractmethod
    def tree(self) -> list[str]: ...
    @abstractmethod
    def read(self, path: str) -> str: ...
    @abstractmethod
    def write(self, path: str, contents: str) -> None: ...
    @abstractmethod
    def exists(self, path: str) -> bool: ...


class LocalWorkspace(Workspace):
    SKIP_PARTS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache"}

    def __init__(self, root_dir: pathlib.Path) -> None:
        self._root = root_dir.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def root(self) -> str:
        return str(self._root)

    def _full(self, path: str) -> pathlib.Path:
        _validate_rel_path(path)
        return self._root / path

    def tree(self) -> list[str]:
        out: list[str] = []
        for p in sorted(self._root.rglob("*")):
            if not p.is_file():
                continue
            if any(part in self.SKIP_PARTS for part in p.parts):
                continue
            try:
                if p.stat().st_size >= 200_000:
                    continue
            except OSError:
                continue
            out.append(str(p.relative_to(self._root)))
            if len(out) >= 2000:
                break
        return out

    def read(self, path: str) -> str:
        return self._full(path).read_text()

    def write(self, path: str, contents: str) -> None:
        target = self._full(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)

    def exists(self, path: str) -> bool:
        return self._full(path).is_file()


class RemoteSshWorkspace(Workspace):
    def __init__(self, vps: VpsConfig) -> None:
        if not vps.host:
            raise MichaelError("vps.host is empty")
        self.vps = vps

    def root(self) -> str:
        return f"{self.vps.user}@{self.vps.host}:{self.vps.workspace_dir}"

    def _full(self, path: str) -> str:
        _validate_rel_path(path)
        return f"{self.vps.workspace_dir.rstrip('/')}/{path}"

    def tree(self) -> list[str]:
        wd = shlex.quote(self.vps.workspace_dir)
        cmd = (
            f"find {wd} -type f "
            f"-not -path '*/.git/*' -not -path '*/__pycache__/*' "
            f"-not -path '*/.venv/*' -not -path '*/node_modules/*' "
            f"-size -200k -printf '%P\\n' 2>/dev/null | head -2000"
        )
        cp = _ssh_run(self.vps, cmd, timeout=30)
        if cp.returncode != 0:
            raise MichaelError(f"workspace.tree failed: {cp.stderr[:200]}")
        return [line for line in cp.stdout.splitlines() if line]

    def read(self, path: str) -> str:
        full = shlex.quote(self._full(path))
        cp = _ssh_run(self.vps, f"cat {full}", timeout=30)
        if cp.returncode != 0:
            raise MichaelError(f"workspace.read failed: {cp.stderr[:200]}")
        return cp.stdout

    def write(self, path: str, contents: str) -> None:
        full = self._full(path)
        full_q = shlex.quote(full)
        full_dir_q = shlex.quote(str(pathlib.PurePosixPath(full).parent))
        cmd = (
            f"umask 077 && mkdir -p {full_dir_q} && "
            f"cat > {full_q}.tmp && mv {full_q}.tmp {full_q}"
        )
        cp = _ssh_run(self.vps, cmd, input_data=contents, timeout=30)
        if cp.returncode != 0:
            raise MichaelError(f"workspace.write failed: {cp.stderr[:200]}")

    def exists(self, path: str) -> bool:
        full = shlex.quote(self._full(path))
        cp = _ssh_run(self.vps, f"test -f {full}", timeout=10)
        return cp.returncode == 0


def make_workspace(cfg: Config) -> Workspace:
    if cfg.vps_active():
        return RemoteSshWorkspace(cfg.vps)
    return LocalWorkspace(pathlib.Path.cwd())


# ---------------------------------------------------------------------------
# Sandbox backends
# ---------------------------------------------------------------------------


def _safe_tail(data: Any, n: int) -> str:
    if data is None:
        return ""
    if isinstance(data, (bytes, bytearray)):
        return data[-n:].decode(errors="replace")
    return str(data)[-n:]


class SandboxBackend(ABC):
    @abstractmethod
    def run(
        self, code: str, *, network: bool = False, timeout_s: int = 30
    ) -> subprocess.CompletedProcess: ...


class DisabledSandboxBackend(SandboxBackend):
    def run(self, code, *, network=False, timeout_s=30):
        raise MichaelError(
            "sandbox unavailable: configure vps.host (recommended on phone) "
            "or install podman/docker locally"
        )


class LocalPodmanBackend(SandboxBackend):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.runtime = "podman" if shutil.which("podman") else "docker"

    def run(
        self, code: str, *, network: bool = False, timeout_s: int = 30
    ) -> subprocess.CompletedProcess:
        sbx = self.cfg.sandbox
        sbx_id = uuid.uuid4().hex[:12]
        tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"michael-sbx-{sbx_id}-", dir="/tmp"))
        try:
            (tmp / "main.py").write_text(code)
            os.chmod(tmp, 0o755)
            os.chmod(tmp / "main.py", 0o644)

            mount = f"{tmp}:/workspace:ro"
            if self.runtime == "podman":
                mount += ",Z"

            argv = [
                self.runtime, "run", "--rm",
                "--name", f"sbx_{sbx_id}",
                "--network", "bridge" if network else "none",
                f"--memory={sbx.memory_mb}m",
                f"--memory-swap={sbx.memory_mb}m",
                f"--cpus={sbx.cpus}",
                f"--pids-limit={sbx.pids}",
                "--read-only",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                "--tmpfs", "/home/sandbox:rw,nosuid,size=64m",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--user", "1000:1000",
                "-v", mount, "-w", "/workspace",
                sbx.image, "python3", "main.py",
            ]

            append_event(
                "sandbox.run",
                {
                    "id": sbx_id,
                    "host": "local",
                    "runtime": self.runtime,
                    "network": network,
                    "timeout_s": timeout_s,
                    "image": sbx.image,
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
                self._cleanup(sbx_id)
                append_event(
                    "sandbox.exit",
                    {
                        "id": sbx_id,
                        "host": "local",
                        "rc": 124,
                        "duration_s": round(duration, 3),
                        "stdout_truncated": _safe_tail(e.stdout, 2000),
                        "stderr_truncated": _safe_tail(e.stderr, 2000),
                        "timed_out": True,
                    },
                )
                raise MichaelError(f"sandbox timed out after {timeout_s}s") from e

            append_event(
                "sandbox.exit",
                {
                    "id": sbx_id,
                    "host": "local",
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

    def _cleanup(self, sbx_id: str) -> None:
        subprocess.run(
            [self.runtime, "rm", "-f", f"sbx_{sbx_id}"],
            capture_output=True,
            timeout=10,
            check=False,
        )


class RemoteSshPodmanBackend(SandboxBackend):
    def __init__(self, cfg: Config) -> None:
        if not cfg.vps_active():
            raise MichaelError("vps.host required for remote sandbox")
        self.cfg = cfg
        self.vps = cfg.vps

    def run(
        self, code: str, *, network: bool = False, timeout_s: int = 30
    ) -> subprocess.CompletedProcess:
        sbx = self.cfg.sandbox
        sbx_id = uuid.uuid4().hex[:12]
        sandbox_dir = f"/tmp/michael-sbx-{sbx_id}"
        sandbox_dir_q = shlex.quote(sandbox_dir)

        stage = (
            f"set -e; mkdir -p {sandbox_dir_q}; chmod 700 {sandbox_dir_q}; "
            f"cat > {sandbox_dir_q}/main.py; chmod 644 {sandbox_dir_q}/main.py"
        )
        cp_stage = _ssh_run(self.vps, stage, input_data=code, timeout=30)
        if cp_stage.returncode != 0:
            raise MichaelError(f"remote stage failed: {cp_stage.stderr[:200]}")

        runtime = "podman"
        mount = f"{sandbox_dir}:/workspace:ro,Z"
        podman_argv = [
            runtime, "run", "--rm",
            "--name", f"sbx_{sbx_id}",
            "--network", "bridge" if network else "none",
            f"--memory={sbx.memory_mb}m",
            f"--memory-swap={sbx.memory_mb}m",
            f"--cpus={sbx.cpus}",
            f"--pids-limit={sbx.pids}",
            "--read-only",
            "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
            "--tmpfs", "/home/sandbox:rw,nosuid,size=64m",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--user", "1000:1000",
            "-v", mount, "-w", "/workspace",
            sbx.image, "python3", "main.py",
        ]
        podman_str = " ".join(shlex.quote(a) for a in podman_argv)
        lock_path = shlex.quote(f"/home/{self.vps.user}/.michael/sandbox.lock")
        run_cmd = (
            f"mkdir -p $(dirname {lock_path}) && "
            f"flock -w 60 {lock_path} -c {shlex.quote(podman_str)}"
        )

        append_event(
            "sandbox.run",
            {
                "id": sbx_id,
                "host": "vps",
                "runtime": runtime,
                "network": network,
                "timeout_s": timeout_s,
                "image": sbx.image,
                "argv_summary": " ".join(podman_argv[:6]) + " ...",
            },
        )

        t0 = time.monotonic()
        timed_out = False
        try:
            cp = _ssh_run(self.vps, run_cmd, timeout=timeout_s + 15)
            duration = time.monotonic() - t0
        except subprocess.TimeoutExpired as e:
            duration = time.monotonic() - t0
            timed_out = True
            self._cleanup(sbx_id, sandbox_dir)
            append_event(
                "sandbox.exit",
                {
                    "id": sbx_id,
                    "host": "vps",
                    "rc": 124,
                    "duration_s": round(duration, 3),
                    "stdout_truncated": _safe_tail(e.stdout, 2000),
                    "stderr_truncated": _safe_tail(e.stderr, 2000),
                    "timed_out": True,
                },
            )
            raise MichaelError(f"sandbox timed out after {timeout_s}s") from e
        finally:
            if not timed_out:
                self._cleanup(sbx_id, sandbox_dir)

        append_event(
            "sandbox.exit",
            {
                "id": sbx_id,
                "host": "vps",
                "rc": cp.returncode,
                "duration_s": round(duration, 3),
                "stdout_truncated": (cp.stdout or "")[-2000:],
                "stderr_truncated": (cp.stderr or "")[-2000:],
                "timed_out": False,
            },
        )
        return cp

    def _cleanup(self, sbx_id: str, sandbox_dir: str) -> None:
        cmd = (
            f"podman rm -f sbx_{sbx_id} >/dev/null 2>&1 || true; "
            f"rm -rf {shlex.quote(sandbox_dir)}"
        )
        try:
            _ssh_run(self.vps, cmd, timeout=15)
        except subprocess.TimeoutExpired:
            pass


def make_backend(cfg: Config) -> SandboxBackend:
    if cfg.vps_active():
        return RemoteSshPodmanBackend(cfg)
    if shutil.which("podman") or shutil.which("docker"):
        return LocalPodmanBackend(cfg)
    return DisabledSandboxBackend()


# ---------------------------------------------------------------------------
# Vast.ai client
# ---------------------------------------------------------------------------

# On Vast.ai, rent a GPU and launch the vLLM template:
#   Image:  vllm/vllm-openai:v0.20.0
#   Env:    -p 8000:8000  -e HF_TOKEN=...  -e VLLM_API_KEY=...
#   Args:
#     --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
#     --served-model-name qwen3-coder
#     --max-model-len 131072 --gpu-memory-utilization 0.92
#     --enable-auto-tool-choice --tool-call-parser qwen3_coder
#     --api-key ${VLLM_API_KEY}


class VastClient:
    def __init__(self, api_key: str, base: str = "https://console.vast.ai/api/v0") -> None:
        if not api_key:
            raise MichaelError("vast_api_key is not set")
        self.base = base.rstrip("/")
        self._client = httpx.Client(
            timeout=httpx.Timeout(10.0, read=60.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json",
            },
        )

    def _wrap(self, fn_name: str, request) -> Any:
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
) -> tuple[str, dict[str, Any]]:
    """Returns (full_text, usage_dict). usage_dict is {} if unsupported."""
    usage: dict[str, Any] = {}
    if not stream:
        resp = client.chat.completions.create(
            model=model, messages=messages, stream=False, timeout=timeout_s
        )
        text = resp.choices[0].message.content or ""
        if resp.usage is not None:
            usage = _usage_dict(resp.usage)
        console.out(text)
        return text, usage

    chunks: list[str] = []
    stream_resp = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        timeout=timeout_s,
        stream_options={"include_usage": True},
    )
    for chunk in stream_resp:
        if chunk.choices:
            delta = chunk.choices[0].delta.content if chunk.choices[0].delta else None
            if delta:
                chunks.append(delta)
                console.out(delta, end="")
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = _usage_dict(chunk_usage)
    console.out("")
    return "".join(chunks), usage


def _usage_dict(u: Any) -> dict[str, Any]:
    try:
        return u.model_dump()
    except AttributeError:
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", 0),
        }


# ---------------------------------------------------------------------------
# Diff confirmer
# ---------------------------------------------------------------------------


def confirm_change(old: str, new: str, *, title: str) -> Optional[str]:
    patch_id = uuid.uuid4().hex[:12]
    append_event(
        "patch.proposed",
        {"id": patch_id, "target": title, "old": old, "new": new, "host": "phone"},
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
            append_event("patch.accepted", {"id": patch_id, "target": title, "host": "phone"})
            return new
        if choice in ("n", "no"):
            append_event("patch.rejected", {"id": patch_id, "target": title, "host": "phone"})
            return None
        if choice in ("e", "edit"):
            edited = typer.edit(new)
            if edited is None:
                err.print("editor returned no content; try again")
                continue
            append_event(
                "patch.edited",
                {"id": patch_id, "target": title, "new": edited, "host": "phone"},
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


def _require_endpoint(profile: ModelProfile, profile_name: str) -> str:
    endpoint = profile.endpoint
    if not endpoint:
        st = replay().get("models", {}).get(profile_name, {})
        endpoint = st.get("endpoint")
    if not endpoint:
        raise MichaelError(
            f"no endpoint known for {profile_name!r} — run "
            f"`michael up --model {profile_name}` first"
        )
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


def _ssh_preflight(cfg: Config) -> None:
    if not cfg.vps_active():
        return
    cp = subprocess.run(
        _ssh_argv(cfg.vps) + ["true"],
        capture_output=True, text=True, timeout=15, check=False,
    )
    if cp.returncode != 0:
        append_event("ssh.health", {"host": cfg.vps.host, "ok": False, "stderr": cp.stderr[:200]})
        raise MichaelError(
            f"VPS unreachable ({cfg.vps.user}@{cfg.vps.host}): {cp.stderr.strip()[:200]}"
        )
    append_event("ssh.health", {"host": cfg.vps.host, "ok": True})
    atexit.register(_ssh_close_master, cfg.vps)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@app.command()
def init() -> None:
    """Write a stub config file if missing. Idempotent."""
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    if not CONFIG_PATH.is_file():
        make_stub_config().save()
        console.print(f"[green]wrote stub[/] {CONFIG_PATH}")
    else:
        console.print(f"[dim]config exists[/] {CONFIG_PATH}")
    append_event("config.loaded", {"path": str(CONFIG_PATH)})
    console.print(
        Panel(
            "Edit ~/.michael/config.json — fill in:\n"
            "\n"
            "  [bold]vast_api_key[/]              your Vast.ai console API key\n"
            "  [bold]default_model[/]             which profile to use by default\n"
            "  [bold]models.<name>[/]             one entry per Vast.ai instance:\n"
            "    vast_instance_id              numeric instance id\n"
            "    served_model_name             matches --served-model-name on vLLM\n"
            "    vllm_api_key                  the key vLLM was launched with (or empty)\n"
            "\n"
            "[dim]Optional, for remote sandbox on the VPS:[/]\n"
            "  [bold]vps.host[/]                  VPS public IP/hostname\n"
            "  [bold]vps.user[/]                  ssh user (default: michael)\n"
            "  [bold]vps.ssh_key_path[/]          path to private key\n"
            "  [bold]vps.workspace_dir[/]         /home/michael/workspace\n"
            "\n"
            "[dim]Leave vps.host empty to run chat-only (no sandbox).[/]",
            title="checklist",
            border_style="green",
        )
    )


@app.command()
def up(model: str = typer.Option("", "--model", "-m", help="Model profile name.")) -> None:
    """Resume a Vast.ai instance and wait for vLLM to answer /v1/models."""
    cfg = Config.load()
    name, profile = cfg.get_model(model or None)
    if not profile.vast_instance_id:
        raise MichaelError(f"models.{name}.vast_instance_id is not set")
    vast = VastClient(cfg.vast_api_key)
    try:
        vast.start(profile.vast_instance_id)
        append_event(
            "instance.start_requested",
            {"id": profile.vast_instance_id, "model": name},
        )
        console.print(f"[cyan]starting {name} (instance {profile.vast_instance_id})…[/]")
        endpoint: Optional[str] = None
        for i in range(30):
            time.sleep(cfg.boot_poll_s)
            try:
                ep = vast.endpoint_for(profile.vast_instance_id, profile.vllm_internal_port)
            except MichaelError:
                ep = None
            append_event(
                "instance.poll",
                {"i": i + 1, "model": name, "endpoint_known": bool(ep)},
            )
            if not ep:
                console.print(f"[dim]· poll {i + 1}: no endpoint yet[/]")
                continue
            if _ping_vllm(ep, profile.vllm_api_key, timeout_s=10.0):
                endpoint = ep
                break
            console.print(f"[dim]· poll {i + 1}: endpoint {ep} not ready[/]")
        if not endpoint:
            raise MichaelError("instance did not become ready within poll budget")
        append_event("endpoint.discovered", {"endpoint": endpoint, "model": name})
        append_event(
            "instance.started",
            {"id": profile.vast_instance_id, "model": name, "endpoint": endpoint},
        )
        profile.endpoint = endpoint
        cfg.models[name] = profile
        cfg.save()
        console.print(f"[green]ready[/] {name} @ {endpoint}")
    finally:
        vast.close()


@app.command()
def down(model: str = typer.Option("", "--model", "-m", help="Model profile name.")) -> None:
    """Pause a Vast.ai instance (preserve disk, stop GPU billing)."""
    cfg = Config.load()
    name, profile = cfg.get_model(model or None)
    if not profile.vast_instance_id:
        raise MichaelError(f"models.{name}.vast_instance_id is not set")
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
        console.print(f"[yellow]stopped[/] {name} ({profile.vast_instance_id})")
    finally:
        vast.close()


@app.command()
def status() -> None:
    """Show derived state from the event log."""
    cfg = Config.load()
    state = replay()
    table = Table(title="michael status", border_style="cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    if cfg.vps_active():
        table.add_row("vps", f"{cfg.vps.user}@{cfg.vps.host}:{cfg.vps.port}")
        table.add_row("vps.workspace", cfg.vps.workspace_dir)
    else:
        table.add_row("vps", "[dim]not configured (chat-only)[/]")

    table.add_row("default model", cfg.default_model or "[dim](first available)[/]")
    if not cfg.models:
        table.add_row("models", "[dim](none — edit config.json)[/]")
    for name, profile in cfg.models.items():
        st = state.get("models", {}).get(name, {})
        table.add_row(
            f"  {name}",
            f"state={st.get('instance_state', 'unknown')}  "
            f"endpoint={st.get('endpoint') or profile.endpoint or '—'}",
        )

    table.add_row("patches pending", str(len(state["patches_pending"])))
    table.add_row("sandbox runs", str(state["sandbox_runs"]))
    table.add_row("errors", str(state["errors"]))
    console.print(table)


@app.command()
def ask(
    prompt: str = typer.Argument(..., help="One-shot prompt for the LLM."),
    model: str = typer.Option("", "--model", "-m", help="Model profile name."),
) -> None:
    """One-shot LLM call against a running vLLM endpoint."""
    cfg = Config.load()
    name, profile = cfg.get_model(model or None)
    endpoint = _require_endpoint(profile, name)
    client = llm_client(endpoint, profile.vllm_api_key)
    append_event(
        "prompt.sent",
        {"prompt": prompt, "model": name, "served": profile.served_model_name, "host": "phone"},
    )
    text, usage = chat(
        client,
        profile.served_model_name,
        [{"role": "user", "content": prompt}],
        stream=True,
        timeout_s=float(profile.request_timeout_s),
    )
    payload: dict[str, Any] = {
        "chars": len(text),
        "model": name,
        "served": profile.served_model_name,
        "usage": usage,
        "host": "phone",
    }
    if cfg.log_responses:
        payload["text"] = text
    append_event("prompt.received", payload)


@app.command()
def run(
    model: str = typer.Option("", "--model", "-m", help="Model profile name."),
    auto_target: bool = typer.Option(
        False, "--auto-target", help="Skip target-file prompt; use proposed.py"
    ),
) -> None:
    """Interactive REPL: chat → propose patch → Y/n/edit → sandbox."""
    cfg = Config.load()
    name, profile = cfg.get_model(model or None)
    endpoint = _require_endpoint(profile, name)
    _ssh_preflight(cfg)

    client = llm_client(endpoint, profile.vllm_api_key)
    workspace = make_workspace(cfg)
    backend = make_backend(cfg)

    history: list[dict[str, str]] = [
        {"role": "system", "content": cfg.resolved_system_prompt()},
    ]

    runtime_label = (
        f"vps={cfg.vps.host}" if cfg.vps_active() else f"local-fs={workspace.root()}"
    )
    backend_label = (
        "remote-podman" if cfg.vps_active()
        else ("local-podman/docker" if not isinstance(backend, DisabledSandboxBackend)
              else "no-sandbox")
    )
    console.print(
        f"[bold cyan]michael run[/] [dim]model={name}  workspace={runtime_label}  "
        f"sandbox={backend_label}[/]"
    )
    console.print("[dim]empty line or 'quit' to exit[/]")

    while True:
        try:
            user = typer.prompt(">>>", default="", show_default=False)
        except (EOFError, typer.Abort):
            break
        if not user or user.strip().lower() == "quit":
            break

        history.append({"role": "user", "content": user})
        append_event(
            "prompt.sent",
            {"prompt": user, "model": name, "served": profile.served_model_name, "host": "phone"},
        )
        text, usage = chat(
            client,
            profile.served_model_name,
            history,
            stream=True,
            timeout_s=float(profile.request_timeout_s),
        )
        history.append({"role": "assistant", "content": text})
        recv: dict[str, Any] = {
            "chars": len(text),
            "model": name,
            "served": profile.served_model_name,
            "usage": usage,
            "host": "phone",
        }
        if cfg.log_responses:
            recv["text"] = text
        append_event("prompt.received", recv)

        block = extract_python_block(text)
        if not block:
            continue

        target_str = "proposed.py" if auto_target else typer.prompt(
            "target file (relative to workspace)", default="proposed.py"
        )
        try:
            old = workspace.read(target_str) if workspace.exists(target_str) else ""
        except MichaelError as e:
            err.print(str(e))
            continue
        accepted = confirm_change(old, block, title=target_str)
        if accepted is None:
            continue
        try:
            workspace.write(target_str, accepted)
        except MichaelError as e:
            err.print(str(e))
            continue
        console.print(f"[green]wrote[/] {workspace.root()}/{target_str}")

        if typer.confirm("Run in sandbox?", default=True):
            try:
                cp = backend.run(accepted, network=False, timeout_s=cfg.sandbox.timeout_s)
            except MichaelError as e:
                err.print(str(e))
                continue
            stdout_tail = "\n".join((cp.stdout or "").splitlines()[-80:])
            stderr_tail = "\n".join((cp.stderr or "").splitlines()[-40:])
            console.print(
                Panel(
                    stdout_tail or "(empty)",
                    title=f"stdout (rc={cp.returncode})",
                    border_style="green" if cp.returncode == 0 else "red",
                )
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
    _ssh_preflight(cfg)
    backend = make_backend(cfg)
    code = file.read_text()
    cp = backend.run(code, network=net, timeout_s=timeout)
    stdout_tail = "\n".join((cp.stdout or "").splitlines()[-80:])
    stderr_tail = "\n".join((cp.stderr or "").splitlines()[-40:])
    console.print(
        Panel(
            stdout_tail or "(empty)",
            title=f"stdout (rc={cp.returncode})",
            border_style="green" if cp.returncode == 0 else "red",
        )
    )
    if stderr_tail:
        console.print(Panel(stderr_tail, title="stderr", border_style="red"))
    sys.exit(cp.returncode)


@app.command()
def config(
    edit: bool = typer.Option(False, "--edit", "-e", help="Open in $EDITOR (default if no flag)."),
    show: bool = typer.Option(False, "--show", "-s", help="Print to stdout."),
) -> None:
    """Show or edit ~/.michael/config.json."""
    if not CONFIG_PATH.is_file():
        raise MichaelError("no config — run `michael init` first")
    if show:
        console.print(Panel(CONFIG_PATH.read_text(), title=str(CONFIG_PATH), border_style="cyan"))
        return
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
    subprocess.run([editor, str(CONFIG_PATH)], check=False)


@app.command(name="ssh-test")
def ssh_test() -> None:
    """Verify the VPS is reachable and report the ssh handshake time."""
    cfg = Config.load()
    if not cfg.vps_active():
        raise MichaelError("vps.host is not configured")
    t0 = time.monotonic()
    cp = subprocess.run(
        _ssh_argv(cfg.vps) + ["echo ok && podman --version 2>/dev/null || true"],
        capture_output=True, text=True, timeout=15, check=False,
    )
    dt = round(time.monotonic() - t0, 3)
    if cp.returncode != 0:
        append_event("ssh.health", {"host": cfg.vps.host, "ok": False, "stderr": cp.stderr[:200]})
        raise MichaelError(f"ssh failed in {dt}s: {cp.stderr.strip()[:200]}")
    append_event("ssh.health", {"host": cfg.vps.host, "ok": True, "duration_s": dt})
    console.print(
        Panel(
            cp.stdout.strip() or "(no output)",
            title=f"ssh ok in {dt}s — {cfg.vps.user}@{cfg.vps.host}",
            border_style="green",
        )
    )


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
