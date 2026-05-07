"""
michael — air-gapped, event-sourced, AI-native control loop.

Phone-first: runs on Termux (Android) or any Linux host. The phone is the
control plane — config, event log, LLM client, user confirmation. When
`vps.host` is configured, the podman sandbox is delegated over SSH to a
hardened Ubuntu 24.04 VPS (rootless podman). LLM inference runs on Vast.ai
via vLLM OpenAI-compatible endpoints.

Architecture:
- Bare `michael` enters a REPL. `michael <subcmd>` runs one-shot from the
  user's shell. Inside the REPL, commands have no `michael` prefix.
- State at ~/.michael/. Global: instance lifecycle, vast/vllm/vps keys.
  Per-project at ~/.michael/projects/<slug>/: prompts, actions, events.
- Every user prompt rebuilds a fresh "header package" (system message) from
  the project log and a live filesystem snapshot. The LLM is amnesiac across
  user prompts; the project log is its memory. Within one user prompt, the
  agent loop iterates with the OpenAI tool-call protocol so the model sees
  its own tool results — those messages are discarded between user prompts.
- Writes go through a /tmp staging copy. An optional `verify` command runs
  in staging; failures are reported back to the LLM (not the user). On user
  Yes, the pre-change content is snapshotted to per-project trash and the
  staged files are synced into the real workspace. `undo` restores.

REPL commands (also work as `michael <cmd>` one-shot):
    show                          list projects
    new project [name]            create a new project
    new code [--model P]          fresh agent loop, code mode (full toolset)
    new discussion [--model P]    fresh agent loop, read-only chat
    nitro [--model P]             fresh agent loop on the heavy model
    use <slug>                    switch active project
    current                       print the active project
    config                        open ~/.michael/config.json in $EDITOR
    init                          write a stub config if missing
    up [--model PROFILE]          start a Vast.ai instance
    down [--model PROFILE]        stop a Vast.ai instance
    status                        derived state from the event log
    ask "<prompt>" [--model P]    one-shot LLM call
    run [--model PROFILE]         multi-turn tool-calling agent loop (alias of `new code`)
    log [--tail N]                show the event log
    sandbox <file.py>             run a file in the sandbox
    undo [--list] [<id>]          restore the most recent (or named) change
    ssh-test                      verify the VPS is reachable
    quit | exit                   leave the REPL
"""

from __future__ import annotations

import atexit
import difflib
import fcntl
import hashlib
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
from typing import Any, Optional

import httpx
import typer
from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

STATE_DIR = pathlib.Path.home() / ".michael"
GLOBAL_CONFIG_PATH = STATE_DIR / "config.json"
GLOBAL_EVENTS_PATH = STATE_DIR / "events.jsonl"
STATE_FILE_PATH = STATE_DIR / "state.json"
PROJECTS_DIR = STATE_DIR / "projects"
REPL_HISTORY_PATH = STATE_DIR / "repl_history"

console = Console()
err = Console(stderr=True, style="bold red")

app = typer.Typer(
    no_args_is_help=False,
    rich_markup_mode="rich",
    help="michael — air-gapped AI control loop",
)

MAX_FILE_BYTES_INLINE = 50_000
MAX_TOTAL_BYTES_INLINE = 500_000
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".idea", ".vscode", ".next",
    "target", ".cache",
}

AUTO_EXEC_TOOLS = {"read_file", "list_dir"}


class MichaelError(RuntimeError):
    """Domain error surfaced to the user with a clean message."""


# The literal passcode the LLM emits to surface its product to the user.
# Hardcoded by design: one symbol, one source of truth, used by both the
# protocol text the LLM reads and the parser that gates user-presentation.
JA_PASSPHRASE = "Ja"


def _message_ends_with_ja(text: str) -> bool:
    """True iff the message's trailing token (after stripping whitespace and
    common terminal punctuation) is the bareword JA_PASSPHRASE — case-sensitive.

    Catches: 'thoughts.\\nJa', 'thoughts.\\nJa\\n', 'thoughts. Ja', 'work. Ja.'
    Rejects: '', 'Ja, das ist gut' (mid-sentence), 'Yes',
             'Ja im Anfang' (Ja not at the end).
    """
    if not text:
        return False
    stripped = text.rstrip().rstrip(".!?;:")
    if not stripped:
        return False
    last_token = stripped.rsplit(None, 1)[-1]
    return last_token == JA_PASSPHRASE


DEFAULT_SYSTEM_PROMPT = (
    "You are a careful coding assistant connected to the user's machine "
    "through Project Michael. Keep changes small and reviewable. Prefer "
    "editing existing files over creating new ones. Do not add unrequested "
    "comments, error handling, or scaffolding."
)


# ---------------------------------------------------------------------------
# Config (multi-model + VPS + sandbox)
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
    system_prompt_file: str = ""
    log_responses: bool = False
    boot_poll_s: int = 10

    @classmethod
    def load(cls) -> "Config":
        data: dict[str, Any] = {}
        if GLOBAL_CONFIG_PATH.is_file():
            try:
                data = json.loads(GLOBAL_CONFIG_PATH.read_text())
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
        STATE_DIR.mkdir(mode=0o700, exist_ok=True)
        GLOBAL_CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        os.chmod(GLOBAL_CONFIG_PATH, 0o600)

    def get_model(self, name: Optional[str] = None) -> tuple[str, ModelProfile]:
        if not self.models:
            raise MichaelError(
                "no model profiles configured — run `config` and add a 'models' entry"
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


CONFIG_HELP: dict[str, str] = {
    "vast_api_key": "Vast.ai console API key.",
    "default_model": "Profile name used when no --model flag is passed.",
    "models.<name>.vast_instance_id": "Numeric ID of the rented GPU instance.",
    "models.<name>.served_model_name": "Matches --served-model-name on vLLM.",
    "models.<name>.vllm_api_key": "Key vLLM was launched with (or empty).",
    "models.<name>.vllm_internal_port": "Container-internal port (default 8000).",
    "models.<name>.request_timeout_s": "LLM request timeout (seconds).",
    "vps.host": "VPS public IP/hostname (empty = no remote sandbox).",
    "vps.user": "SSH user (default: michael).",
    "vps.ssh_key_path": "Path to private key (default: ~/.ssh/id_ed25519).",
    "vps.workspace_dir": "Default workspace dir on the VPS.",
    "sandbox.image": "Tag of the sandbox image built by bootstrap.sh.",
    "sandbox.memory_mb": "Sandbox memory cap in MB.",
    "sandbox.cpus": "Sandbox CPU cap.",
    "sandbox.pids": "Sandbox PID cap.",
    "sandbox.timeout_s": "Default sandbox timeout (seconds).",
    "system_prompt": "Default system prompt for chat/agent loops.",
    "system_prompt_file": "If set, read system prompt from this file.",
    "log_responses": "If true, log full LLM responses to events.jsonl.",
    "boot_poll_s": "Poll interval while waiting for vLLM to come up.",
}


# ---------------------------------------------------------------------------
# Project model
# ---------------------------------------------------------------------------


@dataclass
class Project:
    slug: str
    name: str
    path: str
    created_at: str

    @classmethod
    def load(cls, slug: str) -> "Project":
        cfg = PROJECTS_DIR / slug / "config.json"
        if not cfg.is_file():
            raise MichaelError(f"unknown project: {slug}")
        data = json.loads(cfg.read_text())
        try:
            return cls(**{k: data[k] for k in cls.__dataclass_fields__})
        except KeyError as e:
            raise MichaelError(f"project {slug} config missing key: {e}") from e

    def save(self) -> None:
        d = PROJECTS_DIR / self.slug
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
        (d / "config.json").write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True)
        )

    @property
    def events_path(self) -> pathlib.Path:
        return PROJECTS_DIR / self.slug / "events.jsonl"


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    s = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return s[:64] or "project"


def list_projects() -> list[Project]:
    if not PROJECTS_DIR.is_dir():
        return []
    out: list[Project] = []
    for d in sorted(PROJECTS_DIR.iterdir()):
        cfg = d / "config.json"
        if not cfg.is_file():
            continue
        try:
            data = json.loads(cfg.read_text())
            out.append(Project(**{k: data[k] for k in Project.__dataclass_fields__}))
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return out


def create_project(name: str, path: pathlib.Path) -> Project:
    base = slugify(name)
    slug = base
    n = 2
    existing = {p.slug for p in list_projects()}
    while slug in existing:
        slug = f"{base}-{n}"
        n += 1
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    proj = Project(
        slug=slug,
        name=name,
        path=str(path),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    proj.save()
    append_event("project.created", {"slug": slug, "name": name, "path": str(path)})
    return proj


# ---------------------------------------------------------------------------
# Active-project state
# ---------------------------------------------------------------------------


def get_active_slug() -> Optional[str]:
    if not STATE_FILE_PATH.is_file():
        return None
    try:
        return json.loads(STATE_FILE_PATH.read_text()).get("active_project")
    except json.JSONDecodeError:
        return None


def set_active_slug(slug: Optional[str]) -> None:
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    STATE_FILE_PATH.write_text(json.dumps({"active_project": slug}, indent=2))
    os.chmod(STATE_FILE_PATH, 0o600)


def get_active_project() -> Optional[Project]:
    s = get_active_slug()
    if not s:
        return None
    try:
        return Project.load(s)
    except MichaelError:
        return None


def require_active_project() -> Project:
    p = get_active_project()
    if not p:
        raise MichaelError("no active project — run `new project` or `use <slug>`")
    return p


# ---------------------------------------------------------------------------
# Event log (global + per-project) — append under flock so two michael
# processes can't tear the seq numbering when they race.
# ---------------------------------------------------------------------------


def _last_seq(path: pathlib.Path) -> int:
    if not path.is_file():
        return 0
    last: Optional[str] = None
    with path.open("rb") as f:
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


def _append(path: pathlib.Path, type_: str, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if not path.exists():
        path.touch(mode=0o600)
    with path.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            seq = _last_seq(path) + 1
            event = {
                "seq": seq,
                "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "type": type_,
                "payload": payload,
            }
            line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return event


def append_event(
    type_: str,
    payload: dict[str, Any],
    *,
    project: Optional[Project] = None,
) -> dict[str, Any]:
    if project is not None:
        path = project.events_path
        scope = f"project:{project.slug}"
    else:
        path = GLOBAL_EVENTS_PATH
        scope = "global"
    ev = _append(path, type_, payload)
    style = "red" if type_ == "error" else "dim"
    console.print(f"[{style}]· {ev['seq']:>4} {type_} ({scope})[/]", highlight=False)
    return ev


def iter_events(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def replay_global() -> dict[str, Any]:
    """Pure fold over the global event log → per-profile state."""
    state: dict[str, Any] = {
        "models": {},  # profile_name -> {instance_state, endpoint, last_poll_ts}
        "errors": 0,
    }

    def m(name: str) -> dict[str, Any]:
        return state["models"].setdefault(
            name,
            {"instance_state": "unknown", "endpoint": None, "last_poll_ts": None},
        )

    for ev in iter_events(GLOBAL_EVENTS_PATH):
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
        elif t == "error":
            state["errors"] += 1
    return state


# ---------------------------------------------------------------------------
# SSH helpers (ControlMaster multiplexing for cheap repeated calls)
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
# Vast.ai client (plain httpx — no `vastai` library)
# ---------------------------------------------------------------------------

# On Vast.ai, rent 1× H100/H200 80 GB and launch the vLLM template:
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


def _usage_dict(u: Any) -> dict[str, Any]:
    try:
        return u.model_dump()
    except AttributeError:
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", 0),
        }


def chat_stream(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout_s: float = 60.0,
) -> tuple[str, dict[str, Any]]:
    """Stream a plain (no-tools) completion to stdout. Returns (text, usage)."""
    chunks: list[str] = []
    usage: dict[str, Any] = {}
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        timeout=timeout_s,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
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


def _require_endpoint(profile: ModelProfile, profile_name: str) -> str:
    endpoint = profile.endpoint
    if not endpoint:
        st = replay_global().get("models", {}).get(profile_name, {})
        endpoint = st.get("endpoint")
    if not endpoint:
        raise MichaelError(
            f"no endpoint known for {profile_name!r} — run `up --model {profile_name}` first"
        )
    return endpoint


# ---------------------------------------------------------------------------
# Sandbox backends (local podman/docker, remote SSH+podman, or disabled)
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
        self, code: str, *, network: bool = False, timeout_s: int = 30,
        project: Optional[Project] = None,
    ) -> subprocess.CompletedProcess: ...


class DisabledSandboxBackend(SandboxBackend):
    def run(self, code, *, network=False, timeout_s=30, project=None):
        raise MichaelError(
            "sandbox unavailable: configure vps.host (recommended on phone) "
            "or install podman/docker locally"
        )


class LocalPodmanBackend(SandboxBackend):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.runtime = "podman" if shutil.which("podman") else "docker"

    def run(
        self, code: str, *, network: bool = False, timeout_s: int = 30,
        project: Optional[Project] = None,
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
                project=project,
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
                    project=project,
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
                project=project,
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
        self, code: str, *, network: bool = False, timeout_s: int = 30,
        project: Optional[Project] = None,
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
            project=project,
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
                project=project,
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
            project=project,
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
# Filesystem snapshot — listing + inlined contents for the LLM brief
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
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS]
        rel_dp = dp_path.relative_to(root)
        for fname in sorted(files):
            f = dp_path / fname
            try:
                size = f.stat().st_size
            except OSError:
                continue
            rel = (rel_dp / fname).as_posix() if str(rel_dp) != "." else fname
            listing_lines.append(f"{rel} ({size}b)")
            if size <= MAX_FILE_BYTES_INLINE and _is_text(f):
                text_files.append((f, size))

    parts: list[str] = []
    parts.append("Directory listing (relative to project root):")
    parts.append("\n".join(listing_lines) if listing_lines else "(empty)")
    parts.append("")
    parts.append(
        f"File contents (text only; per-file cap {MAX_FILE_BYTES_INLINE}b, "
        f"total cap {MAX_TOTAL_BYTES_INLINE}b):"
    )

    text_files.sort(key=lambda x: x[1])
    bodies: list[str] = []
    total = 0
    for f, size in text_files:
        rel = f.relative_to(root).as_posix()
        if total + size > MAX_TOTAL_BYTES_INLINE:
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


def _prompt_history_lines(project: Project) -> list[str]:
    out: list[str] = []
    n = 0
    for ev in iter_events(project.events_path):
        if ev.get("type") == "prompt.sent":
            n += 1
            prompt = (ev.get("payload") or {}).get("prompt", "")
            out.append(f"[{n}] {prompt}")
    return out


def _action_log_lines(project: Project) -> list[str]:
    out: list[str] = []
    n = 0
    for ev in iter_events(project.events_path):
        t = ev.get("type", "")
        p = ev.get("payload", {}) or {}
        if t == "tool.executed":
            n += 1
            out.append(f"[{n}] {p.get('summary', t)}")
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
}


def build_protocol(mode: str = "code") -> str:
    """Header 4 — the protocol Bible. The contract the LLM operates under.

    Generated fresh per package; references JA_PASSPHRASE so the literal
    passcode and the parser stay in lockstep.
    """
    addendum = _MODE_ADDENDUM.get(mode, _MODE_ADDENDUM["code"])
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
        f"THE {JA_PASSPHRASE!r} PASSCODE:",
        f"The user only sees your work when you END a message with the literal",
        f"bareword `{JA_PASSPHRASE}` (case-sensitive, on its own line or as the",
        f"trailing token). That is the ONLY signal Michael reads as 'surface",
        f"this to the user.' Until {JA_PASSPHRASE}, you are talking to Michael,",
        f"not the user.",
        "",
        f"Do NOT use {JA_PASSPHRASE} casually. Do NOT use it mid-thought. Do",
        f"NOT use it as a filler word. {JA_PASSPHRASE} means: 'I am done",
        f"iterating; this is the product I want the user to review.'",
        "",
        f"After {JA_PASSPHRASE}, Michael shows the user the staged delta and",
        f"asks one yes/no question. Yes = the change is committed and this",
        f"prompt cycle ends. No = the staging is discarded; the next user",
        f"prompt re-enters the loop and you will see the rejection in H3.",
        "",
        addendum,
        "",
        "Tools (full schemas in the API call):",
        "  write_file(path, content, expected_changes)        expected_changes required",
        "  apply_patch(path, unified_diff, expected_changes)  expected_changes required",
        "  read_file(path)                                    auto-executes",
        "  list_dir(path='.')                                 auto-executes",
        "  run_in_sandbox(python_code)                        isolated podman, no network",
        "  run_shell(cmd, timeout_s=60)                       runs in the project workspace",
        "",
        "All paths are relative to the project root. Do not escape with '..'.",
    ])


def build_header(
    project: Project,
    system_prompt: str,
    *,
    mode: str = "code",
) -> str:
    """Pack the four-header context package sent to a fresh LLM instance.

    H1 = user's prompts in this project (verbatim).
    H2 = filesystem snapshot.
    H3 = tool calls executed in this project (causal chain).
    H4 = the protocol Bible (build_protocol(mode)).

    The system_prompt sits above H4 as the operator's standing brief.
    """
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


# ---------------------------------------------------------------------------
# Tool schemas (passed to the LLM as `tools=[...]`)
# ---------------------------------------------------------------------------


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Overwrite (or create) a file in the project workspace. Path is "
                "relative to the project root; parent dirs are created. "
                "The change is applied to a staging copy of the project first. "
                "You MUST predict the resulting filesystem delta in "
                "`expected_changes` (every project-relative path that will be "
                "added, modified, or removed). If reality diverges from your "
                "prediction, Michael returns a mismatch error to you and the "
                "user is NOT prompted — re-propose. If `verify` is provided, "
                "it runs in the staging copy after the write; verify failures "
                "are reported back to you. If the prediction matches and verify "
                "passes (or is omitted), the user is shown the diff and asked "
                "to confirm before the change is committed to the real workspace."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to project root."},
                    "content": {"type": "string", "description": "Full file content."},
                    "expected_changes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Required. Project-relative paths you predict will be "
                            "added, modified, or removed. Mismatch with the actual "
                            "staged delta is returned to you as an error."
                        ),
                    },
                    "verify": {
                        "type": "string",
                        "description": (
                            "Optional shell command run in the staging copy after "
                            "applying the write. Exit 0 = pass; non-zero = fail "
                            "and the user is not bothered."
                        ),
                    },
                },
                "required": ["path", "content", "expected_changes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the project workspace. Auto-executes.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a directory in the project workspace. Auto-executes.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Apply a unified diff to a file. Goes through the same "
                "staging + predicted-delta + verify + user-confirm flow as "
                "write_file: `expected_changes` is required, mismatches are "
                "returned to you, and the user is only prompted on a clean "
                "match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "unified_diff": {"type": "string"},
                    "expected_changes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Required. Project-relative paths you predict will be "
                            "added, modified, or removed. Mismatch is returned "
                            "to you as an error."
                        ),
                    },
                    "verify": {"type": "string"},
                },
                "required": ["path", "unified_diff", "expected_changes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_in_sandbox",
            "description": (
                "Run Python code in an isolated podman sandbox: no network, "
                "read-only mount, dropped caps. Requires user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {"python_code": {"type": "string"}},
                "required": ["python_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command in the project workspace (NOT sandboxed). "
                "Requires user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "timeout_s": {"type": "integer", "default": 60},
                },
                "required": ["cmd"],
            },
        },
    },
]


def _resolve_in_project(project: Project, rel: str) -> pathlib.Path:
    root = pathlib.Path(project.path).resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise MichaelError(f"path escapes project root: {rel}") from e
    return candidate


def _summary_for(name: str, args: dict[str, Any]) -> str:
    if name == "write_file":
        return f"write_file({args.get('path', '?')}, {len(args.get('content', ''))}b)"
    if name == "read_file":
        return f"read_file({args.get('path', '?')})"
    if name == "list_dir":
        return f"list_dir({args.get('path', '.')})"
    if name == "apply_patch":
        return f"apply_patch({args.get('path', '?')})"
    if name == "run_in_sandbox":
        return f"run_in_sandbox({len(args.get('python_code', ''))}b)"
    if name == "run_shell":
        cmd = str(args.get("cmd", "?"))
        return f"run_shell({cmd[:80]}{'...' if len(cmd) > 80 else ''})"
    return f"{name}(?)"


def _format_proc_result(cp: subprocess.CompletedProcess) -> str:
    out = [f"rc={cp.returncode}"]
    if cp.stdout:
        out.append(f"stdout (truncated):\n{cp.stdout[-2000:]}")
    if cp.stderr:
        out.append(f"stderr (truncated):\n{cp.stderr[-1000:]}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Staging + trash helpers (verify-before-apply)
# ---------------------------------------------------------------------------


def _stage_ignore(directory: str, contents: list[str]) -> list[str]:
    out: list[str] = []
    for c in contents:
        p = pathlib.Path(directory) / c
        if p.is_dir() and (c.startswith(".") or c in SKIP_DIRS):
            out.append(c)
    return out


def _stage_project(project: Project) -> pathlib.Path:
    src = pathlib.Path(project.path).resolve()
    if not src.is_dir():
        raise MichaelError(f"project root does not exist: {src}")
    parent = pathlib.Path(tempfile.mkdtemp(prefix="michael-stage-", dir="/tmp"))
    dst = parent / src.name
    shutil.copytree(src, dst, ignore=_stage_ignore, symlinks=False)
    return dst


def _file_hashes(root: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    root = root.resolve()
    if not root.is_dir():
        return out
    for dp, dirs, files in os.walk(root):
        dp_path = pathlib.Path(dp)
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in SKIP_DIRS]
        for fn in files:
            fp = dp_path / fn
            try:
                data = fp.read_bytes()
            except OSError:
                continue
            rel = str(fp.relative_to(root))
            out[rel] = hashlib.sha256(data).hexdigest()
    return out


def _diff_hashes(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    added = sorted(p for p in after if p not in before)
    removed = sorted(p for p in before if p not in after)
    modified = sorted(p for p in after if p in before and after[p] != before[p])
    return {"added": added, "removed": removed, "modified": modified}


def _check_expected(expected: list[str], delta: dict[str, list[str]]) -> str:
    actual = set(delta["added"]) | set(delta["modified"]) | set(delta["removed"])
    expected_set = set(expected)
    extra = sorted(actual - expected_set)
    missing = sorted(expected_set - actual)
    if not extra and not missing:
        return ""
    parts: list[str] = []
    if extra:
        parts.append(f"extra={extra}")
    if missing:
        parts.append(f"missing={missing}")
    return "; ".join(parts)


def _run_verify(cmd: str, cwd: pathlib.Path, *, timeout_s: int = 60) -> tuple[int, str]:
    try:
        cp = subprocess.run(
            ["bash", "-c", cmd],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "")[-1000:] if isinstance(e.stdout, str) else ""
        errs = (e.stderr or "")[-500:] if isinstance(e.stderr, str) else ""
        return 124, f"verify timed out after {timeout_s}s\nstdout:\n{out}\nstderr:\n{errs}"
    out = ""
    if cp.stdout:
        out += f"stdout (truncated):\n{cp.stdout[-1500:]}\n"
    if cp.stderr:
        out += f"stderr (truncated):\n{cp.stderr[-500:]}"
    return cp.returncode, out


def _apply_in_staging(name: str, args: dict[str, Any], stage_root: pathlib.Path) -> None:
    rel = str(args.get("path", ""))
    target = (stage_root / rel).resolve()
    try:
        target.relative_to(stage_root.resolve())
    except ValueError as e:
        raise MichaelError(f"path escapes project root: {rel}") from e

    if name == "write_file":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(args["content"]))
        return

    if name == "apply_patch":
        if not target.is_file():
            raise MichaelError(f"apply_patch target does not exist: {rel}")
        if not shutil.which("patch"):
            raise MichaelError("`patch` not installed on host (apt install patch)")
        diff = str(args["unified_diff"])
        cp = subprocess.run(
            ["patch", "--no-backup-if-mismatch", "-u", str(target)],
            input=diff,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if cp.returncode != 0:
            raise MichaelError(
                f"patch failed in staging (rc={cp.returncode}): "
                f"{(cp.stderr or '')[-500:]}"
            )
        return

    raise MichaelError(f"_apply_in_staging: unknown tool {name}")


def _save_trash(
    project: Project,
    op_name: str,
    args: dict[str, Any],
    delta: dict[str, list[str]],
    real_root: pathlib.Path,
    *,
    verify_rc: Optional[int],
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    trash_id = f"{ts}-{uuid.uuid4().hex[:6]}"
    trash_dir = PROJECTS_DIR / project.slug / "trash" / trash_id
    trash_dir.mkdir(parents=True, exist_ok=True)
    before_dir = trash_dir / "before"
    before_dir.mkdir(exist_ok=True)
    for rel in delta["modified"] + delta["removed"]:
        src = real_root / rel
        if not src.is_file():
            continue
        dst = before_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except OSError:
            continue
    metadata = {
        "trash_id": trash_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": op_name,
        "summary": _summary_for(op_name, args),
        "args": args,
        "delta": delta,
        "verify_rc": verify_rc,
    }
    (trash_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    return trash_id


def _sync_to_real(stage_root: pathlib.Path, real_root: pathlib.Path, delta: dict[str, list[str]]) -> None:
    for rel in delta["added"] + delta["modified"]:
        src = stage_root / rel
        dst = real_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for rel in delta["removed"]:
        dst = real_root / rel
        if dst.is_file():
            try:
                dst.unlink()
            except OSError:
                pass


def _list_trash(project: Project) -> list[dict[str, Any]]:
    trash_root = PROJECTS_DIR / project.slug / "trash"
    if not trash_root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(trash_root.iterdir()):
        if not d.is_dir():
            continue
        meta = d / "metadata.json"
        if not meta.is_file():
            continue
        try:
            out.append(json.loads(meta.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _undo_one(project: Project, trash_id: Optional[str] = None) -> dict[str, Any]:
    trash_root = PROJECTS_DIR / project.slug / "trash"
    if not trash_root.is_dir():
        raise MichaelError("no trash entries to undo")
    entries = sorted([d for d in trash_root.iterdir() if d.is_dir()])
    if not entries:
        raise MichaelError("no trash entries to undo")
    if trash_id:
        target = trash_root / trash_id
        if not target.is_dir():
            raise MichaelError(f"unknown trash id: {trash_id}")
    else:
        target = entries[-1]
    metadata = json.loads((target / "metadata.json").read_text())
    delta = metadata.get("delta", {}) or {}
    real_root = pathlib.Path(project.path).resolve()
    for rel in delta.get("modified", []) + delta.get("removed", []):
        src = target / "before" / rel
        if not src.is_file():
            continue
        dst = real_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for rel in delta.get("added", []):
        dst = real_root / rel
        if dst.is_file():
            try:
                dst.unlink()
            except OSError:
                pass
    shutil.rmtree(target, ignore_errors=True)
    return metadata


# ---------------------------------------------------------------------------
# Tool execution (read/list/run paths; write_file & apply_patch go through
# execute_with_staging instead).
# ---------------------------------------------------------------------------


def execute_tool(
    name: str,
    args: dict[str, Any],
    project: Project,
    cfg: Config,
    backend: SandboxBackend,
) -> str:
    if name == "read_file":
        target = _resolve_in_project(project, str(args["path"]))
        if not target.is_file():
            return "error: not a file"
        try:
            text = target.read_text(errors="replace")
        except OSError as e:
            return f"error: {e}"
        if len(text) > 200_000:
            return f"file too large ({len(text)}b) — refusing to read full content"
        return text

    if name == "list_dir":
        target = _resolve_in_project(project, str(args.get("path", ".")))
        if not target.is_dir():
            return "error: not a directory"
        rows = []
        for child in sorted(target.iterdir()):
            try:
                size = child.stat().st_size
            except OSError:
                size = -1
            kind = "dir" if child.is_dir() else "file"
            rows.append(f"{kind}\t{child.name}\t{size}")
        return "\n".join(rows) or "(empty)"

    if name == "run_in_sandbox":
        cp = backend.run(
            str(args["python_code"]),
            network=False,
            timeout_s=cfg.sandbox.timeout_s,
            project=project,
        )
        return _format_proc_result(cp)

    if name == "run_shell":
        timeout_s = int(args.get("timeout_s", 60))
        cwd = pathlib.Path(project.path).resolve()
        try:
            cp = subprocess.run(
                ["bash", "-c", str(args["cmd"])],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            return f"timed out after {timeout_s}s; partial stdout:\n{(e.stdout or '')[-1000:]}"
        return _format_proc_result(cp)

    return f"error: unknown tool {name}"


# ---------------------------------------------------------------------------
# Verify-before-apply flow (write_file, apply_patch)
# ---------------------------------------------------------------------------


def _format_delta(delta: dict[str, list[str]]) -> str:
    parts: list[str] = [
        f"files added:    {len(delta['added'])}",
        f"files modified: {len(delta['modified'])}",
        f"files removed:  {len(delta['removed'])}",
    ]
    if delta["added"]:
        parts.append("  + " + "\n  + ".join(delta["added"]))
    if delta["modified"]:
        parts.append("  ~ " + "\n  ~ ".join(delta["modified"]))
    if delta["removed"]:
        parts.append("  - " + "\n  - ".join(delta["removed"]))
    return "\n".join(parts)


def _format_staging_preview(
    name: str,
    args: dict[str, Any],
    project: Project,
    delta: dict[str, list[str]],
    verify_rc: Optional[int],
    verify_out: str,
    mismatch: str,
) -> str:
    sections: list[str] = []
    if name == "write_file":
        try:
            real_target = _resolve_in_project(project, str(args["path"]))
            old = real_target.read_text(errors="replace") if real_target.is_file() else ""
        except MichaelError:
            old = ""
        diff = "".join(difflib.unified_diff(
            old.splitlines(keepends=True),
            str(args["content"]).splitlines(keepends=True),
            fromfile=f"a/{args.get('path', '?')}",
            tofile=f"b/{args.get('path', '?')}",
        )) or "(no changes)"
        sections.append(diff)
    elif name == "apply_patch":
        sections.append(
            f"patch target: {args.get('path', '?')}\n\n"
            f"{args.get('unified_diff', '')}"
        )

    sections.append("─── staging preview ───")
    sections.append(_format_delta(delta))

    if verify_rc is not None:
        sections.append("")
        sections.append(f"verify: rc={verify_rc}")
        if verify_out:
            sections.append(verify_out[-800:])

    if mismatch:
        sections.append("")
        sections.append(f"⚠ expected_changes mismatch: {mismatch}")

    return "\n\n".join(sections)


@dataclass
class PendingChanges:
    """Per-agent-loop staging state. The LLM and Michael iterate against a
    persistent stage; entries accumulate until the LLM emits the Ja passcode
    and the user approves (commit) or rejects (discard).
    """

    stage_root: Optional[pathlib.Path] = None
    change_log: list[dict[str, Any]] = field(default_factory=list)

    def ensure_stage(self, project: Project) -> pathlib.Path:
        if self.stage_root is None:
            self.stage_root = _stage_project(project)
        return self.stage_root

    def discard(self) -> None:
        if self.stage_root is not None:
            shutil.rmtree(self.stage_root.parent, ignore_errors=True)
            self.stage_root = None
        self.change_log.clear()


def _snapshot_file(stage_root: pathlib.Path, rel: str) -> tuple[bool, Optional[bytes]]:
    """Capture pre-apply content of one staged file so a verify-failure can
    roll back without polluting prior pending changes."""
    target = stage_root / rel
    if not target.is_file():
        return False, None
    try:
        return True, target.read_bytes()
    except OSError:
        return True, None


def _restore_file(
    stage_root: pathlib.Path, rel: str, existed: bool, blob: Optional[bytes]
) -> None:
    target = stage_root / rel
    if not existed:
        if target.is_file():
            try:
                target.unlink()
            except OSError:
                pass
        return
    if blob is None:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)


def _format_review(
    name: str,
    args: dict[str, Any],
    project: Project,
    stage_root: pathlib.Path,
    delta: dict[str, list[str]],
    verify_rc: Optional[int],
    verify_out: str,
    expected: list[str],
    mismatch: str,
) -> str:
    """Structured review fed back to the LLM. Includes prediction vs reality,
    full diff, and verify output. Mismatch is informational, not a rejection.
    """
    sections: list[str] = []
    sections.append(f"tool: {name}({args.get('path', '?')})")
    sections.append(
        f"predicted: added/modified/removed = {sorted(expected)}"
    )
    sections.append(
        f"actual:    added={delta['added']}  "
        f"modified={delta['modified']}  removed={delta['removed']}"
    )
    if mismatch:
        sections.append(f"prediction-vs-reality: {mismatch}")
    else:
        sections.append("prediction-vs-reality: match")

    if name == "write_file":
        rel = str(args.get("path", "?"))
        try:
            real_target = _resolve_in_project(project, rel)
            old = real_target.read_text(errors="replace") if real_target.is_file() else ""
        except MichaelError:
            old = ""
        diff = "".join(difflib.unified_diff(
            old.splitlines(keepends=True),
            str(args.get("content", "")).splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )) or "(no changes)"
        sections.append("diff (vs. real workspace):")
        sections.append(diff)
    elif name == "apply_patch":
        sections.append("patch applied:")
        sections.append(str(args.get("unified_diff", "")))

    if verify_rc is not None:
        tail = (verify_out or "")[-1200:]
        sections.append(f"verify rc={verify_rc}\n{tail}")

    sections.append(
        f"staging committed at {stage_root}; this change is pending. "
        "Continue iterating or end your message with the Ja passcode to "
        "surface to the user."
    )
    return "\n\n".join(sections)


def execute_with_staging(
    name: str,
    args: dict[str, Any],
    project: Project,
    cfg: Config,
    pending: PendingChanges,
) -> str:
    """Apply the LLM's proposal in the per-loop persistent staging dir,
    compute prediction vs reality, and return a structured review back to
    the LLM. Does NOT prompt the user. Does NOT sync to the real workspace.

    The estimation mandate is enforced: missing `expected_changes` returns
    an error (the mandate survives). A mismatch between predicted and actual
    delta is returned as review data — NOT an auto-rejection. Verify
    failures roll back this single call and surface the verify output to
    the LLM. The LLM keeps iterating until it emits the Ja passcode.
    """
    expected_raw = args.get("expected_changes")
    expected_list: list[str] = (
        [str(x) for x in expected_raw]
        if isinstance(expected_raw, list) else []
    )
    if not expected_list:
        append_event(
            "tool.delta_missing",
            {"tool": name, "summary": _summary_for(name, args)},
            project=project,
        )
        return (
            "error: expected_changes is required and must be a non-empty "
            "list of project-relative paths you predict will be added, "
            "modified, or removed. Predict the delta, then re-propose."
        )

    try:
        stage_root = pending.ensure_stage(project)
    except MichaelError as e:
        return f"error: staging failed: {e}"

    rel = str(args.get("path", ""))
    existed, blob = _snapshot_file(stage_root, rel)
    before = _file_hashes(stage_root)
    try:
        _apply_in_staging(name, args, stage_root)
    except MichaelError as e:
        _restore_file(stage_root, rel, existed, blob)
        return f"error applying in staging: {e}"
    after = _file_hashes(stage_root)
    delta = _diff_hashes(before, after)

    verify_rc: Optional[int] = None
    verify_out = ""
    verify_cmd = args.get("verify")
    if isinstance(verify_cmd, str) and verify_cmd.strip():
        verify_rc, verify_out = _run_verify(verify_cmd, stage_root, timeout_s=60)
        if verify_rc != 0:
            _restore_file(stage_root, rel, existed, blob)
            append_event(
                "tool.verify_failed",
                {
                    "tool": name,
                    "summary": _summary_for(name, args),
                    "verify_cmd": verify_cmd,
                    "verify_rc": verify_rc,
                    "delta": delta,
                },
                project=project,
            )
            return (
                f"verify failed in staging (rc={verify_rc}); this call was "
                f"rolled back. Prior pending changes are intact.\n"
                f"delta this call would have made: {delta}\n"
                f"verify output:\n{verify_out[-1500:]}"
            )

    mismatch = _check_expected(expected_list, delta)
    if mismatch:
        append_event(
            "tool.delta_mismatch",
            {
                "tool": name,
                "summary": _summary_for(name, args),
                "expected": expected_list,
                "delta": delta,
                "mismatch": mismatch,
            },
            project=project,
        )
    append_event(
        "tool.staged",
        {
            "tool": name,
            "summary": _summary_for(name, args),
            "delta": delta,
            "verify_rc": verify_rc,
            "mismatch": mismatch,
        },
        project=project,
    )
    pending.change_log.append({
        "tool": name,
        "args": args,
        "delta": delta,
        "verify_rc": verify_rc,
        "expected": expected_list,
        "mismatch": mismatch,
    })
    return _format_review(
        name, args, project, stage_root, delta,
        verify_rc, verify_out, expected_list, mismatch,
    )


def commit_pending(project: Project, pending: PendingChanges) -> list[dict[str, Any]]:
    """At Ja-time + user-yes: sync pending stage to the real workspace,
    save trash for each entry, append tool.executed events, and discard
    the stage. Returns the per-entry summaries for display."""
    if pending.stage_root is None or not pending.change_log:
        return []
    real_root = pathlib.Path(project.path).resolve()
    summaries: list[dict[str, Any]] = []
    for entry in pending.change_log:
        delta = entry["delta"]
        trash_id = _save_trash(
            project, entry["tool"], entry["args"], delta, real_root,
            verify_rc=entry.get("verify_rc"),
        )
        _sync_to_real(pending.stage_root, real_root, delta)
        summary = (
            f"{_summary_for(entry['tool'], entry['args'])} → applied "
            f"+{len(delta['added'])} ~{len(delta['modified'])} "
            f"-{len(delta['removed'])} trash_id={trash_id}"
        )
        append_event(
            "tool.executed",
            {
                "tool": entry["tool"],
                "args": entry["args"],
                "summary": summary[:240],
                "trash_id": trash_id,
                "delta": delta,
                "verify_rc": entry.get("verify_rc"),
            },
            project=project,
        )
        summaries.append({"trash_id": trash_id, "summary": summary})
    pending.discard()
    return summaries


def dispatch_tool_call(
    name: str,
    args: dict[str, Any],
    project: Project,
    cfg: Config,
    backend: SandboxBackend,
    pending: PendingChanges,
) -> str:
    """Route one LLM tool call. write_file/apply_patch stage into the per-loop
    `pending` state and return a review (no user prompt). Other tools auto-
    execute (read_file, list_dir) or run after one user confirmation
    (run_in_sandbox, run_shell)."""
    summary = _summary_for(name, args)

    if name in AUTO_EXEC_TOOLS:
        try:
            result = execute_tool(name, args, project, cfg, backend)
        except MichaelError as e:
            result = f"error: {e}"
        first = (result.splitlines()[0] if result else "ok")[:120]
        append_event(
            "tool.executed",
            {
                "tool": name,
                "args": args,
                "summary": f"{summary} → {first}",
                "result_chars": len(result),
            },
            project=project,
        )
        return result

    if name in ("write_file", "apply_patch"):
        return execute_with_staging(name, args, project, cfg, pending)

    try:
        decision, final_args = confirm_tool_call(name, args, project)
    except (KeyboardInterrupt, typer.Abort):
        decision, final_args = "no", args
    if decision == "no":
        append_event(
            "tool.rejected",
            {"tool": name, "args": args, "summary": summary},
            project=project,
        )
        return "[user rejected this tool call]"
    try:
        result = execute_tool(name, final_args, project, cfg, backend)
    except MichaelError as e:
        result = f"error: {e}"
    first = (result.splitlines()[0] if result else "ok")[:120]
    append_event(
        "tool.executed",
        {
            "tool": name,
            "args": final_args,
            "summary": f"{_summary_for(name, final_args)} → {first}",
            "result_chars": len(result),
        },
        project=project,
    )
    return result


# ---------------------------------------------------------------------------
# Y/n/Edit confirmation for run_in_sandbox / run_shell only
# (write_file & apply_patch do their own preview inside execute_with_staging)
# ---------------------------------------------------------------------------


def _render_for_confirmation(name: str, args: dict[str, Any], project: Project) -> tuple[str, str]:
    if name == "run_in_sandbox":
        return str(args.get("python_code", "")), "python"
    if name == "run_shell":
        return f"cmd: {args.get('cmd', '?')}\ncwd: {project.path}", "bash"
    return json.dumps(args, indent=2), "json"


def _edit_args(name: str, args: dict[str, Any]) -> Optional[dict[str, Any]]:
    if name == "write_file":
        edited = typer.edit(str(args["content"]))
        return {**args, "content": edited} if edited is not None else None
    if name == "apply_patch":
        edited = typer.edit(str(args["unified_diff"]))
        return {**args, "unified_diff": edited} if edited is not None else None
    if name == "run_in_sandbox":
        edited = typer.edit(str(args["python_code"]))
        return {**args, "python_code": edited} if edited is not None else None
    if name == "run_shell":
        new_cmd = typer.prompt("new cmd", default=str(args.get("cmd", "")))
        return {**args, "cmd": new_cmd}
    edited = typer.edit(json.dumps(args, indent=2))
    if edited is None:
        return None
    try:
        return json.loads(edited)
    except json.JSONDecodeError:
        return None


def confirm_tool_call(
    name: str,
    args: dict[str, Any],
    project: Project,
) -> tuple[str, dict[str, Any]]:
    while True:
        rendered, lexer = _render_for_confirmation(name, args, project)
        console.print(
            Panel(
                Syntax(rendered, lexer, theme="ansi_dark", word_wrap=True),
                title=f"[cyan]propose[/] {name}",
                border_style="cyan",
            )
        )
        choice = (typer.prompt("Apply? [Y]es / [n]o / [e]dit", default="y") or "").strip().lower()
        if choice in ("", "y", "yes"):
            return "yes", args
        if choice in ("n", "no"):
            return "no", args
        if choice in ("e", "edit"):
            edited = _edit_args(name, args)
            if edited is None:
                err.print("editor returned no content; try again")
                continue
            args = edited
            continue
        err.print(f"unknown choice: {choice!r}")


# ---------------------------------------------------------------------------
# Subcommand implementations (called by both Typer and the REPL dispatcher)
# ---------------------------------------------------------------------------


def cmd_init() -> None:
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    if not GLOBAL_CONFIG_PATH.is_file():
        make_stub_config().save()
        console.print(f"[green]wrote stub[/] {GLOBAL_CONFIG_PATH}")
    else:
        console.print(f"[dim]config exists[/] {GLOBAL_CONFIG_PATH}")
    append_event("config.loaded", {"path": str(GLOBAL_CONFIG_PATH)})
    console.print(
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
        console.print("0")
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
    console.print(table)


def cmd_new(name: Optional[str]) -> None:
    if not name:
        name = (typer.prompt("name") or "").strip()
    if not name:
        err.print("name is required")
        return
    default_path = pathlib.Path.cwd() / slugify(name)
    path_str = typer.prompt("path", default=str(default_path))
    path = pathlib.Path(path_str).expanduser().resolve()
    proj = create_project(name, path)
    set_active_slug(proj.slug)
    append_event("project.activated", {"slug": proj.slug})
    console.print(f"[green]created[/] {proj.slug} at {proj.path}")


def cmd_use(slug: str) -> None:
    proj = Project.load(slug)
    set_active_slug(proj.slug)
    append_event("project.activated", {"slug": proj.slug})
    console.print(f"[green]active[/] {proj.slug}")


def cmd_current() -> None:
    p = get_active_project()
    if not p:
        console.print("(no active project)")
        return
    console.print(f"{p.slug} — {p.name} — {p.path}")


def cmd_config() -> None:
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    if not GLOBAL_CONFIG_PATH.is_file():
        make_stub_config().save()
    help_lines = [f"[bold]{k}[/] — {v}" for k, v in CONFIG_HELP.items()]
    console.print(
        Panel(
            "\n".join(help_lines),
            title=f"config: {GLOBAL_CONFIG_PATH}",
            border_style="green",
        )
    )
    current_text = GLOBAL_CONFIG_PATH.read_text()
    edited = typer.edit(current_text, extension=".json")
    if edited is None or edited == current_text:
        console.print("[dim]no changes[/]")
        return
    try:
        json.loads(edited)
    except json.JSONDecodeError as e:
        err.print(f"invalid JSON, not saved: {e}")
        return
    GLOBAL_CONFIG_PATH.write_text(edited)
    os.chmod(GLOBAL_CONFIG_PATH, 0o600)
    console.print("[green]config saved[/]")


def cmd_up(model: Optional[str]) -> None:
    cfg = Config.load()
    name, profile = cfg.get_model(model or None)
    if not profile.vast_instance_id:
        raise MichaelError(f"models.{name}.vast_instance_id is not set (run `config`)")
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


def cmd_down(model: Optional[str]) -> None:
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
    for name, profile in cfg.models.items():
        st = state.get("models", {}).get(name, {})
        table.add_row(
            f"  {name}",
            f"state={st.get('instance_state', 'unknown')}  "
            f"endpoint={st.get('endpoint') or profile.endpoint or '—'}",
        )

    table.add_row("errors (global)", str(state["errors"]))
    console.print(table)


def cmd_ask(prompt: str, model: Optional[str]) -> None:
    """One-shot LLM call. If a project is active, includes its brief; else
    runs chat-only with just the system prompt."""
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


def _tools_for_mode(mode: str) -> list[dict[str, Any]]:
    """code/nitro = full toolset; discussion = read-only tools only."""
    if mode == "discussion":
        return [t for t in TOOLS if t["function"]["name"] in AUTO_EXEC_TOOLS]
    return TOOLS


def _resolve_nitro_model(cfg: Config, model: Optional[str]) -> tuple[str, ModelProfile]:
    """Pick the heavy model for nitro: explicit --model wins, then 'nitro',
    then 'big'. No silent fallback to the default profile."""
    if model:
        return cfg.get_model(model)
    for candidate in ("nitro", "big"):
        if candidate in cfg.models:
            return candidate, cfg.models[candidate]
    raise MichaelError(
        "nitro requires a 'nitro' or 'big' model profile in config "
        "(or pass --model NAME explicitly)"
    )


_NUDGE_NO_JA = (
    "system reminder: you ended your turn without tool calls and without "
    f"the {JA_PASSPHRASE!r} passcode. Either use tools to keep iterating, "
    f"or end your message with `{JA_PASSPHRASE}` on its own line to surface "
    "your work to the user. Until then you are talking to Michael, not "
    "the user."
)


def _present_pending_to_user(
    project: Project,
    pending: PendingChanges,
    final_text: str,
) -> bool:
    """Render the LLM's accumulated pending changes for the user and ask
    one yes/no. Returns True on apply, False on reject. If there are no
    pending changes, just shows the LLM's final text and returns True."""
    if final_text:
        console.print(
            Panel(final_text, title="assistant — Ja", border_style="green")
        )
    if not pending.change_log:
        return True

    for i, entry in enumerate(pending.change_log, 1):
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
            except MichaelError:
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
        console.print(
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
) -> None:
    """Shared agent-loop body for `run`, `new code`, `new discussion`, `nitro`.

    Per user prompt: a fresh PendingChanges holds the persistent staging dir
    and the change log. The LLM and Michael iterate indefinitely (no turn
    cap) until the LLM emits the Ja passcode (then user is asked) or the
    user hits Ctrl-C (graceful abort). The user is NEVER prompted mid-loop.
    """
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
    console.print(
        f"[bold cyan]michael {verb_label}[/] [dim]project={project.slug}  "
        f"model={name}  mode={mode}  sandbox={backend_label}[/]"
    )
    console.print(
        f"[dim]empty line or 'quit' to exit · Ctrl-C aborts an in-flight "
        f"loop · LLM surfaces with the {JA_PASSPHRASE!r} passcode[/]"
    )

    session = PromptSession(
        history=FileHistory(str(REPL_HISTORY_PATH)),
        auto_suggest=AutoSuggestFromHistory(),
    )

    while True:
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

        # Fresh brief and fresh pending state per user prompt. Tool messages
        # only flow within this loop; pending stage is discarded at end.
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
                console.print(
                    f"[dim]· turn {turn}: model thinking…[/]"
                )
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
                    err.print(f"LLM error: {e}")
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
                        console.print(
                            f"[dim]· turn {turn}: tool {tc.function.name}[/]"
                        )
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
                    # After tool-call execution, loop back for the next LLM turn.
                    continue

                # No tool calls. Either the LLM said Ja → present, or it
                # stopped without saying Ja → nudge it back into the loop.
                if _message_ends_with_ja(content):
                    ja_received = True
                    append_event(
                        "assistant.ja",
                        {
                            "turn": turn,
                            "pending": len(pending.change_log),
                        },
                        project=project,
                    )
                    break
                console.print(
                    f"[yellow]· turn {turn}: no {JA_PASSPHRASE} and no tool "
                    f"calls — nudging the model back into the loop[/]"
                )
                messages.append({"role": "system", "content": _NUDGE_NO_JA})
                continue
        except KeyboardInterrupt:
            err.print(
                f"\nturn {turn}: aborted by user; pending changes discarded"
            )
            append_event(
                "agent.aborted",
                {"turn": turn, "pending": len(pending.change_log)},
                project=project,
            )
            pending.discard()
            continue

        if not ja_received:
            # LLM error or non-Ja exit — discard staging, bail out.
            pending.discard()
            continue

        # Ja received. Show the user the pending product and ask one y/n.
        approved = _present_pending_to_user(project, pending, content)
        if approved:
            summaries = commit_pending(project, pending)
            if summaries:
                console.print(
                    f"[green]applied[/] {len(summaries)} change(s)"
                )
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
            console.print("[yellow]rejected[/] pending changes discarded")


def cmd_run(model: Optional[str]) -> None:
    project = require_active_project()
    cfg = Config.load()
    name, profile = cfg.get_model(model or None)
    _run_agent_loop(project, cfg, name, profile, mode="code", verb_label="run")


def cmd_new_code(model: Optional[str]) -> None:
    """Fresh agent loop in code mode — full toolset, predicted-delta gate."""
    project = require_active_project()
    cfg = Config.load()
    name, profile = cfg.get_model(model or None)
    _run_agent_loop(
        project, cfg, name, profile, mode="code", verb_label="new code"
    )


def cmd_new_discussion(model: Optional[str]) -> None:
    """Fresh agent loop in discussion mode — read-only tools, no writes/exec."""
    project = require_active_project()
    cfg = Config.load()
    name, profile = cfg.get_model(model or None)
    _run_agent_loop(
        project, cfg, name, profile, mode="discussion",
        verb_label="new discussion",
    )


def cmd_nitro(model: Optional[str]) -> None:
    """Fresh agent loop on the heavy model — same contract as code mode."""
    project = require_active_project()
    cfg = Config.load()
    name, profile = _resolve_nitro_model(cfg, model or None)
    console.print(
        f"[bold yellow]⚡ nitro engaged[/] [dim]heavy model `{name}` — "
        f"cold-start may take a few minutes; stop with `down --model {name}` "
        "when finished[/]"
    )
    _run_agent_loop(
        project, cfg, name, profile, mode="nitro", verb_label="nitro"
    )


def cmd_log(tail: int) -> None:
    project = get_active_project()
    if project:
        events = iter_events(project.events_path)
        title = f"events (project: {project.slug})"
    else:
        events = iter_events(GLOBAL_EVENTS_PATH)
        title = "events (global)"
    if not events:
        console.print("[dim](no events)[/]")
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
    console.print(table)


def cmd_undo(list_only: bool = False, trash_id: Optional[str] = None) -> None:
    project = require_active_project()
    if list_only:
        entries = _list_trash(project)
        if not entries:
            console.print("(no trash)")
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
        console.print(table)
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
    console.print(
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
    console.print(
        Panel(
            stdout_tail or "(empty)",
            title=f"stdout (rc={cp.returncode})",
            border_style="green" if cp.returncode == 0 else "red",
        )
    )
    if stderr_tail:
        console.print(Panel(stderr_tail, title="stderr", border_style="red"))


def cmd_ssh_test() -> None:
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
# Typer command bindings (one-shot from the user's shell)
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
) -> None:
    """Fresh agent loop on the heavy model (cold-start aware)."""
    cmd_nitro(model or None)


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
    model: str = typer.Option("", "--model", "-m", help="Model profile name."),
) -> None:
    """Multi-turn tool-calling agent loop in the active project."""
    cmd_run(model or None)


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
    """True when the user hasn't filled in any vast/model credentials yet."""
    if not GLOBAL_CONFIG_PATH.is_file():
        return True
    try:
        cfg = Config.load()
    except MichaelError:
        return True
    if not cfg.vast_api_key:
        return True
    return not any(p.vast_instance_id for p in cfg.models.values())


class MichaelCompleter(Completer):
    """Tab-completion for the REPL: command names, then context-specific args."""

    LOG_FLAGS = ("--tail", "-n")
    UP_FLAGS = ("--model", "-m")
    DOWN_FLAGS = ("--model", "-m")
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
        if head in ("up", "down", "run", "ask", "nitro"):
            prefix = words[-1] if not at_boundary else ""
            for f in self.UP_FLAGS:
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
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(REPL_HISTORY_PATH)),
        auto_suggest=AutoSuggestFromHistory(),
        completer=MichaelCompleter(),
        complete_while_typing=False,
    )
    console.print("hey")
    if _config_is_unset():
        console.print(
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
        except MichaelError as e:
            err.print(f"michael: {e}")
        except typer.Abort:
            err.print("aborted")
        except KeyboardInterrupt:
            err.print("interrupted")


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
        err.print(f"parse error: {e}")
        return
    if not parts:
        return
    cmd, rest = parts[0], parts[1:]

    if cmd == "help":
        console.print("commands: " + ", ".join(sorted(REPL_COMMANDS)))
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
            err.print("usage: use <slug>")
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
        # Strip the flag pair from rest before joining as the prompt.
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
            err.print("usage: ask <prompt> [--model NAME]")
            return
        cmd_ask(" ".join(prompt_parts), model)
    elif cmd == "run":
        cmd_run(_opt_value(rest, "--model", "-m"))
    elif cmd == "nitro":
        cmd_nitro(_opt_value(rest, "--model", "-m"))
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
            err.print("usage: sandbox <file>")
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
        err.print(f"unknown command: {cmd!r}. try 'help'.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    try:
        if len(sys.argv) == 1:
            repl()
        else:
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
