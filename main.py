"""
michael — air-gapped, event-sourced, AI-native control loop.

Runs on a constrained Ubuntu 24.04 LTS VPS and orchestrates a remote Vast.ai
GPU instance that serves an open-source coding LLM via a vLLM
OpenAI-compatible API. The LLM proposes structured tool calls; michael parses
them, asks the user for Y/n/e on every write or execution, and runs them on
the host (or in a podman sandbox for `run_in_sandbox`).

Architecture:
- Bare `michael` enters a REPL. `michael <subcmd>` runs one-shot from the
  user's shell. Inside the REPL, commands have no `michael` prefix.
- State at ~/.michael/. Global: instance lifecycle, vast/vllm keys.
  Per-project at ~/.michael/projects/<slug>/: prompts, actions, events.
- Every user prompt rebuilds a fresh "header package" (system message) from
  the project log and a live filesystem snapshot. The LLM is amnesiac across
  user prompts; the project log is its memory. Within one user prompt, the
  agent loop iterates with the OpenAI tool-call protocol so the model can see
  its own tool results and continue, but those messages are discarded when
  the user prompt finishes.

Commands (work as one-shot or inside the REPL):
    show                        list projects
    new project [name]          create a new project (prompts for path)
    use <slug>                  switch active project
    current                     print the active project
    config                      open ~/.michael/config.json in $EDITOR
    up | down | status          Vast.ai instance lifecycle
    ask "<prompt>"              one-shot LLM call inside the active project
    run                         multi-turn tool-calling agent loop
    log [--tail N]              show project (or global) event log
    sandbox <file.py>           run a file in the throwaway sandbox
    quit | exit                 leave REPL
"""

from __future__ import annotations

import difflib
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

import httpx
import typer
from openai import OpenAI
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
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

RUNTIME = "podman" if shutil.which("podman") else "docker"

# Filesystem snapshot caps (per-file inline, total inline)
MAX_FILE_BYTES_INLINE = 50_000
MAX_TOTAL_BYTES_INLINE = 500_000
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv",
    "dist", "build", ".mypy_cache", ".pytest_cache",
    ".ruff_cache", ".tox", ".idea", ".vscode", ".next",
    "target", ".cache",
}

# Tools the LLM may call without a Y/n/e prompt.
AUTO_EXEC_TOOLS = {"read_file", "list_dir"}

# Hard cap on agent-loop iterations within a single user prompt.
MAX_AGENT_TURNS = 25


class MichaelError(RuntimeError):
    """Domain error surfaced to the user with a clean message."""


# ---------------------------------------------------------------------------
# Global config
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
    request_timeout_s: int = 120
    boot_poll_s: int = 10
    endpoint: Optional[str] = None  # cached after `up`

    @classmethod
    def load(cls) -> "Config":
        data: dict[str, Any] = {}
        if GLOBAL_CONFIG_PATH.is_file():
            try:
                data = json.loads(GLOBAL_CONFIG_PATH.read_text())
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
        STATE_DIR.mkdir(mode=0o700, exist_ok=True)
        GLOBAL_CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        os.chmod(GLOBAL_CONFIG_PATH, 0o600)


CONFIG_HELP: dict[str, str] = {
    "vast_api_key": "Vast.ai console API key.",
    "vast_instance_id": "Numeric ID of the rented GPU instance.",
    "vllm_api_key": "API key passed to vLLM at launch (or empty).",
    "vllm_internal_port": "Container-internal port vLLM listens on (default 8000).",
    "model_name": "Served model name (default qwen3-coder).",
    "sandbox_image": "Tag of the sandbox image built by bootstrap.sh.",
    "sandbox_memory_mb": "Sandbox memory cap in MB.",
    "sandbox_cpus": "Sandbox CPU cap.",
    "sandbox_pids": "Sandbox PID cap.",
    "request_timeout_s": "LLM request timeout (seconds).",
    "boot_poll_s": "Poll interval while waiting for vLLM to come up.",
    "endpoint": "Last-known vLLM endpoint URL (auto-set by `up`).",
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
            out.append(
                Project(**{k: data[k] for k in Project.__dataclass_fields__})
            )
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
# Event log (global + per-project)
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
    seq = _last_seq(path) + 1
    event = {
        "seq": seq,
        "ts": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "type": type_,
        "payload": payload,
    }
    line = json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())
    return event


def append_event(
    type_: str,
    payload: dict[str, Any],
    *,
    project: Optional[Project] = None,
) -> dict[str, Any]:
    """Append to the project log if a project is given, else the global log."""
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
    """Pure fold over the global event log → derived instance/endpoint state."""
    state: dict[str, Any] = {
        "instance_state": "unknown",
        "endpoint": None,
        "last_poll_ts": None,
        "errors": 0,
    }
    for ev in iter_events(GLOBAL_EVENTS_PATH):
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


def chat_stream(
    client: OpenAI,
    model: str,
    messages: list[dict[str, Any]],
    *,
    timeout_s: float = 60.0,
) -> str:
    """Stream a plain (no-tools) completion to stdout, return the joined text."""
    chunks: list[str] = []
    stream = client.chat.completions.create(
        model=model, messages=messages, stream=True, timeout=timeout_s
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            chunks.append(delta)
            console.out(delta, end="")
    console.out("")
    return "".join(chunks)


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


def _require_endpoint(cfg: Config) -> str:
    state = replay_global()
    endpoint = cfg.endpoint or state.get("endpoint")
    if not endpoint:
        raise MichaelError("no endpoint known — run `up` first")
    return endpoint


# ---------------------------------------------------------------------------
# Sandbox runner
# ---------------------------------------------------------------------------


def run_sandbox(
    code: str,
    *,
    network: bool = False,
    timeout_s: int = 30,
    cfg: Optional[Config] = None,
    project: Optional[Project] = None,
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

        # Read-only mount: a compromised payload can't mutate the host source.
        mount = f"{tmp}:/workspace:ro"
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
                project=project,
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
            project=project,
        )
        return cp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Header package — the brief sent to the LLM as a system message.
# Rebuilt fresh from the event log + filesystem on every user prompt.
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
        # Skip dotted dirs (.git, .venv, .vscode, ...) and the explicit list.
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

    text_files.sort(key=lambda x: x[1])  # smaller first → more files fit
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
    return out


def build_header(project: Project) -> str:
    """Build the system-prompt brief. Called fresh per user prompt."""
    prompts = _prompt_history_lines(project)
    actions = _action_log_lines(project)
    snap = filesystem_snapshot(pathlib.Path(project.path))

    return "\n".join([
        "You are a coding agent connected to the user's machine through Project Michael.",
        "",
        "Project Michael is a Python CLI app that:",
        "- Maintains a per-project event log of every user prompt and every tool",
        "  call you have executed in this project.",
        "- Rebuilds this brief from scratch on every user prompt — you have no",
        "  memory across user prompts. The log below is your memory.",
        "- Asks the user to confirm before any write or code execution.",
        "  read_file and list_dir auto-execute.",
        "",
        "Tools (full schemas in the API call):",
        "  write_file(path, content)       overwrite a file in the project",
        "  read_file(path)                 auto-executes",
        "  list_dir(path='.')              auto-executes",
        "  apply_patch(path, unified_diff) apply a unified diff to a file",
        "  run_in_sandbox(python_code)     isolated podman, no network",
        "  run_shell(cmd, timeout_s=60)    runs in the project workspace on host",
        "",
        "All paths are relative to the project root. Do not escape with '..'.",
        "",
        "=== Project ===",
        f"Name: {project.name}",
        f"Slug: {project.slug}",
        f"Root: {project.path}",
        "",
        "=== User's prompts in this project (verbatim, in order) ===",
        "\n".join(prompts) if prompts else "(this is the user's first prompt)",
        "",
        "=== Tool calls executed in this project (in order) ===",
        "\n".join(actions) if actions else "(none yet)",
        "",
        "=== Filesystem snapshot ===",
        snap,
    ])


# ---------------------------------------------------------------------------
# Tool definitions and execution
# ---------------------------------------------------------------------------


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Overwrite (or create) a file in the project workspace. "
                "Path is relative to the project root. Parent dirs are created. "
                "Requires user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to project root."},
                    "content": {"type": "string", "description": "Full file content."},
                },
                "required": ["path", "content"],
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
                "Apply a unified diff to a file in the project workspace. "
                "Requires user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "unified_diff": {"type": "string"},
                },
                "required": ["path", "unified_diff"],
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
                "Run a shell command in the project workspace on the host "
                "(NOT sandboxed). Requires user confirmation."
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


def execute_tool(
    name: str,
    args: dict[str, Any],
    project: Project,
    cfg: Config,
) -> str:
    if name == "write_file":
        target = _resolve_in_project(project, str(args["path"]))
        target.parent.mkdir(parents=True, exist_ok=True)
        content = str(args["content"])
        target.write_text(content)
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
        return f"ok; path={args['path']}; sha256={h}; size={len(content)}"

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

    if name == "apply_patch":
        target = _resolve_in_project(project, str(args["path"]))
        if not shutil.which("patch"):
            return "error: `patch` not installed on host (apt install patch)"
        diff = str(args["unified_diff"])
        cp = subprocess.run(
            ["patch", "--no-backup-if-mismatch", "-u", str(target)],
            input=diff, capture_output=True, text=True, timeout=30, check=False,
        )
        if cp.returncode != 0:
            return f"patch failed (rc={cp.returncode}): {(cp.stderr or '')[-500:]}"
        return f"ok; patched {target.relative_to(pathlib.Path(project.path).resolve())}"

    if name == "run_in_sandbox":
        cp = run_sandbox(
            str(args["python_code"]),
            network=False,
            timeout_s=30,
            cfg=cfg,
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
# Y/n/Edit confirmation for tool calls
# ---------------------------------------------------------------------------


def _render_for_confirmation(name: str, args: dict[str, Any], project: Project) -> tuple[str, str]:
    """Return (rendered_text, syntax_lexer)."""
    if name == "write_file":
        try:
            target = _resolve_in_project(project, str(args["path"]))
            old = target.read_text(errors="replace") if target.is_file() else ""
        except MichaelError:
            old = ""
        diff = "".join(difflib.unified_diff(
            old.splitlines(keepends=True),
            str(args["content"]).splitlines(keepends=True),
            fromfile=f"a/{args.get('path', '?')}",
            tofile=f"b/{args.get('path', '?')}",
        )) or "(no changes)"
        return diff, "diff"
    if name == "apply_patch":
        return f"patch target: {args.get('path', '?')}\n\n{args.get('unified_diff', '')}", "diff"
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
    """Returns (decision, possibly_edited_args). decision in {'yes', 'no'}."""
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
        Config().save()
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


def cmd_up() -> None:
    cfg = Config.load()
    if not cfg.vast_instance_id:
        raise MichaelError("VAST_INSTANCE_ID is not set (run `config`)")
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
            append_event("instance.poll", {"i": i + 1, "endpoint_known": bool(ep)})
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


def cmd_down() -> None:
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


def cmd_status() -> None:
    state = replay_global()
    active = get_active_project()
    table = Table(title="michael status", border_style="cyan")
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("active project", active.slug if active else "(none)")
    table.add_row("instance state", str(state["instance_state"]))
    table.add_row("endpoint", str(state["endpoint"]))
    table.add_row("last poll ts", str(state["last_poll_ts"]))
    table.add_row("errors (global)", str(state["errors"]))
    console.print(table)


def cmd_ask(prompt: str) -> None:
    project = require_active_project()
    cfg = Config.load()
    endpoint = _require_endpoint(cfg)
    client = llm_client(endpoint, cfg.vllm_api_key)
    append_event("prompt.sent", {"prompt": prompt, "model": cfg.model_name}, project=project)
    header = build_header(project)
    messages = [
        {"role": "system", "content": header},
        {"role": "user", "content": prompt},
    ]
    text = chat_stream(
        client,
        cfg.model_name,
        messages,
        timeout_s=float(cfg.request_timeout_s),
    )
    append_event("assistant.message", {"chars": len(text)}, project=project)


def cmd_run() -> None:
    project = require_active_project()
    cfg = Config.load()
    endpoint = _require_endpoint(cfg)
    client = llm_client(endpoint, cfg.vllm_api_key)

    console.print(
        f"[bold cyan]michael run[/] — project: [bold]{project.name}[/] "
        f"({project.slug}) — empty line or 'quit' to exit"
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

        append_event("prompt.sent", {"prompt": user, "model": cfg.model_name}, project=project)

        # Fresh brief per user prompt; tool messages flow within this loop only.
        header = build_header(project)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": header},
            {"role": "user", "content": user},
        ]

        for turn in range(MAX_AGENT_TURNS):
            try:
                resp = client.chat.completions.create(
                    model=cfg.model_name,
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    stream=False,
                    timeout=float(cfg.request_timeout_s),
                )
            except Exception as e:
                err.print(f"LLM error: {e}")
                append_event(
                    "error",
                    {"where": "agent_loop", "msg": str(e)},
                    project=project,
                )
                break

            msg = resp.choices[0].message
            if msg.content:
                console.print(
                    Panel(msg.content, title="assistant", border_style="green")
                )
                append_event(
                    "assistant.message",
                    {"chars": len(msg.content)},
                    project=project,
                )

            tool_calls = msg.tool_calls or []
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
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

            if not tool_calls:
                break

            for tc in tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name in AUTO_EXEC_TOOLS:
                    decision, final_args = "yes", args
                else:
                    try:
                        decision, final_args = confirm_tool_call(name, args, project)
                    except (KeyboardInterrupt, typer.Abort):
                        decision, final_args = "no", args

                if decision == "no":
                    result = "[user rejected this tool call]"
                    append_event(
                        "tool.rejected",
                        {"tool": name, "args": args, "summary": _summary_for(name, args)},
                        project=project,
                    )
                else:
                    try:
                        result = execute_tool(name, final_args, project, cfg)
                    except MichaelError as e:
                        result = f"error: {e}"
                    first_line = (result.splitlines()[0] if result else "ok")[:120]
                    append_event(
                        "tool.executed",
                        {
                            "tool": name,
                            "args": final_args,
                            "summary": f"{_summary_for(name, final_args)} → {first_line}",
                            "result_chars": len(result),
                        },
                        project=project,
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            err.print(f"hit max-turn limit ({MAX_AGENT_TURNS}); stopping")


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


def cmd_sandbox(file: pathlib.Path, net: bool = False, timeout: int = 30) -> None:
    cfg = Config.load()
    project = get_active_project()
    code = pathlib.Path(file).read_text()
    cp = run_sandbox(code, network=net, timeout_s=timeout, cfg=cfg, project=project)
    stdout_tail = "\n".join((cp.stdout or "").splitlines()[-80:])
    stderr_tail = "\n".join((cp.stderr or "").splitlines()[-40:])
    console.print(
        Panel(
            stdout_tail or "(empty)",
            title=f"stdout (rc={cp.returncode})",
            border_style="green",
        )
    )
    if stderr_tail:
        console.print(Panel(stderr_tail, title="stderr", border_style="red"))


# ---------------------------------------------------------------------------
# Typer commands (one-shot from the user's shell)
# ---------------------------------------------------------------------------


@app.command(name="show")
def show_cmd() -> None:
    """List projects."""
    cmd_show()


@app.command(name="new")
def new_cmd(
    keyword: Optional[str] = typer.Argument(None, help="'project' or the project name"),
    name: Optional[str] = typer.Argument(None, help="Project name (if 'project' was passed)"),
) -> None:
    """Create a new project. Usage: `new project [name]` or `new <name>`."""
    if keyword == "project":
        actual_name = name
    elif keyword and name is None:
        actual_name = keyword
    else:
        actual_name = name
    cmd_new(actual_name)


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
    """Open the global config file in $EDITOR."""
    cmd_config()


@app.command(name="up")
def up_cmd() -> None:
    """Resume the Vast.ai instance and wait for vLLM."""
    cmd_up()


@app.command(name="down")
def down_cmd() -> None:
    """Pause the Vast.ai instance."""
    cmd_down()


@app.command(name="status")
def status_cmd() -> None:
    """Show derived state."""
    cmd_status()


@app.command(name="ask")
def ask_cmd(prompt: str = typer.Argument(..., help="One-shot prompt for the LLM.")) -> None:
    """One-shot LLM call inside the active project (no tool calls)."""
    cmd_ask(prompt)


@app.command(name="run")
def run_cmd() -> None:
    """Multi-turn tool-calling agent loop in the active project."""
    cmd_run()


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
    """Run a Python file in the throwaway sandbox."""
    cmd_sandbox(file, net, timeout)
    sys.exit(0)


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


REPL_COMMANDS = {
    "show", "new", "use", "current", "config",
    "up", "down", "status",
    "ask", "run", "log", "sandbox",
    "quit", "exit", "help",
}


def repl() -> None:
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)
    session = PromptSession(
        history=FileHistory(str(REPL_HISTORY_PATH)),
        auto_suggest=AutoSuggestFromHistory(),
    )
    console.print("hey")
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
    elif cmd == "new":
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
        cmd_up()
    elif cmd == "down":
        cmd_down()
    elif cmd == "status":
        cmd_status()
    elif cmd == "ask":
        if not rest:
            err.print("usage: ask <prompt>")
            return
        cmd_ask(" ".join(rest))
    elif cmd == "run":
        cmd_run()
    elif cmd == "log":
        n = 20
        if "--tail" in rest:
            i = rest.index("--tail")
            if i + 1 < len(rest):
                try:
                    n = int(rest[i + 1])
                except ValueError:
                    pass
        elif "-n" in rest:
            i = rest.index("-n")
            if i + 1 < len(rest):
                try:
                    n = int(rest[i + 1])
                except ValueError:
                    pass
        cmd_log(n)
    elif cmd == "sandbox":
        if not rest:
            err.print("usage: sandbox <file>")
            return
        cmd_sandbox(pathlib.Path(rest[0]))
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
