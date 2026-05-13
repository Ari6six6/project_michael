"""SSH helpers, Vast.ai client, LLM client, and sandbox backends."""
from __future__ import annotations

import atexit
import dataclasses
import json as _json
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

import httpx

import michael.globals as G
from michael.config import Config, ModelProfile, SandboxConfig, VpsConfig
from michael.project import append_event, replay_global

if TYPE_CHECKING:
    from michael.project import Project


# ---------------------------------------------------------------------------
# SSH helpers (ControlMaster multiplexing)
# ---------------------------------------------------------------------------


def _ssh_argv(vps: VpsConfig) -> list[str]:
    sock = pathlib.Path(tempfile.gettempdir()) / "ssh-%C.sock"
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
    sock = pathlib.Path(tempfile.gettempdir()) / "ssh-%C.sock"
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
        raise G.MichaelError(
            f"VPS unreachable ({cfg.vps.user}@{cfg.vps.host}): {cp.stderr.strip()[:200]}"
        )
    append_event("ssh.health", {"host": cfg.vps.host, "ok": True})
    atexit.register(_ssh_close_master, cfg.vps)


# ---------------------------------------------------------------------------
# Vast.ai client (plain httpx)
# ---------------------------------------------------------------------------


class VastClient:
    def __init__(self, api_key: str, base: str = "https://console.vast.ai/api/v0") -> None:
        if not api_key:
            raise G.MichaelError("vast_api_key is not set")
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
            raise G.MichaelError(msg) from e
        except httpx.HTTPError as e:
            msg = f"vast {fn_name}: {e}"
            append_event("error", {"where": fn_name, "msg": msg})
            raise G.MichaelError(msg) from e

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
        """Return f"http://{ip}:{HostPort}/v1" or None when the mapping isn't ready."""
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
# LLM client — minimal OpenAI-protocol implementation using plain httpx.
# The openai SDK requires jiter which requires Rust; Termux can't build it.
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Function:
    name: str
    arguments: str


@dataclasses.dataclass
class _ToolCall:
    id: str
    type: str
    function: _Function


@dataclasses.dataclass
class _Delta:
    content: Optional[str] = None


@dataclasses.dataclass
class _Message:
    content: Optional[str] = None
    tool_calls: Optional[list] = None  # list[_ToolCall]


@dataclasses.dataclass
class _Choice:
    message: _Message
    index: int = 0


@dataclasses.dataclass
class _StreamChoice:
    delta: _Delta
    index: int = 0


@dataclasses.dataclass
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def model_dump(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclasses.dataclass
class _CompletionResponse:
    choices: list  # list[_Choice]
    usage: Optional[_Usage] = None


@dataclasses.dataclass
class _StreamChunk:
    choices: list  # list[_StreamChoice]
    usage: Optional[_Usage] = None


class _Completions:
    def __init__(self, endpoint: str, http: httpx.Client, headers: dict) -> None:
        self._endpoint = endpoint
        self._http = http
        self._headers = headers

    def create(
        self,
        *,
        model: str,
        messages: list,
        tools: Optional[list] = None,
        tool_choice: Optional[Any] = None,
        stream: bool = False,
        timeout: float = 60.0,
        stream_options: Optional[dict] = None,
        **_kw: Any,
    ) -> Any:
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        if stream and stream_options:
            body["stream_options"] = stream_options
        if stream:
            return self._stream_iter(body, timeout)
        r = self._http.post(
            f"{self._endpoint}/chat/completions",
            json=body,
            timeout=timeout,
        )
        r.raise_for_status()
        return self._parse_response(r.json())

    def _stream_iter(self, body: dict, timeout: float):
        client = httpx.Client(
            headers=self._headers,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
        try:
            with client.stream(
                "POST", f"{self._endpoint}/chat/completions", json=body
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            yield self._parse_chunk(_json.loads(line[6:]))
                        except (_json.JSONDecodeError, KeyError):
                            continue
        finally:
            client.close()

    def _parse_response(self, data: dict) -> _CompletionResponse:
        choices = []
        for c in data.get("choices", []):
            m = c.get("message", {})
            tcs = m.get("tool_calls") or []
            tool_calls: Optional[list] = [
                _ToolCall(
                    id=tc.get("id", ""),
                    type=tc.get("type", "function"),
                    function=_Function(
                        name=tc.get("function", {}).get("name", ""),
                        arguments=tc.get("function", {}).get("arguments", "{}"),
                    ),
                )
                for tc in tcs
            ] or None
            choices.append(
                _Choice(
                    message=_Message(content=m.get("content"), tool_calls=tool_calls),
                    index=c.get("index", 0),
                )
            )
        u = data.get("usage")
        usage = (
            _Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            )
            if u
            else None
        )
        return _CompletionResponse(choices=choices, usage=usage)

    def _parse_chunk(self, data: dict) -> _StreamChunk:
        choices = [
            _StreamChoice(
                delta=_Delta(content=c.get("delta", {}).get("content")),
                index=c.get("index", 0),
            )
            for c in data.get("choices", [])
        ]
        u = data.get("usage")
        usage = (
            _Usage(
                prompt_tokens=u.get("prompt_tokens", 0),
                completion_tokens=u.get("completion_tokens", 0),
                total_tokens=u.get("total_tokens", 0),
            )
            if u
            else None
        )
        return _StreamChunk(choices=choices, usage=usage)


class _Chat:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class LLMClient:
    """Minimal OpenAI-protocol HTTP client using plain httpx. No Rust/jiter required."""

    def __init__(self, endpoint: str, api_key: Optional[str]) -> None:
        self._endpoint = endpoint.rstrip("/")
        key = api_key or "EMPTY"
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        self._http = httpx.Client(
            timeout=httpx.Timeout(30.0, read=300.0),
            headers=self._headers,
        )
        self.chat = _Chat(_Completions(self._endpoint, self._http, self._headers))

    def close(self) -> None:
        self._http.close()


def llm_client(endpoint: str, api_key: Optional[str]) -> LLMClient:
    """Return an httpx-based LLM client speaking the OpenAI REST protocol."""
    return LLMClient(endpoint, api_key)


def _usage_dict(u: Any) -> dict[str, Any]:
    if u is None:
        return {}
    try:
        return u.model_dump()
    except AttributeError:
        return {
            "prompt_tokens": getattr(u, "prompt_tokens", 0),
            "completion_tokens": getattr(u, "completion_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", 0),
        }


def chat_stream(
    client: LLMClient,
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
                G.console.out(delta, end="")
        chunk_usage = getattr(chunk, "usage", None)
        if chunk_usage is not None:
            usage = _usage_dict(chunk_usage)
    G.console.out("")
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
        raise G.MichaelError(
            f"no endpoint known for {profile_name!r} — run `up --model {profile_name}` first"
        )
    return endpoint


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
        self, code: str, *, network: bool = False, timeout_s: int = 30,
        project: Optional["Project"] = None,
    ) -> subprocess.CompletedProcess: ...


class DisabledSandboxBackend(SandboxBackend):
    def run(self, code, *, network=False, timeout_s=30, project=None):
        raise G.MichaelError(
            "sandbox unavailable: configure vps.host (recommended on phone) "
            "or install podman/docker locally"
        )


class LocalPodmanBackend(SandboxBackend):
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.runtime = "podman" if shutil.which("podman") else "docker"

    def run(
        self, code: str, *, network: bool = False, timeout_s: int = 30,
        project: Optional["Project"] = None,
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
                raise G.MichaelError(f"sandbox timed out after {timeout_s}s") from e

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
            raise G.MichaelError("vps.host required for remote sandbox")
        self.cfg = cfg
        self.vps = cfg.vps

    def run(
        self, code: str, *, network: bool = False, timeout_s: int = 30,
        project: Optional["Project"] = None,
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
            raise G.MichaelError(f"remote stage failed: {cp_stage.stderr[:200]}")

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
            raise G.MichaelError(f"sandbox timed out after {timeout_s}s") from e
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
