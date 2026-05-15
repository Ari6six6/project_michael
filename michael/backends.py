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
from michael.config import Config, GpuConfig, ModelProfile, SandboxConfig, VpsConfig

if TYPE_CHECKING:
    from michael.project import Project


# ---------------------------------------------------------------------------
# Event helpers (lazy import to avoid circular dependency)
# ---------------------------------------------------------------------------


def append_event(event_type: str, payload: dict, *, project=None) -> None:
    from michael.project import append_event as _ae
    _ae(event_type, payload, project=project)


# ---------------------------------------------------------------------------
# VPS SSH helpers
# ---------------------------------------------------------------------------


def _ssh_argv(vps: VpsConfig) -> list[str]:
    args = [
        "ssh",
        "-o", f"ControlMaster=auto",
        "-o", f"ControlPath={G.STATE_DIR}/ssh-%r@%h:%p.sock",
        "-o", f"ControlPersist={vps.control_persist}",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-i", os.path.expanduser(vps.ssh_key_path),
        "-p", str(vps.port),
        f"{vps.user}@{vps.host}",
    ]
    return args


def _ssh_preflight(cfg: Config) -> None:
    if not cfg.vps_active():
        return


# ---------------------------------------------------------------------------
# GPU SSH helpers
# ---------------------------------------------------------------------------


def parse_vast_ssh_cmd(ssh_str: str) -> tuple[str, str, int]:
    """Parse a Vast.ai SSH string and return (user, host, port)."""
    try:
        tokens = shlex.split(ssh_str)
    except ValueError as e:
        raise G.MichaelError(f"could not parse SSH command: {e}") from e

    tokens = [t for t in tokens if t != "ssh"]
    port = 22
    user_host: Optional[str] = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-p" and i + 1 < len(tokens):
            try:
                port = int(tokens[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if tok.startswith("-p") and len(tok) > 2:
            try:
                port = int(tok[2:])
            except ValueError:
                pass
            i += 1
            continue
        if tok.startswith("-") and len(tok) == 2:
            i += 2
            continue
        if tok.startswith("-"):
            i += 1
            continue
        if "@" in tok:
            user_host = tok
        i += 1

    if not user_host:
        raise G.MichaelError(
            f"could not find user@host in SSH command: {ssh_str!r}"
        )
    parts = user_host.split("@", 1)
    return parts[0], parts[1], port


def _gpu_ssh_argv(gpu: GpuConfig) -> list[str]:
    return [
        "ssh",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-i", os.path.expanduser(gpu.ssh_key_path),
        "-p", str(gpu.ssh_port),
        f"{gpu.ssh_user}@{gpu.ssh_host}",
    ]


def _gpu_ssh_run(
    gpu: GpuConfig, cmd: str, *, timeout: int = 30
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            _gpu_ssh_argv(gpu) + [cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise G.MichaelError(f"GPU command timed out after {timeout}s")


def _gpu_ssh_stream(gpu: GpuConfig, cmd: str, *, timeout: int = 600) -> int:
    try:
        proc = subprocess.Popen(
            _gpu_ssh_argv(gpu) + [cmd],
            stdout=None,
            stderr=None,
        )
        proc.wait(timeout=timeout)
        return proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        raise G.MichaelError(f"GPU stream command timed out after {timeout}s")


def gpu_port_forward_cmd(gpu: GpuConfig) -> str:
    key = os.path.expanduser(gpu.ssh_key_path)
    return (
        f"ssh -p {gpu.ssh_port} {gpu.ssh_user}@{gpu.ssh_host} "
        f"-L {gpu.vllm_port}:localhost:{gpu.vllm_port} "
        f"-N -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -i {key}"
    )


def _build_vllm_cmd(gpu: GpuConfig) -> str:
    """Build the nohup vllm serve command for the GPU. Single source of truth."""
    api_key_flag = f"--api-key {gpu.vllm_api_key} " if gpu.vllm_api_key else ""
    return (
        f"nohup vllm serve {gpu.model_repo} "
        f"--host 0.0.0.0 --port {gpu.vllm_port} "
        f"--dtype auto --gpu-memory-utilization 0.95 "
        f"--max-model-len 8192 "
        f"--enable-auto-tool-choice --tool-call-parser hermes "
        f"{api_key_flag}"
        f"> /tmp/vllm.log 2>&1 & echo $!"
    )


def _restart_vllm_on_gpu(gpu: GpuConfig, *, poll_timeout_s: int = 600) -> None:
    """Kill stale vLLM on the GPU and restart with correct flags; poll until ready."""
    G.console.print("[yellow]restarting vLLM on GPU...[/]")
    _gpu_ssh_run(gpu, "pkill -f 'vllm serve' 2>/dev/null || true", timeout=10)
    time.sleep(3)
    cp = _gpu_ssh_run(gpu, _build_vllm_cmd(gpu), timeout=30)
    G.console.print(f"[cyan]vLLM restarting[/] (PID {cp.stdout.strip()})")
    elapsed = 0
    while elapsed < poll_timeout_s:
        time.sleep(15)
        elapsed += 15
        cp = _gpu_ssh_run(
            gpu,
            f"curl -sf http://localhost:{gpu.vllm_port}/v1/models > /dev/null 2>&1 "
            f"&& echo ready || echo down",
            timeout=15,
        )
        if "ready" in cp.stdout:
            G.console.print("[green]vLLM is ready[/]")
            return
        tail = _gpu_ssh_run(
            gpu, "tail -1 /tmp/vllm.log 2>/dev/null || true", timeout=10
        ).stdout.strip()
        G.console.print(f"[dim]· {elapsed}s — {tail or 'loading...'}[/]")
    raise G.MichaelError(f"vLLM did not come up within {poll_timeout_s}s — check /tmp/vllm.log")


_tunnel_proc: Optional[subprocess.Popen] = None  # type: ignore[type-arg]


def _close_tunnel() -> None:
    global _tunnel_proc
    if _tunnel_proc is not None:
        try:
            _tunnel_proc.terminate()
        except Exception:
            pass
        _tunnel_proc = None


def _ensure_tunnel(gpu: GpuConfig) -> None:
    """Auto-spawn SSH port-forward if the vLLM endpoint is not locally reachable.

    Safe to call on every `michael run` — if the tunnel is already up (user-managed
    or from a previous auto-start) it returns immediately. When Termux is killed and
    restarted, this re-establishes the tunnel without any manual step.
    """
    global _tunnel_proc
    if not gpu.ssh_host:
        return
    endpoint = f"http://localhost:{gpu.vllm_port}/v1"
    if _ping_vllm(endpoint, gpu.vllm_api_key):
        return  # already reachable — nothing to do
    G.console.print("[yellow]tunnel not detected — starting SSH port-forward...[/]")
    key = os.path.expanduser(gpu.ssh_key_path)
    _tunnel_proc = subprocess.Popen(
        [
            "ssh", "-p", str(gpu.ssh_port), f"{gpu.ssh_user}@{gpu.ssh_host}",
            "-L", f"{gpu.vllm_port}:localhost:{gpu.vllm_port}",
            "-N",
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "ServerAliveInterval=30",
            "-o", "ServerAliveCountMax=3",
            "-i", key,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    atexit.register(_close_tunnel)
    for _ in range(15):
        time.sleep(2)
        if _ping_vllm(endpoint, gpu.vllm_api_key):
            G.console.print("[green]tunnel up[/]")
            return
    _close_tunnel()
    raise G.MichaelError(
        "SSH tunnel failed to come up after 30s — check gpu.ssh_host / ssh_key_path in config"
    )


# ---------------------------------------------------------------------------
# Vast.ai API client
# ---------------------------------------------------------------------------


class VastClient:
    base = "https://console.vast.ai/api/v0"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise G.MichaelError("vast_api_key is not set")
        self._client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30,
        )

    def close(self) -> None:
        self._client.close()

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

    def list(self) -> list[dict[str, Any]]:
        data = self._wrap("list", lambda: self._client.get(f"{self.base}/instances/"))
        return data.get("instances", []) or []

    def endpoint_for(self, inst_id: str | int, port: int) -> Optional[str]:
        info = self.get(inst_id)
        if not info:
            return None
        ip = info.get("public_ipaddr") or info.get("ssh_host")
        if not ip:
            return None
        return f"http://{ip}:{port}/v1"


# ---------------------------------------------------------------------------
# LLM client
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _ToolCall:
    id: str
    name: str
    arguments: str


@dataclasses.dataclass
class _Choice:
    content: Optional[str]
    tool_calls: Optional[list[_ToolCall]]
    finish_reason: Optional[str]


@dataclasses.dataclass
class _CompletionResponse:
    choices: list[_Choice]
    usage: Optional[dict]


class _Completions:
    def __init__(
        self, endpoint: str, http: httpx.Client, headers: dict, enable_thinking: bool = False
    ) -> None:
        self._endpoint = endpoint
        self._http = http
        self._headers = headers
        self._enable_thinking = enable_thinking

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
        if self._enable_thinking and not any(m.get("role") == "tool" for m in messages):
            body["chat_template_kwargs"] = {"enable_thinking": True}
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
        try:
            data = r.json()
        except Exception as exc:
            raise G.MichaelError(
                f"vLLM returned non-JSON response: {exc} — body: {r.text[:200]!r}"
            ) from exc
        return self._parse_response(data)

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
                    name=tc.get("function", {}).get("name", ""),
                    arguments=tc.get("function", {}).get("arguments", ""),
                )
                for tc in tcs
            ] or None
            choices.append(
                _Choice(
                    content=m.get("content"),
                    tool_calls=tool_calls,
                    finish_reason=c.get("finish_reason"),
                )
            )
        return _CompletionResponse(choices=choices, usage=data.get("usage"))

    def _parse_chunk(self, data: dict) -> _Choice:
        c = data.get("choices", [{}])[0]
        delta = c.get("delta", {})
        tcs = delta.get("tool_calls") or []
        tool_calls: Optional[list] = [
            _ToolCall(
                id=tc.get("id", ""),
                name=(tc.get("function") or {}).get("name", ""),
                arguments=(tc.get("function") or {}).get("arguments", ""),
            )
            for tc in tcs
        ] or None
        return _Choice(
            content=delta.get("content"),
            tool_calls=tool_calls,
            finish_reason=c.get("finish_reason"),
        )


class _ChatCompletions:
    def __init__(self, completions: _Completions) -> None:
        self.completions = completions


class LLMClient:
    def __init__(self, endpoint: str, api_key: str = "", enable_thinking: bool = False) -> None:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._http = httpx.Client(headers=headers, timeout=120)
        _completions = _Completions(endpoint, self._http, headers, enable_thinking)
        self.chat = _ChatCompletions(_completions)

    def close(self) -> None:
        self._http.close()


def llm_client(endpoint: str, api_key: str = "", enable_thinking: bool = False) -> LLMClient:
    return LLMClient(endpoint, api_key, enable_thinking)


def _require_endpoint(profile: ModelProfile, name: str) -> str:
    if not profile.endpoint:
        raise G.MichaelError(
            f"model '{name}' has no endpoint — run `michael up` or `michael gpu up` first"
        )
    return profile.endpoint


def _ping_vllm(endpoint: str, api_key: str = "", *, timeout_s: float = 5.0) -> bool:
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        r = httpx.get(f"{endpoint}/models", headers=headers, timeout=timeout_s)
        return r.status_code == 200
    except Exception:
        return False


def chat_stream(
    client: LLMClient,
    model: str,
    messages: list,
    *,
    timeout_s: float = 120.0,
) -> tuple[str, Optional[dict]]:
    chunks: list[str] = []
    usage: Optional[dict] = None
    for chunk in client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        timeout=timeout_s,
        stream_options={"include_usage": True},
    ):
        if chunk.content:
            chunks.append(chunk.content)
    text = "".join(chunks)
    return text, usage


# ---------------------------------------------------------------------------
# Sandbox backends
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SandboxResult:
    stdout: str
    stderr: str
    returncode: int


class SandboxBackend(ABC):
    @abstractmethod
    def run(
        self,
        code: str,
        *,
        network: bool = False,
        timeout_s: int = 30,
        project=None,
    ) -> SandboxResult:
        ...


class LocalPodmanBackend(SandboxBackend):
    def __init__(self, cfg: SandboxConfig) -> None:
        self._cfg = cfg

    def run(
        self,
        code: str,
        *,
        network: bool = False,
        timeout_s: int = 30,
        project=None,
    ) -> SandboxResult:
        cfg = self._cfg
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write(code)
            tmp = f.name
        try:
            cmd = [
                "podman", "run", "--rm",
                "--memory", f"{cfg.memory_mb}m",
                "--cpus", str(cfg.cpus),
                "--pids-limit", str(cfg.pids),
                "--network", "bridge" if network else "none",
                "-v", f"{tmp}:/sandbox/script.py:ro",
                cfg.image,
                "python3", "/sandbox/script.py",
            ]
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_s,
                    check=False,
                )
                return SandboxResult(
                    stdout=result.stdout,
                    stderr=result.stderr,
                    returncode=result.returncode,
                )
            except subprocess.TimeoutExpired:
                raise G.MichaelError(f"sandbox timed out after {timeout_s}s") from None
        finally:
            pathlib.Path(tmp).unlink(missing_ok=True)


class RemotePodmanBackend(SandboxBackend):
    def __init__(self, cfg: Config) -> None:
        self._cfg = cfg

    def _stage(self, code: str, remote_path: str) -> None:
        cfg = self._cfg
        vps = cfg.vps
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False
        ) as f:
            f.write(code)
            tmp = f.name
        try:
            scp_cmd = [
                "scp",
                "-o", f"ControlMaster=auto",
                "-o", f"ControlPath={G.STATE_DIR}/ssh-%r@%h:%p.sock",
                "-o", f"ControlPersist={vps.control_persist}",
                "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-i", os.path.expanduser(vps.ssh_key_path),
                "-P", str(vps.port),
                tmp,
                f"{vps.user}@{vps.host}:{remote_path}",
            ]
            cp_stage = subprocess.run(
                scp_cmd, capture_output=True, text=True, timeout=30, check=False
            )
            if cp_stage.returncode != 0:
                raise G.MichaelError(f"remote stage failed: {cp_stage.stderr[:200]}")
        finally:
            pathlib.Path(tmp).unlink(missing_ok=True)

    def run(
        self,
        code: str,
        *,
        network: bool = False,
        timeout_s: int = 30,
        project=None,
    ) -> SandboxResult:
        cfg = self._cfg
        vps = cfg.vps
        sb = cfg.sandbox
        run_id = uuid.uuid4().hex[:8]
        remote_script = f"{vps.workspace_dir}/run_{run_id}.py"
        self._stage(code, remote_script)
        net_flag = "bridge" if network else "none"
        podman_cmd = (
            f"podman run --rm "
            f"--memory {sb.memory_mb}m "
            f"--cpus {sb.cpus} "
            f"--pids-limit {sb.pids} "
            f"--network {net_flag} "
            f"-v {remote_script}:/sandbox/script.py:ro "
            f"{sb.image} python3 /sandbox/script.py"
        )
        try:
            cp = subprocess.run(
                _ssh_argv(vps) + [podman_cmd],
                capture_output=True,
                text=True,
                timeout=timeout_s + 15,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise G.MichaelError(f"sandbox timed out after {timeout_s}s") from None
        cleanup_cmd = f"rm -f {remote_script}"
        subprocess.run(
            _ssh_argv(vps) + [cleanup_cmd],
            capture_output=True, timeout=10, check=False,
        )
        return SandboxResult(
            stdout=cp.stdout,
            stderr=cp.stderr,
            returncode=cp.returncode,
        )


def make_backend(cfg: Config) -> SandboxBackend:
    if cfg.vps_active():
        return RemotePodmanBackend(cfg)
    return LocalPodmanBackend(cfg.sandbox)
