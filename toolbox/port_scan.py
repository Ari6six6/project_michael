"""Global tool: port_scan — active TCP port scan (requires authorization declaration)."""
from __future__ import annotations
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "port_scan",
        "description": (
            "Active TCP port scan against a target. REQUIRES authorized_by declaration — "
            "will refuse to run without it. Uses nmap if available (service/version detection), "
            "falls back to Python socket connect scan. "
            "authorized_by must state: 'I own this system', 'bug_bounty:<program>', or 'pentest:<client>'. "
            "The user must explicitly declare this is their system or an authorized target."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "IP address or hostname.",
                },
                "ports": {
                    "type": "string",
                    "description": "Port spec: '80', '1-1000', '22,80,443', '1-65535'. Default: top 1000.",
                },
                "timing": {
                    "type": "string",
                    "description": "Nmap timing template: T1 (slow) to T5 (insane). Default T3.",
                },
                "authorized_by": {
                    "type": "string",
                    "description": (
                        "REQUIRED. Your authorization declaration. Examples: "
                        "'I own this system', 'bug_bounty:HackerOne program name', "
                        "'pentest:Client Corp Ltd'."
                    ),
                },
            },
            "required": ["target", "authorized_by"],
        },
    },
}

_BLOCKED = (
    "BLOCKED: port_scan requires an explicit authorization declaration.\n\n"
    "Set authorized_by to one of:\n"
    "  'I own this system'\n"
    "  'bug_bounty:<program name>'\n"
    "  'pentest:<client name>'\n\n"
    "Without a declaration this tool will not run. "
    "The user must explicitly state the target is theirs or an authorized engagement."
)


def port_scan(
    target: str,
    ports: str = "1-1000",
    timing: str = "T3",
    authorized_by: str = "",
    **kwargs: Any,
) -> str:
    if not authorized_by or not authorized_by.strip():
        return _BLOCKED

    import subprocess
    import socket

    target = target.strip()
    auth_banner = f"[authorized: {authorized_by}]"

    # Try nmap first — gives service/version info
    try:
        result = subprocess.run(
            ["nmap", f"-{timing}", "--open", "-sV", "-p", ports, target],
            capture_output=True, text=True, timeout=180, check=False,
        )
        out = result.stdout or result.stderr
        return f"{auth_banner}\n\n{out.strip()}"
    except FileNotFoundError:
        pass  # nmap not installed, fall back

    # Python socket connect scan fallback
    try:
        if "-" in ports and "," not in ports:
            start, end = ports.split("-", 1)
            port_list = list(range(int(start), min(int(end) + 1, 65536)))
        elif "," in ports:
            port_list = [int(p.strip()) for p in ports.split(",")]
        else:
            port_list = [int(ports.strip())]
    except ValueError:
        return f"error: invalid port spec {ports!r}"

    open_ports: list[tuple[int, str]] = []
    for port in port_list:
        try:
            with socket.create_connection((target, port), timeout=0.5) as s:
                # Grab banner if available
                try:
                    s.settimeout(1.0)
                    banner = s.recv(256).decode("utf-8", errors="replace").strip()[:80]
                except Exception:
                    banner = ""
                open_ports.append((port, banner))
        except (socket.timeout, ConnectionRefusedError, OSError):
            pass

    if not open_ports:
        return f"{auth_banner}\nno open ports found on {target} in range {ports}"

    lines = [f"{auth_banner}", f"open ports on {target} ({len(open_ports)} found):"]
    for port, banner in open_ports:
        b = f"  [{banner}]" if banner else ""
        lines.append(f"  {port}/tcp  OPEN{b}")
    return "\n".join(lines)
