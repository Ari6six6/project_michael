"""Global tool: whois_lookup — passive WHOIS query for a domain or IP."""
from __future__ import annotations
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "whois_lookup",
        "description": (
            "WHOIS lookup for a domain or IP address. Returns registrar, creation/expiry dates, "
            "nameservers, registrant info, and registrar abuse contact. Passive — no active probing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Domain name or IP address to look up.",
                }
            },
            "required": ["target"],
        },
    },
}


def whois_lookup(target: str, **kwargs: Any) -> str:
    import subprocess
    target = target.strip()
    if not target:
        return "error: target is required"
    try:
        r = subprocess.run(
            ["whois", target],
            capture_output=True, text=True, timeout=20, check=False,
        )
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        if not out and err:
            return f"whois error: {err[:500]}"
        return out[:4000] or "(no output)"
    except FileNotFoundError:
        return "error: whois not installed (apt install whois / brew install whois)"
    except subprocess.TimeoutExpired:
        return "error: whois timed out after 20s"
