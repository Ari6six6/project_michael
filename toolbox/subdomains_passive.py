"""Global tool: subdomains_passive — passive subdomain enumeration from public sources."""
from __future__ import annotations
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "subdomains_passive",
        "description": (
            "Passive subdomain enumeration from certificate transparency logs (crt.sh) "
            "and DNS-based sources. No active probing — entirely read-only queries to public APIs. "
            "Returns unique subdomain list sorted alphabetically."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Apex domain (e.g. example.com).",
                },
            },
            "required": ["domain"],
        },
    },
}


def subdomains_passive(domain: str, **kwargs: Any) -> str:
    import httpx
    domain = domain.strip().lstrip("*.").rstrip(".")
    subs: set[str] = set()
    errors: list[str] = []

    # Source 1: crt.sh certificate transparency
    try:
        r = httpx.get(
            f"https://crt.sh/?q=%.{domain}&output=json",
            timeout=25,
            follow_redirects=True,
        )
        r.raise_for_status()
        for entry in r.json():
            for name in (entry.get("name_value") or "").splitlines():
                name = name.strip().lstrip("*.")
                if name and (name == domain or name.endswith(f".{domain}")):
                    subs.add(name)
    except Exception as e:
        errors.append(f"crt.sh: {e}")

    # Source 2: Hackertarget (free tier, no key required)
    try:
        r = httpx.get(
            f"https://api.hackertarget.com/hostsearch/?q={domain}",
            timeout=15,
            follow_redirects=True,
        )
        for line in r.text.splitlines():
            if "," in line:
                sub = line.split(",")[0].strip()
                if sub and sub.endswith(domain):
                    subs.add(sub)
    except Exception as e:
        errors.append(f"hackertarget: {e}")

    sorted_subs = sorted(subs)
    lines = [f"passive subdomains for {domain}: {len(sorted_subs)} unique"]
    if errors:
        lines.append(f"source errors: {'; '.join(errors)}")
    lines.extend(f"  {s}" for s in sorted_subs[:400])
    if len(sorted_subs) > 400:
        lines.append(f"  ... ({len(sorted_subs) - 400} more truncated)")
    return "\n".join(lines)
