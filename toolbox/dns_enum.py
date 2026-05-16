"""Global tool: dns_enum — passive DNS record enumeration."""
from __future__ import annotations
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "dns_enum",
        "description": (
            "DNS record enumeration for a domain. Queries A, AAAA, MX, NS, TXT, CNAME, SOA, "
            "and optionally SRV records. Reveals mail servers, SPF/DKIM/DMARC policies, "
            "name servers, and delegations. Passive — uses system DNS resolver."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Domain to enumerate.",
                },
                "record_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Record types to query. Default: A AAAA MX NS TXT CNAME SOA.",
                },
            },
            "required": ["domain"],
        },
    },
}

_DEFAULT_TYPES = ["A", "AAAA", "MX", "NS", "TXT", "CNAME", "SOA"]


def dns_enum(domain: str, record_types: list[str] | None = None, **kwargs: Any) -> str:
    import subprocess
    domain = domain.strip().rstrip(".")
    types = record_types or _DEFAULT_TYPES
    sections: list[str] = [f"DNS enumeration: {domain}"]

    for rtype in types:
        try:
            r = subprocess.run(
                ["dig", "+short", "+time=5", "+tries=2", rtype, domain],
                capture_output=True, text=True, timeout=12, check=False,
            )
            out = r.stdout.strip()
            if out:
                sections.append(f"\n{rtype}:")
                for line in out.splitlines():
                    sections.append(f"  {line}")
        except FileNotFoundError:
            # dig not available — fall back to nslookup
            try:
                r = subprocess.run(
                    ["nslookup", "-type=" + rtype, domain],
                    capture_output=True, text=True, timeout=12, check=False,
                )
                out = r.stdout.strip()
                if out and "NXDOMAIN" not in out:
                    sections.append(f"\n{rtype} (nslookup):\n  " + "\n  ".join(out.splitlines()[:10]))
            except FileNotFoundError:
                sections.append("\nerror: neither dig nor nslookup found")
                break
        except subprocess.TimeoutExpired:
            sections.append(f"\n{rtype}: timed out")

    return "\n".join(sections)
