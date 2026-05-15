"""Global tool: target_define — structured target definition and recon agenda."""
from __future__ import annotations
from typing import Any
import json

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "target_define",
        "description": (
            "Define a structured reconnaissance target. Stores the target record in the project "
            "and returns a recommended recon sequence tailored to the authorization level. "
            "Authorization levels: 'passive_only' | 'own' | 'bug_bounty:<program>' | 'pentest:<client>'. "
            "Use this as Room 1's first call when starting any recon project."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short label for this target (e.g. 'acme-corp').",
                },
                "domain": {
                    "type": "string",
                    "description": "Primary domain or IP address.",
                },
                "authorization": {
                    "type": "string",
                    "description": (
                        "Authorization context. One of: "
                        "'passive_only' — no active scanning; "
                        "'own' — this is my system; "
                        "'bug_bounty:<program name>' — in-scope bug bounty; "
                        "'pentest:<client name>' — authorized engagement."
                    ),
                },
                "scope": {
                    "type": "string",
                    "description": "In-scope assets, comma-separated. Default: primary domain only.",
                },
                "notes": {
                    "type": "string",
                    "description": "Any additional context about this target.",
                },
            },
            "required": ["name", "domain", "authorization"],
        },
    },
}


def target_define(
    name: str,
    domain: str,
    authorization: str,
    scope: str = "",
    notes: str = "",
    **kwargs: Any,
) -> str:
    domain = domain.strip().rstrip("/")
    auth = authorization.strip().lower()
    active_ok = auth.startswith(("own", "bug_bounty:", "pentest:"))

    record = {
        "name": name,
        "domain": domain,
        "scope": scope or domain,
        "authorization": authorization,
        "active_scanning_authorized": active_ok,
        "notes": notes,
    }

    if active_ok:
        auth_tag = f"authorized_by='{authorization}: {domain}'"
    else:
        auth_tag = None

    seq = [
        f"Target defined: {name} ({domain})",
        f"Authorization:  {authorization}",
        f"Active scanning: {'YES' if active_ok else 'NO (passive_only)'}",
        "",
        "Recommended recon sequence:",
        f"  1. dns_enum(domain='{domain}')                          — DNS landscape",
        f"  2. cert_transparency(domain='{domain}')                 — subdomain discovery",
        f"  3. subdomains_passive(domain='{domain}')                — multi-source subdomains",
        f"  4. ssl_inspect(host='{domain}')                         — TLS cert + SANs",
        f"  5. http_fingerprint(url='https://{domain}')             — tech stack + headers",
        f"  6. whois_lookup(target='{domain}')                      — registration + contacts",
    ]

    if active_ok:
        seq += [
            "",
            "Active phase (authorization confirmed):",
            f"  7. port_scan(target='{domain}', ports='1-65535', {auth_tag})",
            f"  8. dir_enum(url='https://{domain}', {auth_tag})",
            f"  9. For each open HTTP port: http_fingerprint(url='http://{domain}:<port>')",
            f" 10. For each subdomain discovered: repeat steps 4-8",
        ]
    else:
        seq += [
            "",
            "Active scanning NOT authorized. To enable, call target_define again",
            "with authorization='own' or 'bug_bounty:<program>' or 'pentest:<client>'.",
        ]

    seq += [
        "",
        f"Target record:\n{json.dumps(record, indent=2)}",
    ]
    return "\n".join(seq)
