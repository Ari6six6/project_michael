"""Certificate Transparency log subdomain enumeration via crt.sh.

CT logs are public records of every TLS certificate ever issued. Every subdomain
that was ever certified shows up here — including dev environments, staging servers,
internal tools, and decommissioned services that never made it into DNS brute-force
lists. Far more comprehensive than common-prefix guessing.
"""
from __future__ import annotations

import json
import re
from typing import Any

import httpx

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_ct_enum",
        "description": (
            "Enumerate subdomains via Certificate Transparency logs (crt.sh). "
            "CT logs are public records of every TLS cert ever issued — reveals "
            "dev/staging/internal subdomains, decommissioned services, wildcard scopes, "
            "and infrastructure that DNS brute-force would never find. "
            "Returns deduplicated list of all unique subdomains ever certified, "
            "grouped by cert issuer type (Let's Encrypt vs CA vs self-signed). "
            "Also shows certificate issuance timeline — frequent new certs = active devops. "
            "Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Apex domain to search, e.g. 'example.com'",
                },
                "include_expired": {
                    "type": "boolean",
                    "description": "Include expired certificates (default true — expired certs still reveal historical subdomains)",
                },
            },
            "required": ["domain"],
        },
    },
}

CRT_SH_URL = "https://crt.sh/"


def _parse_names(name_value: str) -> list[str]:
    """Split multi-SAN name_value field into individual hostnames."""
    names: list[str] = []
    for raw in re.split(r"[\n,\s]+", name_value):
        h = raw.strip().lower().lstrip("*.")
        if h and "." in h and not h.startswith("http"):
            names.append(h)
    return names


def web_ct_enum(domain: str, include_expired: bool = True) -> str:
    apex = domain.lower().strip()
    # Strip any subdomain — query the apex
    parts = apex.split(".")
    if len(parts) > 2:
        apex = ".".join(parts[-2:])

    lines: list[str] = [f"=== web_ct_enum: {apex} ===\n"]

    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            params: dict[str, Any] = {
                "q": f"%.{apex}",
                "output": "json",
            }
            if not include_expired:
                params["exclude"] = "expired"

            r = client.get(CRT_SH_URL, params=params)
            r.raise_for_status()
            records: list[dict[str, Any]] = r.json()
    except Exception as exc:
        lines.append(f"ERROR: crt.sh query failed: {exc}")
        lines.append("(Network access to crt.sh required — works on deployed system)")
        return "\n".join(lines)

    if not records:
        lines.append("No certificates found in CT logs for this domain.")
        return "\n".join(lines)

    # Collect unique subdomains and metadata
    subdomains: dict[str, dict[str, Any]] = {}  # hostname → {first_seen, last_seen, issuers}
    issuers: dict[str, int] = {}

    for rec in records:
        name_val = rec.get("name_value", "")
        common = rec.get("common_name", "")
        logged_at = (rec.get("logged_at") or "")[:10]  # date only
        not_after = (rec.get("not_after") or "")[:10]
        issuer = rec.get("issuer_name", "")

        # Count issuer types
        issuer_org = "unknown"
        if "Let's Encrypt" in issuer:
            issuer_org = "Let's Encrypt"
        elif "DigiCert" in issuer:
            issuer_org = "DigiCert"
        elif "Comodo" in issuer or "Sectigo" in issuer:
            issuer_org = "Sectigo/Comodo"
        elif "GlobalSign" in issuer:
            issuer_org = "GlobalSign"
        elif "GoDaddy" in issuer:
            issuer_org = "GoDaddy"
        elif "Amazon" in issuer:
            issuer_org = "Amazon (ACM)"
        elif "Google" in issuer:
            issuer_org = "Google Trust Services"
        elif issuer:
            issuer_org = re.sub(r"^CN=|^O=", "", issuer.split(",")[0])[:40]
        issuers[issuer_org] = issuers.get(issuer_org, 0) + 1

        # Collect hostnames
        for name in _parse_names(name_val) + _parse_names(common):
            if not name.endswith(apex):
                continue
            if name not in subdomains:
                subdomains[name] = {
                    "first_seen": logged_at,
                    "last_seen": logged_at,
                    "last_expiry": not_after,
                    "issuers": set(),
                }
            entry = subdomains[name]
            if logged_at and (not entry["first_seen"] or logged_at < entry["first_seen"]):
                entry["first_seen"] = logged_at
            if logged_at and logged_at > entry["last_seen"]:
                entry["last_seen"] = logged_at
                entry["last_expiry"] = not_after
            entry["issuers"].add(issuer_org)

    # Sort subdomains: apex first, then alphabetical
    sorted_subs = sorted(subdomains.items(), key=lambda x: (0 if x[0] == apex else 1, x[0]))

    lines.append(f"Total certs in log : {len(records)}")
    lines.append(f"Unique subdomains  : {len(subdomains)}")
    lines.append("")

    lines.append("[CERTIFICATE ISSUERS]")
    for issuer_name, count in sorted(issuers.items(), key=lambda x: -x[1]):
        lines.append(f"  {count:>4}x  {issuer_name}")
    lines.append("")

    lines.append("[SUBDOMAINS — sorted, with last cert date]")
    now_expired: list[str] = []
    active: list[str] = []
    from datetime import date
    today = date.today().isoformat()
    for host, meta in sorted_subs:
        expiry = meta["last_expiry"]
        is_expired = expiry and expiry < today
        row = f"  {host:<50}  first={meta['first_seen']}  last={meta['last_seen']}"
        if is_expired:
            row += "  [EXPIRED]"
            now_expired.append(host)
        else:
            active.append(host)
        lines.append(row)

    lines.append("")
    lines.append(f"Active (non-expired) subdomains  : {len(active)}")
    lines.append(f"Historically expired subdomains  : {len(now_expired)}")

    if len(subdomains) > 50:
        lines.append("")
        lines.append(f"NOTE: {len(subdomains)} subdomains found — this is a large infrastructure. "
                     "Prioritize: api.*, admin.*, internal.*, staging.*, dev.*, auth.*")

    # Highlight interesting patterns
    interesting = [h for h in subdomains if any(k in h for k in
        ("admin", "internal", "staging", "dev", "api", "auth", "vpn", "mail", "git",
         "jenkins", "jira", "confluence", "grafana", "kibana", "elastic", "dashboard",
         "portal", "beta", "test", "uat", "prod", "secret", "secure", "private"))]
    if interesting:
        lines.append("")
        lines.append("[INTERESTING SUBDOMAINS]")
        for h in sorted(interesting)[:30]:
            meta = subdomains[h]
            lines.append(f"  {h}  (last cert: {meta['last_seen']})")

    return "\n".join(lines)
