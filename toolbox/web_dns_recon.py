"""DNS reconnaissance via Cloudflare DNS-over-HTTPS. No external libraries needed."""
from __future__ import annotations

import json

import httpx

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_dns_recon",
        "description": (
            "Query DNS records for a domain via Cloudflare DoH (1.1.1.1). "
            "Returns A, AAAA, MX, TXT, NS, CNAME, SOA records. "
            "Optionally probes common subdomains (www, api, admin, staging, etc.) "
            "to discover which ones resolve. Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Domain to query, e.g. 'example.com' or 'api.example.com'",
                },
                "probe_subdomains": {
                    "type": "boolean",
                    "description": "If true, also try common subdomain prefixes against the apex domain",
                },
            },
            "required": ["domain"],
        },
    },
}

DOH_URL = "https://cloudflare-dns.com/dns-query"
DEFAULT_TYPES = ["A", "AAAA", "MX", "TXT", "NS", "CNAME", "SOA"]
COMMON_SUBDOMAINS = [
    "www", "api", "mail", "admin", "dev", "staging", "app",
    "auth", "cdn", "static", "assets", "media", "beta",
    "internal", "dashboard", "portal", "api-v2",
]


def _doh_query(client: httpx.Client, domain: str, rtype: str) -> list[str]:
    try:
        r = client.get(
            DOH_URL,
            params={"name": domain, "type": rtype, "ct": "application/dns-json"},
            headers={"Accept": "application/dns-json"},
        )
        r.raise_for_status()
        data = r.json()
        return [
            f"{a['data']} (TTL {a.get('TTL', '?')})"
            for a in data.get("Answer", [])
        ]
    except Exception as exc:
        return [f"error: {exc}"]


def web_dns_recon(domain: str, probe_subdomains: bool = False) -> str:
    records: dict[str, list[str]] = {}
    with httpx.Client(timeout=10.0) as client:
        for rtype in DEFAULT_TYPES:
            vals = _doh_query(client, domain, rtype)
            records[rtype] = vals

        subdomain_results: dict[str, list[str]] = {}
        if probe_subdomains:
            apex = domain if domain.count(".") == 1 else ".".join(domain.rsplit(".", 2)[-2:])
            for prefix in COMMON_SUBDOMAINS:
                sub = f"{prefix}.{apex}"
                a_records = _doh_query(client, sub, "A")
                if a_records and not a_records[0].startswith("error"):
                    subdomain_results[sub] = a_records

    out: list[str] = [f"=== web_dns_recon: {domain} ===\n"]
    out.append("[RECORDS]")
    for rtype, vals in records.items():
        if vals and not (len(vals) == 1 and vals[0].startswith("error")):
            out.append(f"  {rtype:<6}: {', '.join(vals[:8])}")
    out.append("")

    if probe_subdomains:
        out.append("[SUBDOMAIN PROBE]")
        if subdomain_results:
            for sub, addrs in subdomain_results.items():
                out.append(f"  {sub}: {', '.join(addrs)}")
        else:
            out.append("  No common subdomains resolved")

    return "\n".join(out)
