"""Full recon orchestrator — passive + active (port scan + dir enum).

ACTIVE: requires authorized_by declaration. Blocked without it.
Auto-executes.
"""
from __future__ import annotations

import importlib.util
import pathlib
from datetime import datetime, timezone

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "recon_full",
        "description": (
            "Full recon on a domain: all passive tools (DNS, CT, TLS, HTTP, fingerprint, IP intel) "
            "plus active scanning (port scan, directory enumeration). "
            "ACTIVE — requires authorized_by. Set it to 'I own this system', "
            "'bug_bounty:<program>', or 'pentest:<client>'. "
            "Returns structured markdown sections ready to merge into targets/<domain>.md. "
            "Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Target domain, e.g. 'example.com'",
                },
                "authorized_by": {
                    "type": "string",
                    "description": (
                        "Authorization declaration. Required. "
                        "Examples: 'I own this system', 'bug_bounty:HackerOne', 'pentest:ClientCorp'"
                    ),
                },
                "ports": {
                    "type": "string",
                    "description": "Port range to scan (default: '1-1024')",
                },
            },
            "required": ["domain", "authorized_by"],
        },
    },
}

_BLOCKED = (
    "BLOCKED: recon_full requires an explicit authorization declaration.\n\n"
    "Set authorized_by to one of:\n"
    "  'I own this system'\n"
    "  'bug_bounty:<program name>'\n"
    "  'pentest:<client name>'\n\n"
    "For passive-only recon (no auth needed), use recon_passive instead."
)

_HERE = pathlib.Path(__file__).parent


def _tool(name: str):
    py = _HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, py)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return getattr(mod, name)


def _run(name: str, **kwargs) -> str:
    try:
        fn = _tool(name)
        result = fn(**kwargs)
        return str(result)
    except Exception as exc:
        return f"[{name} error: {exc}]"


def recon_full(domain: str, authorized_by: str = "", ports: str = "1-1024", **kwargs) -> str:
    if not authorized_by or not authorized_by.strip():
        return _BLOCKED

    domain = domain.strip().lstrip("https://").lstrip("http://").rstrip("/").split("/")[0]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"https://{domain}"

    sections: list[str] = []
    sections.append(f"# recon_full @ {domain}")
    sections.append(f"*Run: {ts} | Auth: {authorized_by}*\n")

    sections.append("## DNS")
    sections.append(_run("web_dns_recon", domain=domain, probe_subdomains=True))

    sections.append("\n## Subdomains (Certificate Transparency)")
    sections.append(_run("web_ct_enum", domain=domain))

    sections.append("\n## TLS")
    sections.append(_run("web_tls_info", url=url))

    sections.append("\n## HTTP Stack")
    sections.append(_run("web_http_probe", url=url))

    sections.append("\n## Software Fingerprint")
    sections.append(_run("web_fingerprint", url=url))

    sections.append("\n## IP Intel")
    sections.append(_run("web_ip_intel", domain=domain))

    sections.append("\n## Services (Port Scan)")
    sections.append(_run("port_scan", target=domain, ports=ports, authorized_by=authorized_by))

    sections.append("\n## Endpoints (Directory Enum)")
    sections.append(_run("dir_enum", url=url, authorized_by=authorized_by))

    sections.append("\n---")
    sections.append(
        f"Merge these sections into targets/{domain}.md using the canonical template. "
        "Update Recon History with a one-line summary. "
        "Set Open Questions based on gaps, anomalies, and findings above. "
        "Call source_map for any detected software versions before committing."
    )

    return "\n".join(sections)
