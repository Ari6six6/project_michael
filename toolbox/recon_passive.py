"""Passive recon orchestrator — chains all passive tools, returns canonical target model section.

No active scanning. Safe to run against any target without authorization.
Auto-executes.
"""
from __future__ import annotations

import importlib.util
import pathlib
from datetime import datetime, timezone
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "recon_passive",
        "description": (
            "Full passive recon on a domain in one call: DNS, certificate transparency, "
            "TLS, HTTP fingerprint, HTTP probe, IP intel. No active scanning — safe on any target. "
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
            },
            "required": ["domain"],
        },
    },
}

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


def recon_passive(domain: str, **kwargs) -> str:
    domain = domain.strip().lstrip("https://").lstrip("http://").rstrip("/").split("/")[0]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = f"https://{domain}"

    sections: list[str] = []
    sections.append(f"# recon_passive @ {domain}")
    sections.append(f"*Run: {ts}*\n")

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

    sections.append("\n---")
    sections.append(
        f"Merge these sections into targets/{domain}.md using the canonical template. "
        "Update Recon History with a one-line summary. "
        "Set Open Questions based on gaps found above."
    )

    return "\n".join(sections)
