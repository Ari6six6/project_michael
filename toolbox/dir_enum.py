"""Global tool: dir_enum — active HTTP path enumeration (requires authorization declaration)."""
from __future__ import annotations
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "dir_enum",
        "description": (
            "Active HTTP directory and path enumeration. REQUIRES authorized_by declaration. "
            "Probes a list of common paths on the target web server. Returns status codes, "
            "response sizes, and content-type for each discovered path. "
            "Detects: admin panels, API endpoints, config leaks, backup files, "
            "framework-specific paths, and exposed internal tooling."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Base URL (e.g. https://example.com). No trailing slash.",
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Custom path list. If omitted, uses built-in wordlist.",
                },
                "authorized_by": {
                    "type": "string",
                    "description": (
                        "REQUIRED. Authorization declaration: 'I own this system', "
                        "'bug_bounty:<program>', 'pentest:<client>'."
                    ),
                },
                "timeout_s": {
                    "type": "integer",
                    "description": "Per-request timeout in seconds. Default 5.",
                },
                "interesting_only": {
                    "type": "boolean",
                    "description": "If true (default), suppress 404/400/410 responses.",
                },
            },
            "required": ["url", "authorized_by"],
        },
    },
}

_BLOCKED = (
    "BLOCKED: dir_enum requires an explicit authorization declaration.\n\n"
    "Set authorized_by='I own this system' or 'bug_bounty:<program>' or 'pentest:<client>'."
)

_WORDLIST = [
    # Admin / auth
    "admin", "login", "logout", "signin", "signup", "register", "dashboard",
    "administrator", "admin/login", "admin/dashboard", "wp-admin", "wp-login.php",
    # API
    "api", "api/v1", "api/v2", "api/v3", "graphql", "graphiql", "query",
    "api-docs", "swagger", "swagger-ui", "swagger-ui.html", "swagger.json",
    "openapi.json", "openapi.yaml", "v1", "v2", "v3",
    # Config / secrets
    ".env", ".env.local", ".env.production", "config", "config.json", "config.yaml",
    "settings.py", "web.config", ".htaccess", ".htpasswd", "credentials.json",
    "secrets.json", "private", "backup", "backup.zip", "dump.sql",
    # Frameworks
    "actuator", "actuator/health", "actuator/env", "actuator/beans", "actuator/mappings",
    "server-status", "server-info", "phpinfo.php", "info.php", "test.php",
    "wp-json", "wp-json/wp/v2/users", "xmlrpc.php",
    # Files
    "robots.txt", "sitemap.xml", "sitemap.xml.gz", "security.txt", ".well-known/security.txt",
    "crossdomain.xml", "clientaccesspolicy.xml", "package.json", "composer.json",
    # Git / source
    ".git/HEAD", ".git/config", ".git/COMMIT_EDITMSG", ".svn/entries",
    "Dockerfile", "docker-compose.yml", "Makefile",
    # Monitoring / internal
    "health", "healthz", "ready", "readyz", "status", "ping", "metrics",
    "prometheus", "jaeger", "zipkin", "kibana", "grafana",
    # Uploads / media
    "uploads", "upload", "files", "media", "static", "assets", "images",
    # Common pages
    "about", "contact", "help", "support", "terms", "privacy",
]


def dir_enum(
    url: str,
    paths: list[str] | None = None,
    authorized_by: str = "",
    timeout_s: int = 5,
    interesting_only: bool = True,
    **kwargs: Any,
) -> str:
    if not authorized_by or not authorized_by.strip():
        return _BLOCKED

    import httpx

    base = url.strip().rstrip("/")
    probe_paths = paths or _WORDLIST
    auth_banner = f"[authorized: {authorized_by}]"
    results: list[str] = []
    errors = 0

    with httpx.Client(
        timeout=timeout_s,
        follow_redirects=False,
        headers={"User-Agent": "Mozilla/5.0 (compatible; michael-recon/1.0)"},
    ) as client:
        for path in probe_paths:
            target = f"{base}/{path.lstrip('/')}"
            try:
                r = client.get(target)
                status = r.status_code
                if interesting_only and status in (404, 400, 410, 501):
                    continue
                ct = r.headers.get("content-type", "")[:40]
                size = len(r.content)
                redirect = ""
                if status in (301, 302, 303, 307, 308):
                    redirect = f"  → {r.headers.get('location', '?')[:80]}"
                results.append(
                    f"  [{status}] /{path:<45} {size:>8}b  {ct}{redirect}"
                )
            except httpx.TimeoutException:
                errors += 1
            except Exception:
                errors += 1

    if not results:
        return (
            f"{auth_banner}\n"
            f"dir_enum on {base}: nothing interesting found "
            f"(errors: {errors}, paths checked: {len(probe_paths)})"
        )

    header = (
        f"{auth_banner}\n"
        f"dir_enum on {base}: {len(results)} interesting paths "
        f"(errors: {errors}, checked: {len(probe_paths)})\n"
    )
    return header + "\n".join(results)
