"""HTTP endpoint probe: detailed request/response analysis with stack fingerprinting."""
from __future__ import annotations

import json
from typing import Any

import httpx

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_http_probe",
        "description": (
            "Probe a specific HTTP endpoint (or a host with well-known paths) for detailed "
            "response analysis. Returns: status code, timing, full response headers, "
            "tech stack fingerprints (server, CDN, framework from cookies/headers), "
            "body preview (first 3000 chars), and auth challenge details (401/403 + "
            "WWW-Authenticate header). "
            "If paths is provided, probes each path and returns a status-code table. "
            "verify=False so self-signed certs don't block. Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to probe, e.g. 'https://api.example.com/openapi.json'",
                },
                "method": {
                    "type": "string",
                    "description": "HTTP method (default GET)",
                },
                "headers": {
                    "type": "object",
                    "description": "Additional request headers as key/value object",
                },
                "body": {
                    "type": "string",
                    "description": "Request body string (for POST/PUT/PATCH)",
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "If given, probe each path against the URL's base (scheme://host) "
                        "and return a status table instead of full response detail."
                    ),
                },
                "follow_redirects": {
                    "type": "boolean",
                    "description": "Follow redirects (default true)",
                },
                "timeout": {
                    "type": "number",
                    "description": "Request timeout in seconds (default 10)",
                },
            },
            "required": ["url"],
        },
    },
}

FRAMEWORK_COOKIES = {
    "PHPSESSID": "PHP",
    "JSESSIONID": "Java/JVM",
    "ASP.NET_SessionId": "ASP.NET",
    "_rails": "Ruby on Rails",
    "laravel_session": "Laravel/PHP",
    "csrftoken": "Django/Python",
    "rack.session": "Rack/Ruby",
    "connect.sid": "Express.js/Node",
    "_session_id": "Rails (possible)",
}


def _fingerprint(headers: httpx.Headers) -> list[str]:
    hints: list[str] = []
    h = {k.lower(): v for k, v in headers.items()}
    for key in ("server", "x-powered-by", "x-generator", "x-aspnet-version",
                "x-aspnetmvc-version", "x-runtime", "x-backend"):
        if key in h:
            hints.append(f"{key}: {h[key]}")
    if "cf-ray" in h:
        hints.append("CDN: Cloudflare")
    if "x-amz-cf-id" in h:
        hints.append("CDN: CloudFront (AWS)")
    if "x-served-by" in h and "fastly" in h.get("x-served-by", "").lower():
        hints.append("CDN: Fastly")
    if "via" in h and "varnish" in h["via"].lower():
        hints.append("CDN: Varnish")
    return hints


def web_http_probe(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None,
    paths: list[str] | None = None,
    follow_redirects: bool = True,
    timeout: float = 10.0,
) -> str:
    from urllib.parse import urlparse
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    client_kwargs: dict[str, Any] = {
        "follow_redirects": follow_redirects,
        "timeout": timeout,
        "verify": False,
    }

    # Multi-path table mode
    if paths:
        lines: list[str] = [f"=== web_http_probe path table: {base} ===\n"]
        with httpx.Client(**client_kwargs) as client:
            for path in paths:
                target = base + path
                try:
                    r = client.request(method, target, headers=headers or {})
                    ct = r.headers.get("content-type", "")[:50]
                    loc = r.headers.get("location", "")
                    loc_str = f" → {loc}" if loc else ""
                    lines.append(f"  {r.status_code}  {path}{loc_str}  [{ct}]")
                except Exception as exc:
                    lines.append(f"  ERR  {path}  ({exc})")
        return "\n".join(lines)

    # Single URL deep probe
    lines = [f"=== web_http_probe: {method} {url} ===\n"]
    try:
        with httpx.Client(**client_kwargs) as client:
            req_kwargs: dict[str, Any] = {"headers": headers or {}}
            if body is not None:
                req_kwargs["content"] = body.encode()
            r = client.request(method, url, **req_kwargs)

        lines.append(f"Status      : {r.status_code} {r.reason_phrase}")
        lines.append(f"Elapsed     : {round(r.elapsed.total_seconds() * 1000)}ms")
        lines.append(f"Final URL   : {r.url}")
        lines.append(f"Content-Type: {r.headers.get('content-type', '?')}")
        content_len = r.headers.get("content-length", str(len(r.content)))
        lines.append(f"Size        : {content_len} bytes")
        lines.append("")

        hints = _fingerprint(r.headers)
        if hints:
            lines.append("[STACK FINGERPRINT]")
            for h in hints:
                lines.append(f"  {h}")
            lines.append("")

        lines.append("[RESPONSE HEADERS]")
        for k, v in r.headers.items():
            lines.append(f"  {k}: {v[:200]}")
        lines.append("")

        # Cookie analysis
        cookie_names = []
        for ch in r.headers.get_list("set-cookie"):
            name = ch.split("=")[0].strip()
            cookie_names.append(name)
        if cookie_names:
            lines.append("[COOKIES]")
            for name in cookie_names:
                hint = FRAMEWORK_COOKIES.get(name, "")
                lines.append(f"  {name}" + (f" → {hint}" if hint else ""))
            lines.append("")

        if r.status_code in (401, 403):
            lines.append("[AUTH CHALLENGE]")
            www_auth = r.headers.get("www-authenticate", "")
            lines.append(f"  Status: {r.status_code}")
            if www_auth:
                lines.append(f"  WWW-Authenticate: {www_auth}")
            lines.append("")

        lines.append("[BODY PREVIEW — first 3000 chars]")
        try:
            text = r.text[:3000]
        except Exception:
            text = r.content[:3000].decode("utf-8", errors="replace")
        lines.append(text)

    except Exception as exc:
        lines.append(f"ERROR: {exc}")

    return "\n".join(lines)
