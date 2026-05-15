"""Global tool: http_fingerprint — HTTP headers, tech stack, and surface analysis."""
from __future__ import annotations
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "http_fingerprint",
        "description": (
            "HTTP fingerprint: server banner, response headers, tech stack detection, "
            "security header audit, cookie flags, and body preview. "
            "Detects: nginx/apache/cloudflare, WordPress/Drupal/Django/Rails/Laravel/Next.js, "
            "GraphQL endpoints, Swagger/OpenAPI, Spring Boot actuators, and more. "
            "Passive — single GET request."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL including scheme (https://example.com).",
                },
                "follow_redirects": {
                    "type": "boolean",
                    "description": "Follow HTTP redirects. Default true.",
                },
            },
            "required": ["url"],
        },
    },
}

_TECH_SIGS: list[tuple[str, list[str]]] = [
    ("nginx",       ["server: nginx"]),
    ("apache",      ["server: apache"]),
    ("cloudflare",  ["server: cloudflare", "cf-ray", "cf-cache-status"]),
    ("fastly",      ["x-served-by", "fastly-restarts"]),
    ("AWS/ELB",     ["server: awselb", "x-amzn-requestid", "x-amz-cf-id"]),
    ("WordPress",   ["wp-content/", "wp-json/", "x-pingback"]),
    ("Drupal",      ["x-generator: drupal", "drupal.settings"]),
    ("Joomla",      ["/components/com_", "/media/jui/"]),
    ("Django",      ["csrftoken", "x-frame-options: sameorigin"]),
    ("Rails",       ["x-powered-by: phusion passenger", "x-request-id", "_session_id"]),
    ("Laravel",     ["laravel_session", "x-powered-by: php"]),
    ("Next.js",     ["x-powered-by: next.js", "__next_data__", "__NEXT_DATA__"]),
    ("React",       ["react-dom", "__react"]),
    ("Vue.js",      ["vue.js", "__vue__"]),
    ("Angular",     ["ng-version", "_nghost"]),
    ("Spring Boot", ["x-application-context", "x-content-type-options", "actuator"]),
    ("GraphQL",     ["/graphql", "graphiql", "__schema"]),
    ("Swagger",     ["swagger-ui", "swagger.json", "openapi.json", "api-docs"]),
    ("PHP",         ["x-powered-by: php", ".php"]),
    ("ASP.NET",     ["x-aspnet-version", "x-aspnetmvc-version", "__viewstate"]),
    ("IIS",         ["server: microsoft-iis"]),
]

_SECURITY_HEADERS = [
    "strict-transport-security",
    "content-security-policy",
    "x-frame-options",
    "x-content-type-options",
    "referrer-policy",
    "permissions-policy",
    "x-xss-protection",
]


def http_fingerprint(url: str, follow_redirects: bool = True, **kwargs: Any) -> str:
    import httpx
    url = url.strip()
    try:
        r = httpx.get(
            url,
            timeout=15,
            follow_redirects=follow_redirects,
            headers={"User-Agent": "Mozilla/5.0 (compatible; michael-recon/1.0)"},
        )
    except Exception as e:
        return f"http_fingerprint error: {e}"

    headers_lower = {k.lower(): v for k, v in r.headers.items()}
    body_preview = r.text[:8000]
    combined = " ".join(headers_lower.values()) + " " + body_preview

    # Tech detection
    detected = []
    for tech, markers in _TECH_SIGS:
        if any(m.lower() in combined.lower() for m in markers):
            detected.append(tech)

    # Security header audit
    present_sec = [h for h in _SECURITY_HEADERS if h in headers_lower]
    missing_sec = [h for h in _SECURITY_HEADERS if h not in headers_lower]

    # Cookie flags
    cookies_info = []
    for name, val in r.cookies.items():
        cookies_info.append(f"  {name}")

    lines = [
        f"URL:         {url}",
        f"Final URL:   {r.url}",
        f"Status:      {r.status_code}",
        f"Content-Type:{headers_lower.get('content-type', '?')}",
        f"Server:      {headers_lower.get('server', '?')}",
        f"X-Powered-By:{headers_lower.get('x-powered-by', '—')}",
        "",
        "=== Detected Tech ===",
        ("  " + ", ".join(detected)) if detected else "  (nothing fingerprinted)",
        "",
        "=== All Headers ===",
    ]
    for k, v in r.headers.items():
        lines.append(f"  {k}: {v}")

    lines += [
        "",
        f"=== Security Headers ===",
        f"  present ({len(present_sec)}): {', '.join(present_sec) or 'none'}",
        f"  missing ({len(missing_sec)}): {', '.join(missing_sec) or 'none'}",
    ]

    if cookies_info:
        lines += ["", "=== Cookies ==="] + cookies_info

    lines += [
        "",
        f"=== Body Preview (first 800 chars) ===",
        r.text[:800],
    ]
    return "\n".join(lines)
