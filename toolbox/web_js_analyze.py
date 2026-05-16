"""JavaScript bundle intelligence: extract API endpoints, internal hosts, env vars.

Modern SPAs compile everything into JS bundles. Those bundles contain hardcoded
API routes, internal hostnames, feature flags, and environment variable names —
the full map of what the app talks to and how, readable without auth.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_js_analyze",
        "description": (
            "Fetch and analyze JavaScript bundle files from a URL. "
            "Extracts: API endpoint paths (URL patterns hardcoded in the bundle), "
            "internal hostnames and IP addresses referenced in code, "
            "environment variable names (REACT_APP_*, VUE_APP_*, NEXT_PUBLIC_*, etc.), "
            "feature flag keys, third-party service integrations (Stripe, Segment, "
            "Sentry, Intercom, analytics, etc.), and framework/library versions from "
            "bundle strings. "
            "Works on React/Vue/Angular/Next.js SPAs where the API surface is compiled "
            "into the frontend code. Fetches up to max_files JS files, first 80KB each. "
            "Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Page URL to start from — extracts script tags and fetches bundles",
                },
                "max_files": {
                    "type": "integer",
                    "description": "Max JS files to fetch and analyze (default 5)",
                },
            },
            "required": ["url"],
        },
    },
}

# API endpoint patterns — URL strings that look like API routes
API_PATH_PATTERNS = [
    r'["\`]/(?:api|v\d+|graphql|rest|service|rpc|ws)/[^\s"\'`\\]{3,80}',
    r'["\`](?:https?://[^"\s`\'\\]{10,120})',
    r'(?:endpoint|baseUrl|apiUrl|API_URL|BASE_URL)\s*[:=]\s*["\`]([^"\`\n]{5,100})',
    r'(?:path|route|url)\s*:\s*["\`](/[^"\`\n]{3,80})',
]

# Internal hostname patterns
INTERNAL_HOST_PATTERNS = [
    r'\b([a-z0-9][a-z0-9-]{2,}\.(?:internal|local|corp|lan|intranet|priv|int))\b',
    r'\b([a-z0-9][a-z0-9-]{2,}\.(?:svc\.cluster\.local))\b',  # Kubernetes
    r'\b((?:10|172|192)\.(?:\d{1,3}\.){2}\d{1,3})\b',  # RFC1918 IPs
]

# Environment variable name patterns
ENV_VAR_PATTERNS = [
    r'(?:process\.env\.|import\.meta\.env\.)([A-Z][A-Z0-9_]{2,50})',
    r'["\']([A-Z][A-Z0-9_]*(?:_URL|_KEY|_SECRET|_TOKEN|_HOST|_PORT|_API)["\'])',
    r'(?:REACT_APP|VUE_APP|NEXT_PUBLIC|NUXT_|VITE_)_([A-Z0-9_]+)',
]

# Third-party service fingerprints
THIRD_PARTY_SIGNALS: list[tuple[str, str]] = [
    (r'stripe\.com|stripe\.js|Stripe\(',                    "Stripe (payments)"),
    (r'segment\.com|analytics\.track|analytics\.identify',  "Segment (analytics)"),
    (r'sentry\.io|Sentry\.init|@sentry/',                   "Sentry (error tracking)"),
    (r'intercom\.io|Intercom\(',                             "Intercom (support)"),
    (r'mixpanel\.com|mixpanel\.track',                       "Mixpanel (analytics)"),
    (r'amplitude\.com|amplitude\.init',                     "Amplitude (analytics)"),
    (r'launchdarkly\.com|LDClient|ld-client',               "LaunchDarkly (feature flags)"),
    (r'fullstory\.com|FullStory\(',                         "FullStory (session recording)"),
    (r'hotjar\.com|hj\(',                                   "Hotjar (session recording)"),
    (r'datadog-rum|datadog\.com',                            "Datadog RUM"),
    (r'newrelic\.com|NREUM\b',                              "New Relic"),
    (r'rollbar\.com|Rollbar\.init',                         "Rollbar (error tracking)"),
    (r'firebase\.google\.com|initializeApp\(',              "Firebase"),
    (r'supabase\.co|createClient\(',                        "Supabase"),
    (r'auth0\.com|createAuth0Client\(',                     "Auth0"),
    (r'cognito-idp\.|AmazonCognitoIdentity',               "AWS Cognito"),
    (r'okta\.com|OktaAuth\(',                              "Okta"),
    (r'hubspot\.com|HubSpotForms',                          "HubSpot"),
    (r'zendesk\.com|zE\(',                                  "Zendesk"),
    (r'twilio\.com|TwilioVideo',                           "Twilio"),
    (r'sendgrid\.net|sendgrid\.com',                        "SendGrid"),
    (r'cloudinary\.com',                                    "Cloudinary (media)"),
    (r'mapbox\.com|mapbox-gl',                              "Mapbox"),
    (r'google-analytics\.com|gtag\(',                      "Google Analytics"),
    (r'googletagmanager\.com',                              "Google Tag Manager"),
    (r'facebook\.net|fbq\(',                               "Facebook Pixel"),
    (r'amazonaws\.com/[a-z0-9-]+/',                        "AWS S3/CloudFront"),
    (r'graphql',                                            "GraphQL"),
    (r'socket\.io|io\.connect\(',                          "Socket.IO (WebSocket)"),
]

# Version strings in bundles
BUNDLE_VERSION_PATTERNS: list[tuple[str, str]] = [
    (r'"version"\s*:\s*"(\d+\.\d+[\.\d]*)"',  "version"),
    (r'VERSION\s*=\s*["\'](\d+\.\d+[\.\d]*)["\']', "VERSION"),
    (r'react@(\d+\.\d+[\.\d]*)',              "React"),
    (r'vue@(\d+\.\d+[\.\d]*)',               "Vue.js"),
    (r'angular@(\d+\.\d+[\.\d]*)',           "Angular"),
    (r'next@(\d+\.\d+[\.\d]*)',              "Next.js"),
]


class _ScriptExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.src_urls: list[str] = []
        self.inline_scripts: list[str] = []
        self._current_inline = False
        self._inline_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag != "script":
            return
        a = dict(attrs)
        src = a.get("src", "")
        if src:
            self.src_urls.append(src)
        elif not a.get("type") or "javascript" in a.get("type", ""):
            self._current_inline = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._current_inline:
            self.inline_scripts.append("".join(self._inline_buf))
            self._inline_buf = []
            self._current_inline = False

    def handle_data(self, data: str) -> None:
        if self._current_inline:
            self._inline_buf.append(data)


def _is_bundle(url: str) -> bool:
    """True if the URL looks like a compiled JS bundle (not a library CDN)."""
    path = urlparse(url).path.lower()
    # Include files with content hashes, chunk names, or generic names
    if re.search(r'\.[a-f0-9]{6,}\.(js|mjs)$', path):  # content hash
        return True
    if re.search(r'(?:chunk|bundle|main|app|vendor|runtime|polyfill)[^/]*\.(?:js|mjs)$', path):
        return True
    return False


def _analyze_body(body: str, base_url: str, results: dict) -> None:
    """Scan a JS body for signals, merging into results dict."""
    # API paths
    for pat in API_PATH_PATTERNS:
        for m in re.finditer(pat, body):
            hit = m.group(0).strip('"\'`')
            if len(hit) > 4 and hit not in results["api_paths"]:
                results["api_paths"].add(hit)

    # Internal hosts
    for pat in INTERNAL_HOST_PATTERNS:
        for m in re.finditer(pat, body, re.IGNORECASE):
            results["internal_hosts"].add(m.group(1))

    # Env vars
    for pat in ENV_VAR_PATTERNS:
        for m in re.finditer(pat, body):
            results["env_vars"].add(m.group(1))

    # Third-party services
    for pat, label in THIRD_PARTY_SIGNALS:
        if re.search(pat, body, re.IGNORECASE) and label not in results["third_party"]:
            results["third_party"].add(label)

    # Versions
    for pat, label in BUNDLE_VERSION_PATTERNS:
        m = re.search(pat, body)
        if m and label not in results["versions"]:
            results["versions"][label] = m.group(1)


def web_js_analyze(url: str, max_files: int = 5) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    lines: list[str] = [f"=== web_js_analyze: {parsed.netloc} ===\n"]

    results: dict = {
        "api_paths": set(),
        "internal_hosts": set(),
        "env_vars": set(),
        "third_party": set(),
        "versions": {},
    }

    fetched_files: list[str] = []

    client_kwargs = {"follow_redirects": True, "timeout": 12.0, "verify": False}

    # 1. Fetch the page and extract script tags
    try:
        with httpx.Client(**client_kwargs) as c:
            r = c.get(url)
        extractor = _ScriptExtractor()
        extractor.feed(r.text[:200_000])

        # Analyze inline scripts first
        for inline in extractor.inline_scripts:
            if len(inline) > 50:
                _analyze_body(inline, base, results)

        # Resolve script src URLs
        script_urls: list[str] = []
        for src in extractor.src_urls:
            full = urljoin(base, src)
            script_urls.append(full)

        # Prefer bundles over CDN libraries
        bundles = [u for u in script_urls if _is_bundle(u)]
        others = [u for u in script_urls if not _is_bundle(u)]
        ordered = bundles + others

    except Exception as exc:
        lines.append(f"ERROR fetching page: {exc}")
        return "\n".join(lines)

    lines.append(f"Script tags found  : {len(script_urls)}")
    lines.append(f"  Bundles (local)  : {len(bundles)}")
    lines.append(f"  CDN/external     : {len(others)}")
    lines.append("")

    # 2. Fetch and analyze JS files
    with httpx.Client(**client_kwargs) as c:
        for js_url in ordered[:max_files]:
            try:
                r = c.get(js_url, headers={"Accept": "application/javascript, */*"})
                if r.status_code != 200:
                    continue
                body = r.text[:80_000]
                _analyze_body(body, base, results)
                fetched_files.append(js_url)
            except Exception:
                pass

    lines.append(f"Files analyzed     : {len(fetched_files)}")
    for f in fetched_files:
        lines.append(f"  {f}")
    lines.append("")

    # ── Render findings ──────────────────────────────────────────────────────

    lines.append("[API ENDPOINTS / URL PATTERNS]")
    api_paths = sorted(results["api_paths"])
    if api_paths:
        # Deduplicate and filter noise
        seen_paths: set[str] = set()
        for p in api_paths[:80]:
            # Normalize
            clean = p.strip("'\"` ")
            if clean in seen_paths or len(clean) < 5:
                continue
            seen_paths.add(clean)
            lines.append(f"  {clean}")
    else:
        lines.append("  None found")
    lines.append("")

    lines.append("[THIRD-PARTY SERVICES DETECTED]")
    if results["third_party"]:
        for svc in sorted(results["third_party"]):
            lines.append(f"  {svc}")
    else:
        lines.append("  None detected")
    lines.append("")

    lines.append("[ENVIRONMENT VARIABLES REFERENCED]")
    if results["env_vars"]:
        for var in sorted(results["env_vars"])[:40]:
            lines.append(f"  {var}")
    else:
        lines.append("  None found")
    lines.append("")

    lines.append("[INTERNAL HOSTS / RFC1918 IPs REFERENCED]")
    if results["internal_hosts"]:
        for host in sorted(results["internal_hosts"]):
            lines.append(f"  {host}")
    else:
        lines.append("  None found")
    lines.append("")

    if results["versions"]:
        lines.append("[VERSIONS IN BUNDLE]")
        for label, ver in sorted(results["versions"].items()):
            lines.append(f"  {label}: {ver}")
        lines.append("")

    return "\n".join(lines)
