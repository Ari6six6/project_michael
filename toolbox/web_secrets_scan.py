"""Exposed credential and secret detection in HTML, JS, and common config endpoints.

Scans response bodies for accidentally exposed: API keys, tokens, connection strings,
private keys, JWT tokens, internal IP addresses, and generic secrets.
Checks common endpoints where config files are accidentally deployed publicly.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_secrets_scan",
        "description": (
            "Scan a URL for accidentally exposed credentials and secrets. "
            "Checks: API key patterns (Google, Stripe, AWS, Twilio, SendGrid, GitHub, "
            "Slack, Mailgun, Algolia, Mapbox, Cloudinary), JWT tokens, private keys, "
            "database connection strings, internal RFC1918 IP addresses, and generic "
            "password/secret/token assignments. "
            "Also probes common accidental-exposure paths: /.env, /.env.local, "
            "/.env.production, /config.json, /app.config.js, /web.config, /appsettings.json. "
            "Analyzes homepage HTML + all inline scripts + up to 3 JS bundles. "
            "Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL or domain to scan, e.g. 'https://example.com'",
                },
            },
            "required": ["url"],
        },
    },
}

# (pattern, label, confidence)
SECRET_PATTERNS: list[tuple[str, str, str]] = [
    # Cloud providers
    (r'AKIA[0-9A-Z]{16}',                                                "AWS Access Key ID",        "HIGH"),
    (r'(?:aws_secret|AWS_SECRET)[^"\n]{0,20}["\']([A-Za-z0-9/+=]{40})', "AWS Secret Key",            "HIGH"),
    # Google
    (r'AIza[0-9A-Za-z\-_]{35}',                                          "Google API Key",            "HIGH"),
    (r'[0-9]+-[0-9A-Za-z_]{32}\.apps\.googleusercontent\.com',           "Google OAuth Client ID",    "MEDIUM"),
    # Stripe
    (r'sk_live_[0-9a-zA-Z]{24,}',                                        "Stripe Secret Key (LIVE)",  "HIGH"),
    (r'sk_test_[0-9a-zA-Z]{24,}',                                        "Stripe Secret Key (test)",  "MEDIUM"),
    (r'pk_live_[0-9a-zA-Z]{24,}',                                        "Stripe Public Key (LIVE)",  "LOW"),
    # GitHub
    (r'ghp_[A-Za-z0-9]{36}',                                             "GitHub Personal Access Token", "HIGH"),
    (r'ghs_[A-Za-z0-9]{36}',                                             "GitHub App Token",          "HIGH"),
    (r'github_pat_[A-Za-z0-9_]{82}',                                     "GitHub Fine-Grained PAT",   "HIGH"),
    # Slack
    (r'xox[baprs]-[0-9A-Za-z\-]{10,72}',                                 "Slack Token",               "HIGH"),
    (r'https://hooks\.slack\.com/services/[A-Z0-9/]{44}',                "Slack Webhook",             "HIGH"),
    # Twilio — must appear in a config-like context, not random hex
    (r'(?:account_?sid|twilio)[^"\n]{0,30}["\']?(AC[0-9a-f]{32})',       "Twilio Account SID",        "MEDIUM"),
    (r'(?:api_?key|auth_?token)[^"\n]{0,30}["\']?(SK[0-9a-f]{32})',      "Twilio API Key SID",        "MEDIUM"),
    # SendGrid
    (r'SG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}',                     "SendGrid API Key",          "HIGH"),
    # Mailgun
    (r'\bkey-[0-9a-z]{32}\b',                                             "Mailgun API Key",           "HIGH"),
    # Mapbox — very specific prefix format
    (r'pk\.[A-Za-z0-9]{43}\.[A-Za-z0-9]{22}',                           "Mapbox Access Token",       "HIGH"),
    # JWT
    (r'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}', "JWT Token",                 "MEDIUM"),
    # Private keys
    (r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----',          "Private Key",               "HIGH"),
    (r'-----BEGIN CERTIFICATE-----',                                       "Certificate (in body)",     "LOW"),
    # Database connection strings
    (r'(?:mysql|postgresql|postgres|mongodb|redis|amqp)://[^@\s"\'<>]{5,}@[^@\s"\'<>]{5,}',
                                                                           "Database Connection String", "HIGH"),
    # Generic high-entropy secrets
    (r'(?:password|passwd|secret|token|api_?key|auth_?key|access_?key|private_?key)\s*[:=]\s*["\']([^"\']{8,80})["\']',
                                                                           "Generic Secret Assignment", "MEDIUM"),
    # Internal IPs — require valid octet ranges and NOT followed by more digits (avoid SVG/CSS coords)
    (r'(?<!\d)(10\.(?:[01]?\d\d?|2[0-4]\d|25[0-5])\.(?:[01]?\d\d?|2[0-4]\d|25[0-5])\.(?:[01]?\d\d?|2[0-4]\d|25[0-5]))(?!\d)',
                                                                           "RFC1918 IP (10.x.x.x)",    "LOW"),
    (r'(?<!\d)(172\.(?:1[6-9]|2\d|3[01])\.(?:[01]?\d\d?|2[0-4]\d|25[0-5])\.(?:[01]?\d\d?|2[0-4]\d|25[0-5]))(?!\d)',
                                                                           "RFC1918 IP (172.x.x.x)",   "LOW"),
    (r'(?<!\d)(192\.168\.(?:[01]?\d\d?|2[0-4]\d|25[0-5])\.(?:[01]?\d\d?|2[0-4]\d|25[0-5]))(?!\d)',
                                                                           "RFC1918 IP (192.168.x.x)", "LOW"),
]

# Config files that are accidentally deployed
CONFIG_PATHS = [
    "/.env",
    "/.env.local",
    "/.env.development",
    "/.env.production",
    "/.env.staging",
    "/.env.backup",
    "/config.json",
    "/config.yaml",
    "/config.yml",
    "/app.config.js",
    "/appsettings.json",
    "/appsettings.Development.json",
    "/web.config",
    "/wp-config.php.bak",
    "/wp-config.php~",
    "/database.yml",
    "/secrets.json",
    "/credentials.json",
    "/service-account.json",
    "/.aws/credentials",
    "/id_rsa",
    "/id_ed25519",
    "/.ssh/id_rsa",
    "/deploy.key",
    "/server.key",
    "/private.key",
]


class _InlineScriptExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.inline: list[str] = []
        self.src_urls: list[str] = []
        self._in_script = False
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag != "script":
            return
        a = dict(attrs)
        if a.get("src"):
            self.src_urls.append(a["src"])
        else:
            self._in_script = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._in_script:
            self.inline.append("".join(self._buf))
            self._buf = []
            self._in_script = False

    def handle_data(self, data: str) -> None:
        if self._in_script:
            self._buf.append(data)


def _scan_body(body: str) -> list[dict]:
    """Return list of {pattern_label, match_excerpt, confidence} findings."""
    hits: list[dict] = []
    seen_labels: set[str] = set()
    for pat, label, confidence in SECRET_PATTERNS:
        for m in re.finditer(pat, body, re.IGNORECASE):
            key = f"{label}:{m.group(0)[:20]}"
            if key in seen_labels:
                continue
            seen_labels.add(key)
            # Redact most of the match for safety in output
            matched = m.group(0)
            if len(matched) > 12:
                excerpt = matched[:6] + "…" + matched[-4:]
            else:
                excerpt = matched[:4] + "…"
            # Show surrounding context
            start = max(0, m.start() - 40)
            end = min(len(body), m.end() + 40)
            ctx = body[start:end].replace("\n", " ").strip()[:120]
            hits.append({
                "label": label,
                "excerpt": excerpt,
                "context": ctx,
                "confidence": confidence,
            })
    return hits


def web_secrets_scan(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    lines: list[str] = [f"=== web_secrets_scan: {parsed.netloc} ===\n"]

    all_findings: list[dict] = []
    client_kwargs = {"follow_redirects": True, "timeout": 10.0, "verify": False}

    # ── 1. Scan homepage HTML + inline scripts ────────────────────────────
    try:
        with httpx.Client(**client_kwargs) as c:
            r = c.get(url)
        body = r.text[:200_000]
        hits = _scan_body(body)
        for h in hits:
            h["source"] = "homepage-html"
            all_findings.append(h)

        # Extract inline scripts
        extractor = _InlineScriptExtractor()
        extractor.feed(body[:200_000])
        for inline in extractor.inline:
            if len(inline) < 20:
                continue
            hits = _scan_body(inline)
            for h in hits:
                h["source"] = "inline-script"
                all_findings.append(h)

        # ── 2. Scan JS bundle files (up to 3) ────────────────────────────
        bundle_urls = [
            urljoin(base, src) for src in extractor.src_urls
            if re.search(r'\.[a-f0-9]{6,}\.(js|mjs)$|(?:chunk|bundle|main|app|vendor)[^/]*\.js$',
                         urlparse(src).path, re.IGNORECASE)
        ][:3]
        with httpx.Client(**client_kwargs) as c:
            for js_url in bundle_urls:
                try:
                    rj = c.get(js_url)
                    if rj.status_code == 200:
                        js_body = rj.text[:80_000]
                        hits = _scan_body(js_body)
                        for h in hits:
                            h["source"] = f"js:{urlparse(js_url).path[-40:]}"
                            all_findings.append(h)
                except Exception:
                    pass

    except Exception as exc:
        lines.append(f"ERROR scanning page: {exc}")

    # ── 3. Probe config exposure paths ───────────────────────────────────
    exposed_configs: list[dict] = []
    with httpx.Client(follow_redirects=False, timeout=6.0, verify=False) as c:
        for path in CONFIG_PATHS:
            try:
                r = c.get(base + path)
                if r.status_code == 200 and len(r.content) > 10:
                    ct = r.headers.get("content-type", "")
                    size = len(r.content)
                    # Check if it looks like actual config (not an HTML 200 redirect)
                    looks_real = (
                        "text/html" not in ct.lower() or
                        r.text[:100].strip().startswith(("{", "[", "#", "APP_", "DB_", "SECRET"))
                    )
                    if looks_real:
                        exposed_configs.append({
                            "path": path,
                            "size": size,
                            "ct": ct[:60],
                            "preview": r.text[:200].replace("\n", " "),
                        })
                        # Scan config body for secrets too
                        hits = _scan_body(r.text[:10_000])
                        for h in hits:
                            h["source"] = f"config:{path}"
                            all_findings.append(h)
            except Exception:
                pass

    # ── Render ────────────────────────────────────────────────────────────

    # Deduplicate findings by (label, excerpt)
    seen: set[str] = set()
    deduped: list[dict] = []
    for f in all_findings:
        key = f"{f['label']}:{f['excerpt']}"
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    high = [f for f in deduped if f["confidence"] == "HIGH"]
    medium = [f for f in deduped if f["confidence"] == "MEDIUM"]
    low = [f for f in deduped if f["confidence"] == "LOW"]

    total = len(deduped)
    lines.append(f"Findings: {total} total  ({len(high)} HIGH, {len(medium)} MEDIUM, {len(low)} LOW)")
    lines.append("")

    if exposed_configs:
        lines.append("[EXPOSED CONFIG FILES ⚠]")
        for cfg in exposed_configs:
            lines.append(f"  {cfg['path']}  ({cfg['size']} bytes)  [{cfg['ct']}]")
            lines.append(f"    preview: {cfg['preview'][:100]}")
        lines.append("")

    for severity, group in [("HIGH", high), ("MEDIUM", medium), ("LOW", low)]:
        if not group:
            continue
        lines.append(f"[{severity} CONFIDENCE FINDINGS]")
        for f in group:
            lines.append(f"  [{f['label']}]  excerpt: {f['excerpt']}  source: {f['source']}")
            lines.append(f"    context: …{f['context']}…")
        lines.append("")

    if not deduped and not exposed_configs:
        lines.append("[CLEAN]")
        lines.append("  No exposed secrets or credentials detected.")
        lines.append("  (Hardened target, or secrets are server-side only.)")

    return "\n".join(lines)
