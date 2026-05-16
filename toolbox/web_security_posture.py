"""Security posture analysis: headers, WAF detection, cookie flags, sensitive paths.

Grades the target's security maturity. Missing security headers, lax cookie flags,
WAF presence/absence, and exposed sensitive paths all contribute to the picture
of how seriously this target takes security — which affects how to interact with it.
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_security_posture",
        "description": (
            "Assess a target's security posture. Checks: "
            "(1) Security headers — CSP, HSTS, X-Frame-Options, X-Content-Type-Options, "
            "Referrer-Policy, Permissions-Policy — with grade A-F; "
            "(2) CSP policy parsing — which origins are whitelisted (reveals CDN providers, "
            "payment processors, analytics, third-party scripts); "
            "(3) Cookie security flags — Secure, HttpOnly, SameSite for each cookie; "
            "(4) WAF detection — sends probe payloads and reads response signatures "
            "(Cloudflare, AWS WAF, Akamai, ModSecurity, Imperva, F5, Sucuri); "
            "(5) Sensitive path exposure — /.git/HEAD, /.git/config, /phpinfo.php, "
            "/server-status, /trace, TRACE method, OPTIONS method; "
            "(6) HTTP method enumeration. "
            "Returns security grade and actionable findings. Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL or domain to assess",
                },
            },
            "required": ["url"],
        },
    },
}

SECURITY_HEADERS = [
    {
        "header": "strict-transport-security",
        "label": "HSTS",
        "weight": 2,
        "check": lambda v: "max-age" in v.lower(),
        "good": "max-age present",
        "bad": "missing or no max-age",
    },
    {
        "header": "content-security-policy",
        "label": "CSP",
        "weight": 3,
        "check": lambda v: bool(v),
        "good": "present",
        "bad": "missing — XSS risk",
    },
    {
        "header": "x-frame-options",
        "label": "X-Frame-Options",
        "weight": 1,
        "check": lambda v: v.upper() in ("DENY", "SAMEORIGIN"),
        "good": "DENY or SAMEORIGIN",
        "bad": "missing — clickjacking risk",
    },
    {
        "header": "x-content-type-options",
        "label": "X-Content-Type-Options",
        "weight": 1,
        "check": lambda v: "nosniff" in v.lower(),
        "good": "nosniff",
        "bad": "missing — MIME sniffing risk",
    },
    {
        "header": "referrer-policy",
        "label": "Referrer-Policy",
        "weight": 1,
        "check": lambda v: bool(v) and "unsafe-url" not in v.lower(),
        "good": "set and safe",
        "bad": "missing or unsafe-url",
    },
    {
        "header": "permissions-policy",
        "label": "Permissions-Policy",
        "weight": 1,
        "check": lambda v: bool(v),
        "good": "present",
        "bad": "missing",
    },
    {
        "header": "cross-origin-opener-policy",
        "label": "COOP",
        "weight": 1,
        "check": lambda v: bool(v),
        "good": "present",
        "bad": "missing",
    },
    {
        "header": "cross-origin-embedder-policy",
        "label": "COEP",
        "weight": 1,
        "check": lambda v: bool(v),
        "good": "present",
        "bad": "missing",
    },
]

WAF_SIGNATURES: list[tuple[str, str]] = [
    # (pattern matched in headers+body, WAF name)
    ("cf-ray",                          "Cloudflare WAF"),
    ("__cfduid",                        "Cloudflare"),
    ("x-amzn-requestid.*403",           "AWS WAF"),
    ("x-amzn-waf",                      "AWS WAF"),
    ("akamai",                          "Akamai WAF"),
    ("x-akamai",                        "Akamai"),
    ("incap_ses",                       "Imperva Incapsula"),
    ("visid_incap",                     "Imperva Incapsula"),
    ("x-sucuri",                        "Sucuri WAF"),
    ("sucuri website firewall",         "Sucuri WAF"),
    ("mod_security",                    "ModSecurity"),
    ("noyb",                            "ModSecurity (NOYB)"),
    ("TS[0-9a-f]{8}",                   "F5 BIG-IP ASM"),
    ("barracuda",                       "Barracuda WAF"),
    ("reblaze",                         "Reblaze WAF"),
    ("x-powered-by-plesk",              "Plesk"),
    ("perimeterx",                      "PerimeterX"),
    ("datadome",                        "DataDome (bot protection)"),
    ("x-cdn: imperva",                  "Imperva"),
    ("fortigate",                       "FortiGate WAF"),
    ("bigip",                           "F5 BIG-IP"),
]

SENSITIVE_PATHS = [
    "/.git/HEAD",
    "/.git/config",
    "/.git/COMMIT_EDITMSG",
    "/.svn/entries",
    "/.svn/wc.db",
    "/phpinfo.php",
    "/info.php",
    "/server-status",
    "/server-info",
    "/_profiler",
    "/_profiler/phpinfo",
    "/actuator",
    "/actuator/health",
    "/actuator/env",
    "/actuator/mappings",
    "/actuator/beans",
    "/jolokia",
    "/solr/admin",
    "/wp-cron.php",
    "/xmlrpc.php",
    "/elmah.axd",
    "/trace.axd",
    "/glimpse/config",
    "/telescope/requests",
    "/__debug__/",
    "/debug/default/view",
    "/.DS_Store",
    "/Thumbs.db",
    "/backup.sql",
    "/backup.zip",
    "/backup.tar.gz",
    "/dump.sql",
    "/database.sql",
]


def _parse_csp(csp: str) -> dict[str, list[str]]:
    """Parse a CSP header into directive → sources dict."""
    directives: dict[str, list[str]] = {}
    for directive in csp.split(";"):
        parts = directive.strip().split()
        if not parts:
            continue
        name = parts[0].lower()
        sources = parts[1:]
        directives[name] = sources
    return directives


def _waf_check(headers_str: str, body: str) -> list[str]:
    """Detect WAF from combined headers string and response body."""
    combined = (headers_str + " " + body[:5000]).lower()
    detected: list[str] = []
    for pattern, label in WAF_SIGNATURES:
        if re.search(pattern, combined, re.IGNORECASE) and label not in detected:
            detected.append(label)
    return detected


def web_security_posture(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    lines: list[str] = [f"=== web_security_posture: {parsed.netloc} ===\n"]

    client_kwargs = {"follow_redirects": True, "timeout": 10.0, "verify": False}

    # ── 1. Fetch homepage ─────────────────────────────────────────────────
    try:
        with httpx.Client(**client_kwargs) as c:
            r = c.get(url)
        resp_headers = {k.lower(): v for k, v in r.headers.items()}
        homepage_body = r.text[:20_000]
    except Exception as exc:
        lines.append(f"ERROR: {exc}")
        return "\n".join(lines)

    # ── 2. Security header grading ────────────────────────────────────────
    score = 0
    max_score = sum(h["weight"] for h in SECURITY_HEADERS)
    header_results: list[dict] = []

    for hdef in SECURITY_HEADERS:
        val = resp_headers.get(hdef["header"], "")
        passed = hdef["check"](val) if val else False
        if passed:
            score += hdef["weight"]
        header_results.append({
            "label": hdef["label"],
            "value": val[:120] if val else "",
            "passed": passed,
            "note": hdef["good"] if passed else hdef["bad"],
        })

    pct = int(score / max_score * 100)
    grade = "A" if pct >= 90 else "B" if pct >= 75 else "C" if pct >= 60 else "D" if pct >= 40 else "F"

    lines.append(f"[SECURITY HEADERS — Grade: {grade} ({pct}%)]")
    for hr in header_results:
        status = "✓" if hr["passed"] else "✗"
        val_str = f"  → {hr['value']}" if hr["value"] else ""
        lines.append(f"  {status}  {hr['label']:<30}  {hr['note']}{val_str[:80]}")
    lines.append("")

    # ── 3. CSP analysis ───────────────────────────────────────────────────
    csp_val = resp_headers.get("content-security-policy", "")
    if csp_val:
        directives = _parse_csp(csp_val)
        lines.append("[CSP ANALYSIS]")

        # Dangerous directives
        if "unsafe-inline" in csp_val:
            lines.append("  ⚠  unsafe-inline present — XSS protection weakened")
        if "unsafe-eval" in csp_val:
            lines.append("  ⚠  unsafe-eval present — code injection risk")
        if "'*'" in csp_val or "* " in csp_val:
            lines.append("  ⚠  wildcard (*) source found")

        # Extract whitelisted origins — reveals third-party dependencies
        origins: set[str] = set()
        for sources in directives.values():
            for src in sources:
                if src.startswith(("http://", "https://", "//")):
                    try:
                        host = urlparse(src if "//" in src else "https:" + src).netloc
                        if host:
                            origins.add(host)
                    except Exception:
                        pass
                elif re.match(r'^[a-z0-9][a-z0-9.-]+\.[a-z]{2,}$', src):
                    origins.add(src)

        if origins:
            lines.append(f"  Whitelisted origins ({len(origins)} — reveals 3rd-party integrations):")
            for o in sorted(origins)[:20]:
                lines.append(f"    {o}")
        lines.append("")

    # ── 4. Cookie security ────────────────────────────────────────────────
    cookie_headers = r.headers.get_list("set-cookie")
    if cookie_headers:
        lines.append("[COOKIE SECURITY FLAGS]")
        for ch in cookie_headers:
            name = ch.split("=")[0].strip()
            has_secure = "secure" in ch.lower()
            has_httponly = "httponly" in ch.lower()
            samesite = re.search(r'samesite=(\w+)', ch, re.IGNORECASE)
            samesite_val = samesite.group(1) if samesite else "not set"
            flags = []
            if not has_secure:
                flags.append("⚠ missing Secure")
            if not has_httponly:
                flags.append("⚠ missing HttpOnly")
            if samesite_val == "not set":
                flags.append("⚠ SameSite not set")
            status = "  ".join(flags) if flags else "OK"
            lines.append(f"  {name:<30}  Secure={'Y' if has_secure else 'N'}  "
                         f"HttpOnly={'Y' if has_httponly else 'N'}  "
                         f"SameSite={samesite_val}  {status}")
        lines.append("")

    # ── 5. WAF detection ──────────────────────────────────────────────────
    lines.append("[WAF / BOT PROTECTION]")
    # Check baseline response
    headers_str = " ".join(f"{k}:{v}" for k, v in resp_headers.items())
    waf_detected = _waf_check(headers_str, homepage_body)

    # Send a probe request with XSS/SQLi payload to trigger WAF
    probe_wafs: list[str] = []
    try:
        probe_url = base + "/?__probe__=<script>alert(1)</script>%27%20OR%201=1--"
        with httpx.Client(follow_redirects=False, timeout=6.0, verify=False) as c:
            rp = c.get(probe_url)
        probe_headers_str = " ".join(f"{k}:{v}" for k, v in rp.headers.items())
        probe_wafs = _waf_check(probe_headers_str, rp.text[:5000])
        if rp.status_code in (403, 406, 429, 503) and rp.status_code != r.status_code:
            lines.append(f"  Probe returned {rp.status_code} (baseline was {r.status_code}) — WAF active")
    except Exception:
        pass

    all_wafs = list({*waf_detected, *probe_wafs})
    if all_wafs:
        for waf in all_wafs:
            lines.append(f"  Detected: {waf}")
    else:
        lines.append("  No WAF signature detected")
    lines.append("")

    # ── 6. HTTP method enumeration ────────────────────────────────────────
    lines.append("[HTTP METHODS]")
    try:
        with httpx.Client(follow_redirects=False, timeout=6.0, verify=False) as c:
            opts = c.options(base + "/")
        allow = opts.headers.get("allow", opts.headers.get("access-control-allow-methods", ""))
        if allow:
            lines.append(f"  OPTIONS Allow: {allow}")
            if "TRACE" in allow.upper():
                lines.append("  ⚠  TRACE method allowed — XST (cross-site tracing) risk")
            if "PUT" in allow.upper() or "DELETE" in allow.upper():
                lines.append("  ⚠  PUT/DELETE allowed — check if auth is required")
        else:
            lines.append("  OPTIONS returned no Allow header")
    except Exception:
        lines.append("  OPTIONS method blocked or timed out")
    lines.append("")

    # ── 7. Sensitive path probe ───────────────────────────────────────────
    lines.append("[SENSITIVE PATH EXPOSURE]")
    exposed: list[dict] = []
    with httpx.Client(follow_redirects=False, timeout=5.0, verify=False) as c:
        # Establish baseline: probe a known-nonexistent path to get 404 size
        baseline_size = 0
        try:
            b404 = c.get(base + "/michael-recon-baseline-xyzq404")
            if b404.status_code in (200, 404):
                baseline_size = len(b404.content)
        except Exception:
            pass

        for path in SENSITIVE_PATHS:
            try:
                rp = c.get(base + path)
                if rp.status_code != 200 or len(rp.content) == 0:
                    continue
                ct = rp.headers.get("content-type", "")
                is_html = "text/html" in ct.lower()
                size = len(rp.content)

                # Skip if size is within 10% of baseline (custom 404 catch-all)
                if baseline_size > 0 and abs(size - baseline_size) / baseline_size < 0.10:
                    continue

                preview = rp.text[:80].replace("\n", " ")
                # For HTML responses, require specific keywords to avoid false positives
                content_keywords = [
                    "ref:", "phpinfo", "php version", "git object", "git repository",
                    "server version", "processor", "apache status", "svn", "subversion",
                    "backup", "database", "schema", "username", "password", "private key",
                    "actuator", "jolokia", "spring", "heap", "classloading", "[database]",
                    "datasource", "mongodb", "redis", "secret", "credential",
                ]
                looks_real = (not is_html) or any(kw in rp.text.lower() for kw in content_keywords)
                if looks_real:
                    exposed.append({"path": path, "status": rp.status_code,
                                    "size": size, "preview": preview})
            except Exception:
                pass

    if exposed:
        for e in exposed:
            lines.append(f"  ⚠  {e['path']}  ({e['size']} bytes)")
            lines.append(f"     preview: {e['preview'][:80]}")
    else:
        lines.append("  No sensitive paths exposed")

    return "\n".join(lines)
