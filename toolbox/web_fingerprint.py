"""Deep body-layer fingerprinting: CMS versions, JS frameworks, error page analysis.

Complements web_http_probe (which reads headers). This tool reads the response
body — where version info survives even when headers are stripped.
"""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_fingerprint",
        "description": (
            "Deep software + version identification from response bodies. "
            "Goes beyond headers into: HTML <meta name=generator> (CMS versions), "
            "script/link src CDN URLs (jQuery/Bootstrap/React versions), "
            "SPA framework markers (<div id=root>=React, ng-version attr=Angular+version, "
            "__NEXT_DATA__=Next.js, __NUXT__=Nuxt), "
            "deliberate 404 error page analysis (Django/Rails/Spring/Symfony/Laravel/Flask "
            "all leak exact version in error template), "
            "CMS version disclosure endpoints (/readme.html for WordPress, /CHANGELOG.txt "
            "for Drupal, /actuator/info for Spring Boot), "
            "and inline JS version strings. "
            "Returns a structured software inventory with confidence levels. Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL or domain to fingerprint, e.g. 'https://example.com' or 'wordpress.org'",
                },
            },
            "required": ["url"],
        },
    },
}

# Version disclosure paths and what to look for in each
VERSION_PATHS: list[dict[str, Any]] = [
    {"path": "/readme.html",        "grep": r"WordPress\s+([\d.]+)",             "label": "WordPress"},
    {"path": "/readme.txt",         "grep": r"WordPress\s+([\d.]+)",             "label": "WordPress"},
    {"path": "/CHANGELOG.txt",      "grep": r"Drupal\s+([\d.]+)",                "label": "Drupal"},
    {"path": "/CHANGELOG.md",       "grep": r"(?:^#\s*|version\s+)([\d.]+)",     "label": "changelog"},
    {"path": "/version.txt",        "grep": r"([\d]+\.[\d]+\.?[\d]*)",           "label": "version-file"},
    {"path": "/version",            "grep": r"([\d]+\.[\d]+\.?[\d]*)",           "label": "version-endpoint"},
    {"path": "/api/version",        "grep": r"([\d]+\.[\d]+\.?[\d]*)",           "label": "api-version"},
    {"path": "/info",               "grep": r"([\d]+\.[\d]+\.?[\d]*)",           "label": "info-endpoint"},
    {"path": "/build-info",         "grep": r"([\d]+\.[\d]+\.?[\d]*)",           "label": "build-info"},
    {"path": "/actuator/info",      "grep": r"([\d]+\.[\d]+\.?[\d]*)",           "label": "spring-actuator"},
    {"path": "/actuator/env",       "grep": r"spring\.boot\.version.*?([\d.]+)", "label": "spring-boot-ver"},
    {"path": "/administrator/manifests/files/joomla.xml",
                                    "grep": r"<version>([\d.]+)</version>",      "label": "Joomla"},
    {"path": "/.well-known/security.txt", "grep": None,                          "label": "security.txt"},
    {"path": "/package.json",       "grep": r'"version"\s*:\s*"([\d.]+)"',       "label": "package.json"},
    {"path": "/composer.json",      "grep": r'"version"\s*:\s*"([\d.]+)"',       "label": "composer.json"},
]

# CDN URL patterns that contain version numbers
CDN_VERSION_PATTERNS: list[tuple[str, str]] = [
    # (regex pattern on src/href URL, software label)
    (r"jquery[.-]([\d.]+)(?:\.min)?\.js",               "jQuery"),
    (r"bootstrap[@/]([\d.]+)",                          "Bootstrap"),
    (r"react[@/]([\d.]+)",                              "React"),
    (r"vue[@/]([\d.]+)",                                "Vue.js"),
    (r"angular[@/]([\d.]+)",                            "Angular"),
    (r"lodash[@/]([\d.]+)",                             "Lodash"),
    (r"moment[@/]([\d.]+)",                             "Moment.js"),
    (r"axios[@/]([\d.]+)",                              "Axios"),
    (r"fontawesome[@/]([\d.]+)",                        "Font Awesome"),
    (r"tailwindcss[@/]([\d.]+)",                        "Tailwind CSS"),
    (r"alpinejs[@/]([\d.]+)",                           "Alpine.js"),
    (r"htmx[@/]([\d.]+)",                              "HTMX"),
    (r"d3[@/]([\d.]+)",                                 "D3.js"),
    (r"chart\.js[@/]([\d.]+)",                          "Chart.js"),
    (r"three[@/]([\d.]+)",                              "Three.js"),
    (r"socket\.io[@/]([\d.]+)",                         "Socket.IO"),
    (r"svelte[@/]([\d.]+)",                             "Svelte"),
    (r"wordpress[/-]([\d.]+)",                          "WordPress"),
    (r"wp-includes/js/.+[?&]ver=([\d.]+)",              "WordPress (JS ver)"),
    (r"wp-content/.+[?&]ver=([\d.]+)",                  "WordPress (asset ver)"),
]

# SPA / framework markers in HTML body
SPA_MARKERS: list[dict[str, Any]] = [
    {"pattern": r'<div\s+id=["\']root["\']',                          "label": "React (root div)"},
    {"pattern": r'<div\s+id=["\']app["\']',                           "label": "Vue/generic SPA (app div)"},
    {"pattern": r'ng-version=["\']([^"\']+)["\']',                    "label": "Angular", "capture_version": True},
    {"pattern": r'__NEXT_DATA__',                                     "label": "Next.js"},
    {"pattern": r'__nuxt',                                            "label": "Nuxt.js"},
    {"pattern": r'__svelte',                                          "label": "Svelte"},
    {"pattern": r'data-reactroot',                                    "label": "React (SSR)"},
    {"pattern": r'data-vue-meta',                                     "label": "Vue Meta"},
    {"pattern": r'gatsby-',                                           "label": "Gatsby"},
    {"pattern": r'__REMIX_CONTEXT__',                                 "label": "Remix"},
    {"pattern": r'/_astro/',                                          "label": "Astro"},
    {"pattern": r'<html[^>]+ng-app',                                  "label": "AngularJS (v1)"},
    {"pattern": r'Ember\.VERSION\s*=\s*["\']([^"\']+)["\']',         "label": "Ember.js", "capture_version": True},
    {"pattern": r'backbone',                                          "label": "Backbone.js (possible)"},
]

# Error page signatures — probe a 404 and look for these
ERROR_SIGNATURES: list[dict[str, str]] = [
    {"pattern": r"django\.VERSION\s*=|OperationalError at /|DisallowedHost at /|Django Version:\s*([\d.]+)",
     "label": "Django", "version_group": 1},
    {"pattern": r"Whitelabel Error Page|This application has no explicit mapping for /error",
     "label": "Spring Boot"},
    {"pattern": r"spring-boot-starter|Spring Framework|org\.springframework",
     "label": "Spring Framework"},
    {"pattern": r"ActionController::RoutingError|Rails\.version|Ruby on Rails",
     "label": "Ruby on Rails"},
    {"pattern": r"Symfony\s+([\d.]+)|An exception occurred|Symfony Exception",
     "label": "Symfony", "version_group": 1},
    {"pattern": r"Laravel\s+([\d.]+)|laravel/framework|Illuminate\\",
     "label": "Laravel", "version_group": 1},
    {"pattern": r"Werkzeug|Flask\s+([\d.]+)|werkzeug\.([\d.]+)",
     "label": "Flask/Werkzeug", "version_group": 1},
    {"pattern": r"FastAPI|starlette",
     "label": "FastAPI/Starlette"},
    {"pattern": r"Express|Cannot GET|cannot GET",
     "label": "Express.js (Node)"},
    {"pattern": r"ASP\.NET|__VIEWSTATE|System\.Web",
     "label": "ASP.NET"},
    {"pattern": r"Struts|struts\.version|Apache Struts",
     "label": "Apache Struts"},
    {"pattern": r"Grails|groovy\.lang",
     "label": "Grails"},
    {"pattern": r"wp-content|WordPress|twentytwenty|wp-json",
     "label": "WordPress"},
    {"pattern": r"Drupal|drupal\.org|drupal\.behaviors",
     "label": "Drupal"},
    {"pattern": r"Joomla|joomla\.org",
     "label": "Joomla"},
]

# Inline JS version patterns (searched in homepage body)
INLINE_VERSION_PATTERNS: list[tuple[str, str]] = [
    (r'["\']version["\']\s*:\s*["\'](\d+\.\d+[\.\d]*)["\']',     "inline version string"),
    (r'version\s*=\s*["\'](\d+\.\d+[\.\d]*)["\']',               "inline version assignment"),
    (r'React\.version\s*=\s*["\']([^"\']+)["\']',                "React version"),
    (r'Vue\.version\s*=\s*["\']([^"\']+)["\']',                  "Vue version"),
    (r'"buildVersion"\s*:\s*"([^"]+)"',                          "buildVersion"),
    (r'"appVersion"\s*:\s*"([^"]+)"',                            "appVersion"),
    (r'"releaseVersion"\s*:\s*"([^"]+)"',                        "releaseVersion"),
    (r'__APP_VERSION__\s*=\s*["\']([^"\']+)["\']',              "APP_VERSION global"),
    (r'window\.__version__\s*=\s*["\']([^"\']+)["\']',          "window.__version__"),
]


class _MetaParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.generator: str = ""
        self.app_name: str = ""
        self.scripts: list[str] = []
        self.links: list[str] = []
        self.title: str = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        a = dict(attrs)
        if tag == "meta":
            name = (a.get("name") or "").lower()
            if name == "generator":
                self.generator = a.get("content", "")
            elif name == "application-name":
                self.app_name = a.get("content", "")
        elif tag == "script" and a.get("src"):
            self.scripts.append(a["src"])
        elif tag == "link" and a.get("href"):
            self.links.append(a["href"])
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.title:
            self.title = data.strip()[:120]


def _confidence(source: str) -> str:
    if source in ("meta-generator", "readme-file", "changelog", "joomla-xml", "ng-version"):
        return "HIGH"
    if source in ("cdn-url", "inline-js", "version-endpoint", "spring-actuator"):
        return "MEDIUM"
    return "LOW"


def web_fingerprint(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"

    findings: list[dict[str, str]] = []  # {software, version, confidence, source}
    notes: list[str] = []
    spa_hits: list[str] = []
    version_paths_hit: list[str] = []
    error_hits: list[str] = []

    client_kwargs = {"follow_redirects": True, "timeout": 10.0, "verify": False}

    # ── 1. Fetch homepage ────────────────────────────────────────────────────
    homepage_body = ""
    homepage_title = ""
    try:
        with httpx.Client(**client_kwargs) as c:
            r = c.get(base + "/", headers={"User-Agent": "Mozilla/5.0 (compatible; michael-recon/1.0)"})
        homepage_body = r.text[:200_000]
        homepage_title = ""
    except Exception as exc:
        notes.append(f"Homepage fetch failed: {exc}")

    if homepage_body:
        # ── 2. Meta generator ────────────────────────────────────────────────
        parser = _MetaParser()
        parser.feed(homepage_body)
        homepage_title = parser.title

        if parser.generator:
            # Extract version from generator string e.g. "WordPress 6.4.1"
            ver_match = re.search(r"([\d]+\.[\d]+\.?[\d]*)", parser.generator)
            ver = ver_match.group(1) if ver_match else ""
            software = re.sub(r"\s*[\d.]+\s*$", "", parser.generator).strip()
            findings.append({
                "software": software or parser.generator,
                "version": ver,
                "confidence": "HIGH",
                "source": "meta-generator",
            })

        if parser.app_name:
            notes.append(f"meta application-name: {parser.app_name}")

        # ── 3. CDN URL version extraction ─────────────────────────────────
        all_urls = parser.scripts + parser.links
        for asset_url in all_urls:
            for pattern, label in CDN_VERSION_PATTERNS:
                m = re.search(pattern, asset_url, re.IGNORECASE)
                if m:
                    findings.append({
                        "software": label,
                        "version": m.group(1),
                        "confidence": "HIGH",
                        "source": "cdn-url",
                    })
                    break  # one match per asset URL

        # ── 4. SPA / framework markers ────────────────────────────────────
        for marker in SPA_MARKERS:
            m = re.search(marker["pattern"], homepage_body, re.IGNORECASE | re.DOTALL)
            if m:
                label = marker["label"]
                if marker.get("capture_version") and m.lastindex:
                    ver = m.group(1)
                    label = f"{label} {ver}"
                    findings.append({
                        "software": marker["label"].split(" ")[0],
                        "version": ver,
                        "confidence": "HIGH",
                        "source": "ng-version",
                    })
                spa_hits.append(label)

        # ── 7. Inline JS version strings ──────────────────────────────────
        seen_inline: set[str] = set()
        for pattern, label in INLINE_VERSION_PATTERNS:
            for m in re.finditer(pattern, homepage_body):
                ver = m.group(1)
                key = f"{label}:{ver}"
                if key not in seen_inline and re.match(r"\d+\.\d+", ver):
                    seen_inline.add(key)
                    findings.append({
                        "software": label,
                        "version": ver,
                        "confidence": "MEDIUM",
                        "source": "inline-js",
                    })
                    if len(seen_inline) >= 5:
                        break

    # ── 5. Deliberate 404 error page analysis ─────────────────────────────
    try:
        with httpx.Client(**client_kwargs) as c:
            r404 = c.get(
                base + "/michael-recon-fingerprint-probe-xzqq",
                headers={"User-Agent": "Mozilla/5.0 (compatible; michael-recon/1.0)"},
            )
        err_body = r404.text[:30_000]
        for sig in ERROR_SIGNATURES:
            m = re.search(sig["pattern"], err_body, re.IGNORECASE | re.DOTALL)
            if m:
                label = sig["label"]
                ver = ""
                vg = sig.get("version_group")
                if vg and m.lastindex and m.lastindex >= vg:
                    try:
                        ver = m.group(vg) or ""
                    except IndexError:
                        pass
                error_hits.append(f"{label}" + (f" {ver}" if ver else ""))
                if ver:
                    findings.append({
                        "software": label,
                        "version": ver,
                        "confidence": "MEDIUM",
                        "source": "error-page",
                    })
    except Exception as exc:
        notes.append(f"Error page probe failed: {exc}")

    # ── 6. Version disclosure path probing ────────────────────────────────
    with httpx.Client(follow_redirects=False, timeout=6.0, verify=False) as c:
        for vp in VERSION_PATHS:
            path = vp["path"]
            grep = vp["grep"]
            label = vp["label"]
            try:
                r = c.get(base + path)
                if r.status_code == 200:
                    body_snippet = r.text[:5_000]
                    if grep:
                        m = re.search(grep, body_snippet, re.IGNORECASE | re.MULTILINE)
                        if m:
                            ver = m.group(1) if m.lastindex else ""
                            version_paths_hit.append(f"200  {path}  → {label} {ver}".strip())
                            findings.append({
                                "software": label,
                                "version": ver,
                                "confidence": "HIGH",
                                "source": "readme-file" if "readme" in path.lower() or "CHANGELOG" in path else "version-endpoint",
                            })
                        else:
                            version_paths_hit.append(f"200  {path}  (no version match)")
                    else:
                        # Just log that it exists (e.g. security.txt)
                        version_paths_hit.append(f"200  {path}  [{label}]")
            except Exception:
                pass

    # ── Deduplicate findings ─────────────────────────────────────────────
    seen_findings: dict[str, dict[str, str]] = {}
    for f in findings:
        key = (f["software"].lower(), f["version"])
        if key not in seen_findings:
            seen_findings[key] = f
        else:
            # Keep higher confidence
            order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            existing = seen_findings[key]
            if order[f["confidence"]] < order[existing["confidence"]]:
                seen_findings[key] = f
    deduped = list(seen_findings.values())

    # ── Render output ────────────────────────────────────────────────────
    lines: list[str] = [f"=== web_fingerprint: {parsed.netloc} ==="]
    if homepage_title:
        lines.append(f"Page title: {homepage_title}\n")

    lines.append("[SOFTWARE INVENTORY]")
    if deduped:
        for f in deduped:
            ver_str = f" {f['version']}" if f["version"] else ""
            lines.append(f"  {f['software']}{ver_str:<20}  confidence={f['confidence']:<6}  source={f['source']}")
    else:
        lines.append("  Nothing identified (headers-stripped hardened server, or no detectable signals)")

    lines.append("")
    lines.append("[FRAMEWORK / SPA MARKERS]")
    if spa_hits:
        for hit in spa_hits:
            lines.append(f"  {hit}")
    else:
        lines.append("  No SPA framework markers detected (likely server-rendered HTML)")

    if version_paths_hit:
        lines.append("")
        lines.append("[VERSION DISCLOSURE ENDPOINTS]")
        for h in version_paths_hit:
            lines.append(f"  {h}")

    if error_hits:
        lines.append("")
        lines.append("[ERROR PAGE ANALYSIS — framework leaked in 404 response]")
        for h in error_hits:
            lines.append(f"  {h}")
    else:
        lines.append("")
        lines.append("[ERROR PAGE ANALYSIS]")
        lines.append("  No framework signatures in 404 error page")

    if notes:
        lines.append("")
        lines.append("[NOTES]")
        for n in notes:
            lines.append(f"  {n}")

    return "\n".join(lines)
