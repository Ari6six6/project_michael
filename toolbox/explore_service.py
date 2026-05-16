"""Master exploration tool: full OS-upward recon pipeline for a target domain/URL.

Runs in sequence: TCP port scan → banner grab → DNS → TLS → subdomain discovery
→ HTTP fingerprint → common path probe → homepage link extraction
→ body-layer software fingerprinting (CMS versions, JS frameworks, error pages).
Always runs the full pipeline — no depth switch.
"""
from __future__ import annotations

import socket
import ssl
import threading
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "explore_service",
        "description": (
            "Full OS-upward recon pipeline starting from a domain or URL. "
            "Runs: TCP port scan, banner grabbing, DNS records (via DoH), "
            "TLS deep analysis (cipher, ALPN, cert SANs), common subdomain probing, "
            "HTTP fingerprinting (stack detection from headers/cookies), "
            "well-known path probing (APIs, admin, health, OpenAPI), and "
            "homepage link extraction. Returns a structured text report the LLM "
            "can use to build a mental model of the target service. "
            "Always deep — never truncated. Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Domain name or URL, e.g. 'example.com' or 'https://api.example.com'",
                },
            },
            "required": ["target"],
        },
    },
}

SCAN_PORTS = [
    80, 443, 8080, 8443, 22, 21, 25, 3306, 5432,
    6379, 27017, 9200, 9300, 5601, 4443, 8000, 8001,
    8888, 9000, 3000, 4000, 5000, 7000, 11211, 2181,
]

COMMON_PATHS = [
    "/", "/robots.txt", "/sitemap.xml", "/.well-known/security.txt",
    "/.well-known/openid-configuration", "/.well-known/oauth-authorization-server",
    "/health", "/healthz", "/ping", "/ready", "/status", "/metrics",
    "/api", "/api/v1", "/api/v2", "/v1", "/v2", "/v3",
    "/swagger.json", "/swagger/v1/swagger.json", "/swagger-ui.html",
    "/openapi.json", "/openapi.yaml", "/api-docs", "/api-docs.json",
    "/graphql", "/graphiql",
    "/admin", "/admin/login", "/login", "/auth", "/auth/login",
    "/oauth/token", "/oauth2/token", "/connect/token",
    "/.env", "/config.json", "/wp-json", "/wp-login.php",
]

COMMON_SUBDOMAINS = [
    "www", "api", "mail", "admin", "dev", "staging", "app",
    "auth", "cdn", "static", "assets", "media", "api-v2",
    "beta", "internal", "dashboard", "portal",
]

DOH_URL = "https://cloudflare-dns.com/dns-query"
DNS_TYPES = ["A", "AAAA", "MX", "TXT", "NS", "CNAME"]

CDN_SIGNALS = {
    "cloudflare": ["cf-ray", "cf-cache-status", "server:cloudflare"],
    "fastly": ["x-served-by", "fastly"],
    "akamai": ["x-akamai", "akamai"],
    "cloudfront": ["x-amz-cf-id", "cloudfront"],
    "varnish": ["via:varnish", "x-varnish"],
}

FRAMEWORK_COOKIES = {
    "PHPSESSID": "PHP",
    "JSESSIONID": "Java/JVM",
    "ASP.NET_SessionId": "ASP.NET",
    "_rails": "Ruby on Rails",
    "laravel_session": "Laravel (PHP)",
    "csrftoken": "Django (Python)",
    "rack.session": "Rack/Ruby",
    "connect.sid": "Express.js (Node)",
}


def _normalize_domain(target: str) -> tuple[str, str]:
    """Return (domain, base_url) from a domain or URL."""
    if not target.startswith(("http://", "https://")):
        target = "https://" + target
    parsed = urlparse(target)
    domain = parsed.netloc.split(":")[0]
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    return domain, base_url


def _tcp_scan(domain: str) -> tuple[list[int], dict[int, str]]:
    """Connect-scan SCAN_PORTS; return (open_ports, banners)."""
    open_ports: list[int] = []
    banners: dict[int, str] = {}
    lock = threading.Lock()

    def probe(port: int) -> None:
        try:
            with socket.create_connection((domain, port), timeout=2) as s:
                with lock:
                    open_ports.append(port)
                try:
                    s.settimeout(1.5)
                    raw = s.recv(1024)
                    banner = raw.decode("utf-8", errors="replace").strip()
                    if banner:
                        with lock:
                            banners[port] = banner[:300]
                except Exception:
                    pass
        except Exception:
            pass

    threads = [threading.Thread(target=probe, args=(p,)) for p in SCAN_PORTS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return sorted(open_ports), banners


def _tls_info(domain: str, port: int = 443) -> dict[str, Any]:
    """Extract TLS metadata: version, cipher, ALPN, cert chain SANs."""
    result: dict[str, Any] = {"error": None}
    for verify in (True, False):
        ctx = ssl.create_default_context()
        ctx.set_alpn_protocols(["h2", "http/1.1"])
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_OPTIONAL if verify else ssl.CERT_NONE
        try:
            with socket.create_connection((domain, port), timeout=5) as raw:
                with ctx.wrap_socket(raw, server_hostname=domain) as s:
                    result["tls_version"] = s.version()
                    result["cipher"] = s.cipher()[0] if s.cipher() else None
                    result["alpn"] = s.selected_alpn_protocol()
                    cert = s.getpeercert() or {}
                    result["subject_cn"] = dict(x[0] for x in cert.get("subject", [])).get("commonName")
                    result["issuer"] = dict(x[0] for x in cert.get("issuer", [])).get("organizationName")
                    result["valid_from"] = cert.get("notBefore")
                    result["valid_to"] = cert.get("notAfter")
                    result["sans"] = [v for t, v in cert.get("subjectAltName", []) if t == "DNS"]
            return result
        except ssl.SSLError:
            if not verify:
                result["error"] = "SSL error (self-signed or expired cert)"
                return result
            continue
        except Exception as exc:
            result["error"] = str(exc)
            return result
    return result


def _dns_records(domain: str) -> dict[str, list[str]]:
    """Query DNS via Cloudflare DoH."""
    records: dict[str, list[str]] = {}
    try:
        with httpx.Client(timeout=8.0) as client:
            for rtype in DNS_TYPES:
                try:
                    r = client.get(
                        DOH_URL,
                        params={"name": domain, "type": rtype, "ct": "application/dns-json"},
                        headers={"Accept": "application/dns-json"},
                    )
                    data = r.json()
                    records[rtype] = [a["data"] for a in data.get("Answer", [])]
                except Exception:
                    records[rtype] = []
    except Exception:
        pass
    return records


def _subdomain_probe(domain: str) -> list[str]:
    """DNS-resolve common subdomains; return ones that resolve."""
    found: list[str] = []
    base = domain.split(".", 1)[-1] if domain.count(".") > 1 else domain
    parent = domain if domain.count(".") == 1 else base
    with httpx.Client(timeout=6.0) as client:
        for prefix in COMMON_SUBDOMAINS:
            sub = f"{prefix}.{parent}"
            try:
                r = client.get(
                    DOH_URL,
                    params={"name": sub, "type": "A", "ct": "application/dns-json"},
                    headers={"Accept": "application/dns-json"},
                )
                data = r.json()
                if data.get("Answer"):
                    found.append(sub)
            except Exception:
                pass
    return found


def _http_fingerprint(domain: str) -> dict[str, Any]:
    """Probe http + https, collect headers, detect stack."""
    result: dict[str, Any] = {"scheme": None, "status": None, "final_url": None,
                               "elapsed_ms": None, "headers": {}, "cdns": [],
                               "framework_hints": [], "cookies": []}
    client_kwargs = {"follow_redirects": True, "timeout": 10.0, "verify": False}
    for scheme in ("https", "http"):
        try:
            with httpx.Client(**client_kwargs) as c:
                r = c.get(f"{scheme}://{domain}/")
            result["scheme"] = scheme
            result["status"] = r.status_code
            result["final_url"] = str(r.url)
            result["elapsed_ms"] = round(r.elapsed.total_seconds() * 1000)
            headers_lower = {k.lower(): v for k, v in r.headers.items()}
            result["headers"] = dict(r.headers)

            # CDN detection
            for cdn, signals in CDN_SIGNALS.items():
                for sig in signals:
                    if ":" in sig:
                        hdr, val = sig.split(":", 1)
                        if val in headers_lower.get(hdr, "").lower():
                            result["cdns"].append(cdn)
                            break
                    else:
                        if any(sig in k for k in headers_lower):
                            result["cdns"].append(cdn)
                            break

            # Framework from headers
            for h in ("server", "x-powered-by", "x-generator", "x-aspnet-version",
                      "x-aspnetmvc-version", "x-runtime"):
                if h in headers_lower:
                    result["framework_hints"].append(f"{h}: {headers_lower[h]}")

            # Framework from cookies
            for cookie_header in r.headers.get_list("set-cookie"):
                name = cookie_header.split("=")[0].strip()
                result["cookies"].append(name)
                if name in FRAMEWORK_COOKIES:
                    result["framework_hints"].append(f"cookie:{name} → {FRAMEWORK_COOKIES[name]}")
            break
        except Exception:
            continue
    return result


def _path_probe(domain: str, scheme: str) -> list[dict[str, Any]]:
    """Probe COMMON_PATHS; return status codes and content types."""
    base = f"{scheme}://{domain}"
    results: list[dict[str, Any]] = []
    with httpx.Client(follow_redirects=False, timeout=6.0, verify=False) as client:
        for path in COMMON_PATHS:
            try:
                r = client.get(base + path)
                results.append({
                    "path": path,
                    "status": r.status_code,
                    "ct": r.headers.get("content-type", "")[:60],
                    "size": int(r.headers.get("content-length", len(r.content))),
                    "location": r.headers.get("location", "") if r.status_code in (301, 302, 307, 308) else "",
                })
            except Exception:
                pass
    return results


def _extract_links(domain: str, scheme: str) -> list[str]:
    """Grab homepage and extract <a href> links."""
    from html.parser import HTMLParser

    class _LinkParser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.links: list[str] = []

        def handle_starttag(self, tag: str, attrs: list) -> None:
            if tag == "a":
                for k, v in attrs:
                    if k == "href" and v:
                        self.links.append(v)

    try:
        with httpx.Client(follow_redirects=True, timeout=10.0, verify=False) as c:
            r = c.get(f"{scheme}://{domain}/")
        parser = _LinkParser()
        parser.feed(r.text[:50_000])
        base = f"{scheme}://{domain}"
        seen: set[str] = set()
        links: list[str] = []
        for href in parser.links:
            full = urljoin(base, href)
            if full not in seen:
                seen.add(full)
                links.append(full)
        return links[:80]
    except Exception:
        return []


def explore_service(target: str) -> str:
    """Full OS-upward recon pipeline. Returns a structured text report."""
    domain, _base_url = _normalize_domain(target)
    lines: list[str] = [f"=== explore_service: {domain} ===\n"]

    # 1. TCP port scan + banner grab
    lines.append("[TCP PORT SCAN]")
    open_ports, banners = _tcp_scan(domain)
    if open_ports:
        lines.append(f"  Open ports: {open_ports}")
        for port, banner in sorted(banners.items()):
            first_line = banner.splitlines()[0][:120] if banner else ""
            lines.append(f"  port {port} banner: {first_line}")
    else:
        lines.append("  No open ports found (all filtered or host unreachable)")
    lines.append("")

    # 2. DNS records
    lines.append("[DNS]")
    dns = _dns_records(domain)
    for rtype, vals in dns.items():
        if vals:
            lines.append(f"  {rtype:<6} → {', '.join(vals[:5])}")
    lines.append("")

    # 3. TLS analysis
    tls_port = 443 if 443 in open_ports else (8443 if 8443 in open_ports else None)
    if tls_port:
        lines.append(f"[TLS — port {tls_port}]")
        tls = _tls_info(domain, tls_port)
        if tls.get("error"):
            lines.append(f"  error: {tls['error']}")
        else:
            lines.append(f"  version : {tls.get('tls_version')}")
            lines.append(f"  cipher  : {tls.get('cipher')}")
            lines.append(f"  ALPN    : {tls.get('alpn')} (h2 = HTTP/2 supported)")
            lines.append(f"  subject : {tls.get('subject_cn')}")
            lines.append(f"  issuer  : {tls.get('issuer')}")
            lines.append(f"  valid   : {tls.get('valid_from')} → {tls.get('valid_to')}")
            sans = tls.get("sans", [])
            if sans:
                lines.append(f"  SANs    : {', '.join(sans[:20])}")
                if len(sans) > 20:
                    lines.append(f"            … and {len(sans) - 20} more")
        lines.append("")

    # 4. Subdomain discovery
    lines.append("[SUBDOMAIN DISCOVERY]")
    subs = _subdomain_probe(domain)
    if subs:
        lines.append(f"  Resolving subdomains: {', '.join(subs)}")
    else:
        lines.append("  No common subdomains resolved")
    lines.append("")

    # 5. HTTP fingerprint
    lines.append("[HTTP FINGERPRINT]")
    fp = _http_fingerprint(domain)
    if fp["scheme"]:
        lines.append(f"  {fp['scheme'].upper():5} → {fp['status']} ({fp['elapsed_ms']}ms)")
        lines.append(f"  final URL : {fp['final_url']}")
        if fp["cdns"]:
            lines.append(f"  CDN       : {', '.join(set(fp['cdns']))}")
        if fp["framework_hints"]:
            for hint in fp["framework_hints"]:
                lines.append(f"  stack     : {hint}")
        if fp["cookies"]:
            lines.append(f"  cookies   : {', '.join(fp['cookies'])}")
        for h in ("server", "x-powered-by", "content-type", "x-frame-options",
                  "strict-transport-security", "access-control-allow-origin"):
            val = fp["headers"].get(h) or fp["headers"].get(h.title())
            if val:
                lines.append(f"  {h}: {val[:120]}")
    else:
        lines.append("  HTTP unreachable")
    lines.append("")

    # 6. Common path probe
    scheme = fp.get("scheme") or "https"
    lines.append("[PATH PROBE]")
    path_results = _path_probe(domain, scheme)
    interesting = [p for p in path_results if p["status"] in (200, 201, 204, 301, 302, 307, 401, 403)]
    for p in interesting:
        loc = f" → {p['location']}" if p["location"] else ""
        ct = f" [{p['ct']}]" if p["ct"] else ""
        lines.append(f"  {p['status']}  {p['path']}{loc}{ct}")
    if not interesting:
        lines.append("  All probed paths returned 404/error")
    lines.append("")

    # 7. Homepage links
    lines.append("[HOMEPAGE LINKS]")
    links = _extract_links(domain, scheme)
    if links:
        same_domain = [l for l in links if domain in l]
        external = [l for l in links if domain not in l]
        lines.append(f"  Same-domain links ({len(same_domain)}):")
        for l in same_domain[:20]:
            lines.append(f"    {l}")
        if external:
            lines.append(f"  External links ({len(external)}) — sample:")
            for l in external[:5]:
                lines.append(f"    {l}")
    else:
        lines.append("  No links extracted")

    def _embed(section_name: str, report: str) -> None:
        lines.append("")
        lines.append(f"[{section_name}]")
        for fl in report.splitlines()[1:]:  # skip "=== tool: host ===" header
            lines.append("  " + fl)

    # 8. IP intelligence
    try:
        from toolbox.web_ip_intel import web_ip_intel as _ipi
        _embed("IP INTELLIGENCE", _ipi(domain))
    except Exception as exc:
        lines.append(f"\n[IP INTELLIGENCE]\n  (failed: {exc})")

    # 9. Body-layer software fingerprinting
    try:
        from toolbox.web_fingerprint import web_fingerprint as _wf
        _embed("BODY FINGERPRINT", _wf(f"{scheme}://{domain}"))
    except Exception as exc:
        lines.append(f"\n[BODY FINGERPRINT]\n  (failed: {exc})")

    # 10. Security posture
    try:
        from toolbox.web_security_posture import web_security_posture as _wsp
        _embed("SECURITY POSTURE", _wsp(f"{scheme}://{domain}"))
    except Exception as exc:
        lines.append(f"\n[SECURITY POSTURE]\n  (failed: {exc})")

    # 11. JS bundle analysis (if it looks like a web app)
    try:
        from toolbox.web_js_analyze import web_js_analyze as _wja
        _embed("JS BUNDLE ANALYSIS", _wja(f"{scheme}://{domain}"))
    except Exception as exc:
        lines.append(f"\n[JS BUNDLE ANALYSIS]\n  (failed: {exc})")

    return "\n".join(lines)
