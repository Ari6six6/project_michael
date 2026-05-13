"""TLS handshake analysis: version, cipher, ALPN, certificate chain and SANs."""
from __future__ import annotations

import socket
import ssl

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_tls_info",
        "description": (
            "Connect to a host via TLS and extract: TLS version, negotiated cipher suite, "
            "ALPN protocol (h2 means HTTP/2 is supported; grpc means gRPC), "
            "full certificate chain with Subject Alternative Names (SANs reveal all covered "
            "subdomains), issuer, validity window. "
            "Works on any port (default 443). Uses ssl stdlib — no extra deps. Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "host": {
                    "type": "string",
                    "description": "Hostname to connect to, e.g. 'api.example.com'",
                },
                "port": {
                    "type": "integer",
                    "description": "TCP port (default 443)",
                },
            },
            "required": ["host"],
        },
    },
}


def _tls_connect(host: str, port: int) -> tuple[ssl.SSLSocket, dict]:
    """Try to get TLS connection + cert dict. First try with cert validation
    (gives full cert dict), fall back to no-verify (skips expired/self-signed)."""
    for verify in (True, False):
        ctx = ssl.create_default_context()
        ctx.set_alpn_protocols(["h2", "http/1.1"])
        ctx.check_hostname = False
        if verify:
            ctx.verify_mode = ssl.CERT_OPTIONAL
        else:
            ctx.verify_mode = ssl.CERT_NONE
        try:
            raw = socket.create_connection((host, port), timeout=8)
            s = ctx.wrap_socket(raw, server_hostname=host)
            cert = s.getpeercert() or {}
            return s, cert
        except ssl.SSLError:
            if not verify:
                raise
            continue
    raise RuntimeError("unreachable")


def web_tls_info(host: str, port: int = 443) -> str:
    lines: list[str] = [f"=== web_tls_info: {host}:{port} ===\n"]
    try:
        s, cert = _tls_connect(host, port)
        with s:
            ver = s.version()
            cipher = s.cipher()
            alpn = s.selected_alpn_protocol()

        lines.append(f"TLS version : {ver}")
        lines.append(f"Cipher      : {cipher[0] if cipher else 'unknown'} ({cipher[2] if cipher else '?'} bits)")
        lines.append(f"ALPN        : {alpn or 'not negotiated'}")
        lines.append("")

        subject = dict(x[0] for x in cert.get("subject", []))
        issuer = dict(x[0] for x in cert.get("issuer", []))
        lines.append(f"Subject CN  : {subject.get('commonName', '?')}")
        lines.append(f"Issuer      : {issuer.get('organizationName', '?')} / {issuer.get('commonName', '?')}")
        lines.append(f"Valid from  : {cert.get('notBefore', '?')}")
        lines.append(f"Valid to    : {cert.get('notAfter', '?')}")
        lines.append("")

        sans = [v for t, v in cert.get("subjectAltName", []) if t == "DNS"]
        ip_sans = [v for t, v in cert.get("subjectAltName", []) if t == "IP Address"]
        lines.append(f"[SANs — {len(sans)} DNS name(s)]")
        for san in sans:
            lines.append(f"  {san}")
        if ip_sans:
            lines.append(f"[SANs — IP addresses]")
            for ip in ip_sans:
                lines.append(f"  {ip}")

        lines.append("")
        if alpn == "h2":
            lines.append("NOTE: HTTP/2 supported (ALPN=h2)")
        if "Let's Encrypt" in issuer.get("organizationName", ""):
            lines.append("NOTE: Let's Encrypt cert — modern devops / automated cert management")
        wildcard_sans = [s for s in sans if s.startswith("*")]
        if wildcard_sans:
            lines.append(f"NOTE: Wildcard SANs: {', '.join(wildcard_sans)}")

    except Exception as exc:
        lines.append(f"ERROR: {exc}")

    return "\n".join(lines)
