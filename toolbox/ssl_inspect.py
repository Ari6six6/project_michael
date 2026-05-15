"""Global tool: ssl_inspect — TLS certificate and cipher suite analysis."""
from __future__ import annotations
from typing import Any

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ssl_inspect",
        "description": (
            "Inspect TLS/SSL certificate and cipher suite for a host. Returns subject, issuer, "
            "all Subject Alternative Names (SANs), validity window, serial number, cipher suite, "
            "and TLS version. SANs often reveal additional hosts and internal names. "
            "Passive — standard TLS handshake, no probing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "host": {"type": "string", "description": "Hostname or IP address."},
                "port": {"type": "integer", "description": "Port number. Default 443."},
            },
            "required": ["host"],
        },
    },
}


def ssl_inspect(host: str, port: int = 443, **kwargs: Any) -> str:
    import ssl
    import socket
    host = host.strip()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE  # inspect even self-signed certs
    try:
        raw_sock = socket.create_connection((host, port), timeout=10)
        with ctx.wrap_socket(raw_sock, server_hostname=host) as tls:
            cert = tls.getpeercert()
            cipher = tls.cipher()
            tls_version = tls.version()
    except Exception as e:
        return f"ssl_inspect error: {e}"

    def _flatten(rdns):
        return {k: v for pair in rdns for k, v in pair}

    subject = _flatten(cert.get("subject", []))
    issuer = _flatten(cert.get("issuer", []))
    sans = cert.get("subjectAltName", [])

    lines = [
        f"Host:       {host}:{port}",
        f"TLS:        {tls_version}",
        f"Cipher:     {cipher[0] if cipher else '?'}",
        f"",
        f"Subject:    {subject}",
        f"Issuer:     {issuer}",
        f"Not Before: {cert.get('notBefore', '?')}",
        f"Not After:  {cert.get('notAfter', '?')}",
        f"Serial:     {cert.get('serialNumber', '?')}",
        f"",
        f"Subject Alternative Names ({len(sans)}):",
    ]
    for san_type, san_val in sans:
        lines.append(f"  {san_type:<5} {san_val}")

    return "\n".join(lines)
