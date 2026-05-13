"""IP address intelligence: ASN, cloud provider, geolocation, reverse DNS.

Resolves a domain to its IPs, identifies cloud hosting provider from ASN data,
checks for CDN/hosting signatures, and performs reverse DNS lookups.
Knowing the ASN tells you who actually runs the infrastructure — AWS, GCP, Azure,
Cloudflare, Fastly, DigitalOcean, Hetzner, OVH, etc.
"""
from __future__ import annotations

import re
import socket
from typing import Any

import httpx

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_ip_intel",
        "description": (
            "Resolve domain → IP(s) and identify cloud/hosting provider. "
            "Returns: all A/AAAA records, reverse DNS for each IP, ASN number and org "
            "(via ip-api.com — free, no auth), cloud provider detection from ASN "
            "(AWS/GCP/Azure/Cloudflare/Fastly/DigitalOcean/Hetzner/OVH/Linode), "
            "geolocation (country, city, ISP), and whether the IP is a known CDN edge "
            "node (meaning the real origin is hidden). "
            "Multiple IPs = load balancer or CDN anycast. "
            "Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Domain to look up, e.g. 'example.com' or 'api.example.com'",
                },
            },
            "required": ["domain"],
        },
    },
}

# ASN → cloud provider mapping (partial — covers the major players)
# Format: (regex on org name, provider label)
CLOUD_ASN_PATTERNS: list[tuple[str, str]] = [
    (r"amazon|aws|ec2",                         "AWS (Amazon Web Services)"),
    (r"google|gcp|googleapis",                  "GCP (Google Cloud)"),
    (r"microsoft|azure",                        "Microsoft Azure"),
    (r"cloudflare",                              "Cloudflare"),
    (r"fastly",                                  "Fastly CDN"),
    (r"akamai",                                  "Akamai"),
    (r"digitalocean",                            "DigitalOcean"),
    (r"hetzner",                                 "Hetzner"),
    (r"ovh",                                     "OVH"),
    (r"linode|akamai.*linode",                  "Linode (Akamai)"),
    (r"vultr",                                   "Vultr"),
    (r"rackspace",                               "Rackspace"),
    (r"leaseweb",                                "LeaseWeb"),
    (r"contabo",                                 "Contabo"),
    (r"oracle|oraclecloud",                     "Oracle Cloud"),
    (r"ibm|softlayer",                          "IBM Cloud"),
    (r"alibaba|aliyun",                         "Alibaba Cloud"),
    (r"tencent",                                 "Tencent Cloud"),
    (r"heroku",                                  "Heroku"),
    (r"netlify",                                 "Netlify"),
    (r"vercel",                                  "Vercel"),
    (r"github",                                  "GitHub Pages"),
    (r"render",                                  "Render"),
    (r"fly\.io",                                 "Fly.io"),
    (r"railway",                                 "Railway"),
]

# Known CDN ASN numbers (anycast) — if IP resolves to these, origin is hidden
CDN_ASNS = {
    13335: "Cloudflare",
    20940: "Akamai",
    54113: "Fastly",
    16509: "AWS CloudFront",
    15169: "Google Cloud/CDN",
    36459: "GitHub",
    14618: "AWS",
    16276: "OVH",
}


def _resolve_ips(domain: str) -> tuple[list[str], list[str]]:
    """Return (ipv4_list, ipv6_list) for a domain."""
    ipv4: list[str] = []
    ipv6: list[str] = []
    try:
        results = socket.getaddrinfo(domain, None)
        for family, _, _, _, addr in results:
            ip = addr[0]
            if family == socket.AF_INET and ip not in ipv4:
                ipv4.append(ip)
            elif family == socket.AF_INET6 and ip not in ipv6:
                ipv6.append(ip)
    except Exception:
        pass
    return ipv4, ipv6


def _reverse_dns(ip: str) -> str:
    try:
        return socket.gethostbyaddr(ip)[0]
    except Exception:
        return ""


def _ip_info(ip: str) -> dict[str, Any]:
    """Query ip-api.com for ASN, org, country, city."""
    try:
        with httpx.Client(timeout=8.0) as c:
            r = c.get(
                f"http://ip-api.com/json/{ip}",
                params={"fields": "status,country,regionName,city,isp,org,as,asname,hosting,proxy,mobile"},
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return {}


def _detect_cloud(org: str, asname: str, asn_num: int) -> str:
    combined = f"{org} {asname}".lower()
    for pattern, label in CLOUD_ASN_PATTERNS:
        if re.search(pattern, combined, re.IGNORECASE):
            return label
    if asn_num in CDN_ASNS:
        return CDN_ASNS[asn_num]
    return ""


def web_ip_intel(domain: str) -> str:
    lines: list[str] = [f"=== web_ip_intel: {domain} ===\n"]

    # Resolve
    ipv4, ipv6 = _resolve_ips(domain)
    all_ips = ipv4 + ipv6

    if not all_ips:
        lines.append("ERROR: Domain does not resolve (NXDOMAIN or DNS error)")
        return "\n".join(lines)

    lines.append(f"IPv4 : {', '.join(ipv4) if ipv4 else 'none'}")
    if ipv6:
        lines.append(f"IPv6 : {', '.join(ipv6[:4])}")

    # Multiple IPs = load balancer or CDN anycast
    if len(ipv4) > 1:
        lines.append(f"NOTE : {len(ipv4)} IPv4 addresses — likely load balancer or CDN anycast")
    lines.append("")

    # Per-IP intel
    for ip in ipv4[:4]:  # limit to 4 to avoid rate limits
        lines.append(f"[IP: {ip}]")

        rdns = _reverse_dns(ip)
        if rdns:
            lines.append(f"  rDNS       : {rdns}")

        info = _ip_info(ip)
        if info.get("status") == "success":
            country = info.get("country", "")
            city = info.get("city", "")
            region = info.get("regionName", "")
            isp = info.get("isp", "")
            org = info.get("org", "")
            asn_str = info.get("as", "")  # e.g. "AS13335 Cloudflare, Inc."
            asname = info.get("asname", "")

            asn_num = 0
            asn_match = re.match(r"AS(\d+)", asn_str)
            if asn_match:
                asn_num = int(asn_match.group(1))

            cloud = _detect_cloud(org, asname, asn_num)

            lines.append(f"  Location   : {city}, {region}, {country}")
            lines.append(f"  ISP        : {isp}")
            lines.append(f"  Org        : {org}")
            lines.append(f"  ASN        : {asn_str}")
            if cloud:
                lines.append(f"  Cloud      : {cloud}")
            if info.get("hosting"):
                lines.append(f"  Hosting    : yes (datacenter IP, not residential)")
            if info.get("proxy"):
                lines.append(f"  Proxy/VPN  : yes")

            # CDN note
            if asn_num in CDN_ASNS:
                lines.append(f"  ⚠ CDN edge node — real origin IP is hidden behind {CDN_ASNS[asn_num]}")
        else:
            lines.append(f"  (ip-api.com lookup failed — check network access)")
            # Fallback: try to infer from rDNS
            if rdns:
                for pattern, label in CLOUD_ASN_PATTERNS:
                    if re.search(pattern, rdns, re.IGNORECASE):
                        lines.append(f"  Cloud (rDNS hint): {label}")
                        break
        lines.append("")

    # Summary
    lines.append("[INFRASTRUCTURE SUMMARY]")
    cloud_hints: set[str] = set()
    for ip in ipv4[:4]:
        rdns = _reverse_dns(ip)
        for pattern, label in CLOUD_ASN_PATTERNS:
            if re.search(pattern, rdns, re.IGNORECASE):
                cloud_hints.add(label)
    if cloud_hints:
        lines.append(f"  Provider(s) detected: {', '.join(cloud_hints)}")
    if len(ipv4) == 1:
        lines.append("  Single IP — likely dedicated server or VPS (not CDN)")
    elif len(ipv4) > 2:
        lines.append("  Multiple IPs — CDN anycast or load-balanced pool")

    return "\n".join(lines)
