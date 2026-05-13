"""Fetch and parse OpenAPI/Swagger/GraphQL schema into a compact LLM-readable summary."""
from __future__ import annotations

import json
from typing import Any

import httpx

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_fetch_spec",
        "description": (
            "Fetch an API specification document (OpenAPI 2.x/3.x, Swagger, or GraphQL "
            "introspection) and return a compact summary: endpoint list with methods/paths/"
            "summaries, auth schemes, server URLs, and total endpoint count. "
            "For GraphQL: runs introspection and returns type/field inventory. "
            "Use when explore_service or web_http_probe discovers an /openapi.json, "
            "/swagger.json, /api-docs, or /graphql endpoint. "
            "Returns condensed text — not the raw spec. Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL of the spec document or GraphQL endpoint",
                },
                "format": {
                    "type": "string",
                    "description": "Force format: 'openapi', 'graphql', or 'auto' (default auto)",
                },
                "max_endpoints": {
                    "type": "integer",
                    "description": "Max endpoints to list (default 100)",
                },
                "headers": {
                    "type": "object",
                    "description": "Additional request headers (e.g. Authorization: Bearer ...)",
                },
                "timeout": {
                    "type": "number",
                    "description": "Request timeout in seconds (default 15)",
                },
            },
            "required": ["url"],
        },
    },
}

GRAPHQL_INTROSPECTION = """
{
  __schema {
    queryType { name }
    mutationType { name }
    subscriptionType { name }
    types {
      name
      kind
      description
      fields {
        name
        type { name kind ofType { name kind } }
        args { name type { name kind } }
      }
    }
  }
}
"""


def _parse_openapi(spec: dict[str, Any], max_endpoints: int) -> str:
    lines: list[str] = []
    info = spec.get("info", {})
    lines.append(f"Title   : {info.get('title', '?')}")
    lines.append(f"Version : {info.get('version', '?')}")

    # Servers
    servers = spec.get("servers", [])
    if servers:
        lines.append(f"Servers : {', '.join(s.get('url', '?') for s in servers[:5])}")
    elif spec.get("host"):  # Swagger 2.x
        scheme = (spec.get("schemes") or ["https"])[0]
        lines.append(f"Servers : {scheme}://{spec['host']}{spec.get('basePath', '')}")

    # Auth
    sec_defs = spec.get("securityDefinitions", spec.get("components", {}).get("securitySchemes", {}))
    if sec_defs:
        lines.append(f"Auth    : {', '.join(sec_defs.keys())}")

    lines.append("")
    lines.append("[ENDPOINTS]")
    paths = spec.get("paths", {})
    total = 0
    shown = 0
    for path, path_item in paths.items():
        for method in ("get", "post", "put", "patch", "delete", "options", "head"):
            op = path_item.get(method)
            if op is None:
                continue
            total += 1
            if shown >= max_endpoints:
                continue
            summary = op.get("summary", op.get("operationId", ""))[:80]
            params = len(op.get("parameters", []))
            has_body = "requestBody" in op or "parameters" in op and any(
                p.get("in") == "body" for p in op.get("parameters", [])
            )
            codes = list(op.get("responses", {}).keys())[:5]
            body_marker = " [body]" if has_body else ""
            lines.append(f"  {method.upper():<7} {path:<50} {summary}")
            if params or codes:
                lines.append(f"          params={params}  responses={codes}{body_marker}")
            shown += 1

    if total > shown:
        lines.append(f"  … {total - shown} more endpoints not shown (increase max_endpoints)")
    lines.append(f"\nTotal endpoints: {total}")
    return "\n".join(lines)


def _parse_graphql(schema_data: dict[str, Any]) -> str:
    lines: list[str] = []
    schema = schema_data.get("data", {}).get("__schema", {})
    if not schema:
        return "GraphQL introspection returned empty schema"

    lines.append(f"Query type       : {schema.get('queryType', {}).get('name', '?')}")
    lines.append(f"Mutation type    : {(schema.get('mutationType') or {}).get('name', 'none')}")
    lines.append(f"Subscription type: {(schema.get('subscriptionType') or {}).get('name', 'none')}")
    lines.append("")

    user_types = [
        t for t in schema.get("types", [])
        if not t["name"].startswith("__") and t["kind"] in ("OBJECT", "INPUT_OBJECT", "ENUM")
    ]
    lines.append(f"[TYPES — {len(user_types)} user-defined]")
    for t in user_types[:60]:
        fields = t.get("fields") or []
        field_names = [f["name"] for f in fields[:8]]
        suffix = ", ..." if len(fields) > 8 else ""
        lines.append(f"  {t['kind']:<14} {t['name']}  [{', '.join(field_names)}{suffix}]")

    if len(user_types) > 60:
        lines.append(f"  … {len(user_types) - 60} more types")

    return "\n".join(lines)


def web_fetch_spec(
    url: str,
    format: str = "auto",
    max_endpoints: int = 100,
    headers: dict[str, str] | None = None,
    timeout: float = 15.0,
) -> str:
    lines: list[str] = [f"=== web_fetch_spec: {url} ===\n"]

    req_headers = {"User-Agent": "michael-recon/1.0"}
    if headers:
        req_headers.update(headers)

    with httpx.Client(follow_redirects=True, timeout=timeout, verify=False) as client:
        # Detect/force GraphQL
        is_graphql = format == "graphql" or (
            format == "auto" and ("graphql" in url.lower() or url.rstrip("/").endswith("/graphql"))
        )

        if is_graphql:
            lines.append("[FORMAT: GraphQL introspection]\n")
            try:
                r = client.post(url, json={"query": GRAPHQL_INTROSPECTION}, headers=req_headers)
                r.raise_for_status()
                data = r.json()
                if "errors" in data and not data.get("data"):
                    lines.append(f"GraphQL introspection disabled or errored: {data['errors'][:2]}")
                else:
                    lines.append(_parse_graphql(data))
            except Exception as exc:
                lines.append(f"ERROR: {exc}")
            return "\n".join(lines)

        # OpenAPI / Swagger
        lines.append("[FORMAT: OpenAPI/Swagger]\n")
        try:
            r = client.get(url, headers=req_headers)
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if "yaml" in ct or url.endswith((".yaml", ".yml")):
                try:
                    import yaml  # type: ignore[import]
                    spec = yaml.safe_load(r.text)
                except ImportError:
                    lines.append("NOTE: YAML spec detected but PyYAML not installed — returning raw")
                    lines.append(r.text[:4000])
                    return "\n".join(lines)
            else:
                spec = r.json()

            if not isinstance(spec, dict):
                lines.append(f"Unexpected response format: {type(spec)}")
                lines.append(r.text[:2000])
            elif "openapi" in spec or "swagger" in spec or "paths" in spec:
                lines.append(_parse_openapi(spec, max_endpoints))
            else:
                lines.append("Not a recognized OpenAPI/Swagger document. Raw preview:")
                lines.append(r.text[:3000])
        except Exception as exc:
            lines.append(f"ERROR: {exc}")

    return "\n".join(lines)
