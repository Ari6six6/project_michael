"""BFS link crawler scoped to a single domain. Discovers pages, forms, and API paths."""
from __future__ import annotations

from collections import deque
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_crawl",
        "description": (
            "BFS spider from a starting URL, discovering all reachable pages and "
            "endpoints within the same domain. Extracts <a href>, <form action>, "
            "<link href>, <script src> URLs. Returns discovered URLs with status codes, "
            "content types, page titles, and all forms found. "
            "Useful for mapping the full surface of a web application. "
            "Does not execute JavaScript. verify=False for self-signed certs. Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_url": {
                    "type": "string",
                    "description": "Starting URL, e.g. 'https://example.com' or 'https://example.com/docs'",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "BFS depth limit (default 2, max 4)",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Max pages to visit (default 40)",
                },
                "follow_external": {
                    "type": "boolean",
                    "description": "Follow links to other domains (default false)",
                },
                "timeout": {
                    "type": "number",
                    "description": "Per-request timeout in seconds (default 8)",
                },
            },
            "required": ["start_url"],
        },
    },
}


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []
        self.forms: list[dict[str, str]] = []
        self.scripts: list[str] = []
        self.title: str = ""
        self._in_title = False
        self._current_form: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attr = dict(attrs)
        if tag == "a" and attr.get("href"):
            self.links.append(attr["href"])
        elif tag == "link" and attr.get("href"):
            self.links.append(attr["href"])
        elif tag == "script" and attr.get("src"):
            self.scripts.append(attr["src"])
        elif tag == "form":
            self._current_form = {
                "action": attr.get("action", ""),
                "method": attr.get("method", "GET").upper(),
            }
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "form" and self._current_form is not None:
            self.forms.append(self._current_form)
            self._current_form = None
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title and not self.title:
            self.title = data.strip()[:120]


def web_crawl(
    start_url: str,
    max_depth: int = 2,
    max_pages: int = 40,
    follow_external: bool = False,
    timeout: float = 8.0,
) -> str:
    max_depth = min(max_depth, 4)
    parsed_start = urlparse(start_url)
    origin_netloc = parsed_start.netloc

    visited: set[str] = set()
    # queue: (url, depth)
    queue: deque[tuple[str, int]] = deque([(start_url, 0)])
    pages: list[dict[str, Any]] = []
    all_links: set[str] = set()
    all_forms: list[dict[str, str]] = []
    external_links: set[str] = set()

    with httpx.Client(follow_redirects=True, timeout=timeout, verify=False) as client:
        while queue and len(pages) < max_pages:
            url, depth = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            try:
                r = client.get(url, headers={"User-Agent": "michael-crawl/1.0"})
                ct = r.headers.get("content-type", "")
                page_info: dict[str, Any] = {
                    "url": str(r.url),
                    "status": r.status_code,
                    "depth": depth,
                    "content_type": ct[:80],
                    "title": "",
                    "links": 0,
                    "forms": [],
                }

                if "text/html" in ct and r.status_code == 200:
                    parser = _PageParser()
                    parser.feed(r.text[:100_000])
                    page_info["title"] = parser.title
                    page_info["forms"] = parser.forms

                    for href in parser.links + parser.scripts:
                        full = urljoin(str(r.url), href)
                        full_parsed = urlparse(full)
                        if full_parsed.scheme not in ("http", "https"):
                            continue
                        all_links.add(full)
                        if full_parsed.netloc != origin_netloc:
                            external_links.add(full)
                            if follow_external and full not in visited and depth < max_depth:
                                queue.append((full, depth + 1))
                        elif full not in visited and depth < max_depth:
                            # strip fragment
                            clean = full.split("#")[0]
                            if clean not in visited:
                                queue.append((clean, depth + 1))

                    page_info["links"] = len(parser.links)
                    for f in parser.forms:
                        all_forms.append({"page": str(r.url), **f})

                pages.append(page_info)
            except Exception as exc:
                pages.append({"url": url, "status": "error", "error": str(exc), "depth": depth})

    lines: list[str] = [f"=== web_crawl: {start_url} ===\n"]
    lines.append(f"Pages visited : {len(pages)} (max_depth={max_depth}, max_pages={max_pages})")
    lines.append(f"Total links   : {len(all_links)}")
    lines.append(f"External links: {len(external_links)}")
    lines.append(f"Forms found   : {len(all_forms)}")
    lines.append("")

    lines.append("[PAGES]")
    for p in pages:
        status = p.get("status", "?")
        title = f" — {p['title']}" if p.get("title") else ""
        err = f" ERROR: {p['error']}" if p.get("error") else ""
        lines.append(f"  [{p['depth']}] {status}  {p['url']}{title}{err}")

    if all_forms:
        lines.append("\n[FORMS]")
        for f in all_forms:
            lines.append(f"  {f['method']} {f['action']}  (on {f['page']})")

    if external_links:
        lines.append(f"\n[EXTERNAL LINKS — sample of {min(10, len(external_links))}]")
        for l in list(external_links)[:10]:
            lines.append(f"  {l}")

    remaining = len(queue)
    if remaining:
        lines.append(f"\nNOTE: {remaining} URLs left in queue — increase max_pages to explore further")

    return "\n".join(lines)
