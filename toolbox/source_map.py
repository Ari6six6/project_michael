"""Fetch canonical filesystem structure for a package/version from public registries."""
from __future__ import annotations

import re
from typing import Optional

import httpx

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "source_map",
        "description": (
            "Given a package name and version, fetch the canonical filesystem structure "
            "from public registries (npm, PyPI, GitHub). Returns directory layout and "
            "interesting paths — config files, admin panels, API endpoints, install "
            "artifacts. Use after detecting version numbers in recon to reverse-engineer "
            "the target's expected filesystem: compare expected paths against what is "
            "actually accessible to find hardened paths (200 where should be 403) or "
            "exposed artifacts (200 where should not exist). Auto-executes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "package": {
                    "type": "string",
                    "description": (
                        "Package or project name as detected in recon output, "
                        "e.g. 'wordpress', 'django', 'react', 'nginx', 'grafana'"
                    ),
                },
                "version": {
                    "type": "string",
                    "description": "Version string as detected, e.g. '6.4.2', '4.2.5', '1.25.3'",
                },
            },
            "required": ["package", "version"],
        },
    },
}

# Patterns that identify security-relevant or structurally interesting paths
INTERESTING_PATTERNS = [
    "config", "admin", "login", "install", "setup", "readme", "changelog",
    "api", "cron", "xmlrpc", "upload", "uploads", ".env", ".htaccess",
    "web.config", "dockerfile", "compose", "secret", "password", "credential",
    "auth", "oauth", "token", "key", "cert", "ssl", "tls", "backup", "dump",
    "sql", "migration", "seed", "fixture", "debug", "console",
    "shell", "exec", "phpinfo", "server-status", "server-info",
    "actuator", "health", "metrics", "swagger", "openapi", "graphql",
    "wp-login", "wp-admin", "wp-cron", "wp-config", "xmlrpc",
    "index.php", "app.py", "manage.py", "artisan", "bootstrap.php",
]

PATH_DESCRIPTIONS: dict[str, str] = {
    "wp-login.php":               "login endpoint",
    "wp-admin/install.php":       "installer (should be inaccessible post-install)",
    "wp-admin/setup-config.php":  "setup wizard",
    "wp-cron.php":                "cron endpoint (world-accessible by default)",
    "xmlrpc.php":                 "XML-RPC API (commonly targeted)",
    "readme.html":                "version disclosure (should be removed)",
    "readme.txt":                 "version disclosure (should be removed)",
    ".htaccess":                  "Apache config",
    "web.config":                 "IIS config",
    ".env":                       "environment secrets file",
    ".env.example":               "example environment config",
    "wp-config.php":              "WordPress DB credentials",
    "wp-config-sample.php":       "default config template",
    "config.php":                 "PHP config file",
    "manage.py":                  "Django management CLI",
    "artisan":                    "Laravel CLI",
    "composer.json":              "PHP dependency manifest",
    "package.json":               "Node.js dependency manifest",
    "requirements.txt":           "Python dependency list",
    "Gemfile":                    "Ruby dependency manifest",
    "Dockerfile":                 "container build file",
    "docker-compose.yml":         "container orchestration",
    "docker-compose.yaml":        "container orchestration",
    "phpinfo.php":                "PHP config dump",
    "install/index.php":          "installer entry point",
    "setup/index.php":            "setup entry point",
}


def _clean_repo_url(url: str) -> Optional[str]:
    if not url:
        return None
    url = re.sub(r"^git\+", "", url).rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    m = re.search(r"github\.com[/:]([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", url)
    return m.group(1) if m else None


def _resolve_via_npm(client: httpx.Client, package: str) -> Optional[str]:
    for url in [
        f"https://registry.npmjs.org/{package}/latest",
        f"https://registry.npmjs.org/{package}",
    ]:
        try:
            resp = client.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            data = resp.json()
            repo = data.get("repository", {})
            raw = repo.get("url", "") if isinstance(repo, dict) else str(repo)
            result = _clean_repo_url(raw)
            if result:
                return result
        except Exception:
            pass
    return None


def _resolve_via_pypi(client: httpx.Client, package: str, version: str) -> Optional[str]:
    for url in [
        f"https://pypi.org/pypi/{package}/{version}/json",
        f"https://pypi.org/pypi/{package}/json",
    ]:
        try:
            resp = client.get(url, timeout=10)
            if resp.status_code != 200:
                continue
            info = resp.json().get("info", {})
            project_urls = info.get("project_urls") or {}
            for key in ("Source", "Repository", "Homepage", "Code", "GitHub", "Source Code"):
                raw = project_urls.get(key, "")
                result = _clean_repo_url(raw)
                if result:
                    return result
            result = _clean_repo_url(info.get("home_page", ""))
            if result:
                return result
        except Exception:
            pass
    return None


def _resolve_via_github_search(client: httpx.Client, package: str) -> Optional[str]:
    try:
        resp = client.get(
            "https://api.github.com/search/repositories",
            params={"q": package, "sort": "stars", "per_page": 5},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        items = resp.json().get("items", [])
        pkg_norm = re.sub(r"[-_.]", "", package.lower())
        for item in items:
            name_norm = re.sub(r"[-_.]", "", item["full_name"].split("/")[-1].lower())
            if name_norm == pkg_norm or pkg_norm in name_norm or name_norm in pkg_norm:
                return item["full_name"]
        return items[0]["full_name"] if items else None
    except Exception:
        return None


def _resolve_tag(client: httpx.Client, repo: str, version: str) -> Optional[tuple[str, str]]:
    """Return (tag_name, sha) for the given version."""
    ver_variants = [
        version,
        f"v{version}",
        f"release-{version}",
        f"release/{version}",
        version.replace(".", "_"),
        f"v{version.replace('.', '_')}",
        f"stable/{version}",
    ]
    # Try direct ref lookup first (avoids pagination issues on large repos)
    for tag in ver_variants:
        try:
            resp = client.get(
                f"https://api.github.com/repos/{repo}/git/refs/tags/{tag}",
                headers={"Accept": "application/vnd.github.v3+json"},
                timeout=8,
            )
            if resp.status_code == 200:
                sha = resp.json().get("object", {}).get("sha", "")
                if sha:
                    return tag, sha
        except Exception:
            pass
    # Fall back to listing tags (first 100) for fuzzy match
    try:
        resp = client.get(
            f"https://api.github.com/repos/{repo}/tags",
            params={"per_page": 100},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=10,
        )
        if resp.status_code == 200:
            tags = {t["name"]: t["commit"]["sha"] for t in resp.json()}
            for name, sha in tags.items():
                if version in name:
                    return name, sha
    except Exception:
        pass
    return None


def _fetch_tree(client: httpx.Client, repo: str, sha: str) -> list[str]:
    try:
        resp = client.get(
            f"https://api.github.com/repos/{repo}/git/trees/{sha}",
            params={"recursive": "1"},
            headers={"Accept": "application/vnd.github.v3+json"},
            timeout=20,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        paths = [item["path"] for item in data.get("tree", []) if item.get("type") == "blob"]
        if data.get("truncated"):
            paths.append("__TRUNCATED__")
        return paths
    except Exception:
        return []


def _build_output(repo: str, version: str, tag: str, sha: str, paths: list[str]) -> str:
    truncated = "__TRUNCATED__" in paths
    paths = [p for p in paths if p != "__TRUNCATED__"]

    lines: list[str] = [f"=== source_map: {repo.split('/')[-1]} {version} ===\n"]
    lines += [
        "[REPOSITORY]",
        f"  https://github.com/{repo} @ {tag} ({sha[:7]})",
        f"  Total files: {len(paths):,}" + (" (truncated — repo too large)" if truncated else ""),
        "",
    ]

    # Directory structure
    dir_counts: dict[str, int] = {}
    root_count = 0
    for path in paths:
        parts = path.split("/")
        if len(parts) == 1:
            root_count += 1
        else:
            d = parts[0] + "/"
            dir_counts[d] = dir_counts.get(d, 0) + 1

    lines.append("[DIRECTORY STRUCTURE]")
    for d, count in sorted(dir_counts.items(), key=lambda x: -x[1])[:20]:
        lines.append(f"  {d:<35} ({count} files)")
    if root_count:
        lines.append(f"  {'[root]':<35} ({root_count} files)")
    lines.append("")

    # Interesting paths
    interesting: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path in paths:
        path_lower = path.lower()
        if any(pat in path_lower for pat in INTERESTING_PATTERNS):
            if path not in seen:
                seen.add(path)
                desc = ""
                for key, d in PATH_DESCRIPTIONS.items():
                    if path_lower == key.lower() or path_lower.endswith("/" + key.lower()):
                        desc = d
                        break
                interesting.append((path, desc))

    lines.append("[INTERESTING PATHS]")
    if interesting:
        for path, desc in sorted(interesting)[:60]:
            if desc:
                lines.append(f"  {path:<50} — {desc}")
            else:
                lines.append(f"  {path}")
    else:
        lines.append("  (none matched known patterns)")

    return "\n".join(lines)


def _plausible(repo: str, package: str) -> bool:
    """True if the repo name bears a reasonable resemblance to the package name."""
    repo_name = re.sub(r"[-_.]", "", repo.split("/")[-1].lower())
    pkg_norm = re.sub(r"[-_.]", "", package.lower())
    return pkg_norm in repo_name or repo_name in pkg_norm


def source_map(package: str, version: str) -> str:
    """Fetch canonical filesystem structure for package@version from public registries."""
    package = package.strip().lower()
    version = version.strip().lstrip("v")

    with httpx.Client(timeout=15.0, follow_redirects=True) as client:
        npm_repo = _resolve_via_npm(client, package)
        if npm_repo and not _plausible(npm_repo, package):
            npm_repo = None

        pypi_repo = _resolve_via_pypi(client, package, version)
        if pypi_repo and not _plausible(pypi_repo, package):
            pypi_repo = None

        repo = npm_repo or pypi_repo or _resolve_via_github_search(client, package)
        if not repo:
            return (
                f"source_map: could not locate a GitHub repository for {package!r}. "
                "Try providing the full GitHub slug as the package name (e.g. 'WordPress/WordPress')."
            )

        result = _resolve_tag(client, repo, version)
        if not result:
            return (
                f"source_map: found repo {repo} but could not resolve version {version!r} to a tag. "
                f"Check https://github.com/{repo}/tags for available versions."
            )
        tag, sha = result

        paths = _fetch_tree(client, repo, sha)
        if not paths:
            return (
                f"source_map: fetched tree for {repo}@{tag} but got no file listing. "
                "The repo may be private or the tree API may be unavailable."
            )

        return _build_output(repo, version, tag, sha, paths)
