"""Tool schemas, dispatch, staging pipeline, trash, and user confirmation."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

import typer

import michael.globals as G
from michael import permissions
from michael import workbench as _wb
from michael.config import Config
from michael.project import Project, append_event, iter_events

if TYPE_CHECKING:
    from michael.backends import SandboxBackend


# ---------------------------------------------------------------------------
# Tool schemas (passed to the LLM as `tools=[...]`)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Overwrite (or create) a file anywhere on the Work FS. "
                "Path may be absolute or relative to the project root; "
                "parent dirs are created automatically. "
                "Writing to ~/.michael/ (the Central FS) is blocked. "
                "The change is applied to a staging copy first. "
                "You MUST predict the resulting filesystem delta in "
                "`expected_changes` (every path that will be added, modified, "
                "or removed — use absolute paths for files outside the project "
                "root). If reality diverges from your prediction, Michael "
                "returns a mismatch error and the user is NOT prompted — "
                "re-propose. If `verify` is provided, it runs in the staging "
                "copy after the write; verify failures roll back this call. "
                "If prediction matches and verify passes (or is omitted), the "
                "user is shown the diff and asked to confirm."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to project root."},
                    "content": {"type": "string", "description": "Full file content."},
                    "expected_changes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Required. Project-relative paths you predict will be "
                            "added, modified, or removed. Mismatch with the actual "
                            "staged delta is returned to you as an error."
                        ),
                    },
                    "verify": {
                        "type": "string",
                        "description": (
                            "Optional shell command run in the staging copy after "
                            "applying the write. Exit 0 = pass; non-zero = fail "
                            "and the user is not bothered."
                        ),
                    },
                },
                "required": ["path", "content", "expected_changes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the project workspace. Auto-executes.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a directory in the project workspace. Auto-executes.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory",
            "description": (
                "Search past LLM responses stored in this project's event log for a "
                "query string (case-insensitive substring match). Returns up to 5 "
                "matching excerpts with timestamps. Auto-executes — no confirmation "
                "needed. Only works when log_responses=true in config."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Substring to search for in past assistant responses.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Apply a unified diff to a file. Goes through the same "
                "staging + predicted-delta + verify + user-confirm flow as "
                "write_file: `expected_changes` is required, mismatches are "
                "returned to you, and the user is only prompted on a clean "
                "match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "unified_diff": {"type": "string"},
                    "expected_changes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Required. Project-relative paths you predict will be "
                            "added, modified, or removed. Mismatch is returned "
                            "to you as an error."
                        ),
                    },
                    "verify": {"type": "string"},
                },
                "required": ["path", "unified_diff", "expected_changes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_in_sandbox",
            "description": (
                "Run Python code in an isolated podman sandbox: NO network access, "
                "read-only mount, dropped caps. Requires user confirmation. "
                "Do NOT use this for anything that needs internet (HTTP requests, "
                "APIs, web scraping) — use run_shell instead for those."
            ),
            "parameters": {
                "type": "object",
                "properties": {"python_code": {"type": "string"}},
                "required": ["python_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Run a shell command in the project workspace (NOT sandboxed). "
                "Has full network access — use this for curl, wget, API calls, "
                "web requests, or any command needing the internet. "
                "Requires user confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "timeout_s": {"type": "integer", "default": 60},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit_changes",
            "description": (
                "Commit all staged file changes to the project workspace. "
                "Call this when your work is complete and you are satisfied with the result. "
                "This is the ONLY way to apply staged write_file / apply_patch calls — "
                "nothing is written to disk until you call commit_changes. "
                "Do not call it unless the goal is fully met."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief description of what was done and why.",
                    }
                },
                "required": ["summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tools",
            "description": (
                "Search the tool catalog for previously built and delivered tools by keyword. "
                "Returns matching tool names, descriptions, and run commands. "
                "Auto-executes — use this before building a new tool to check if one already "
                "exists that meets your needs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Keyword(s) to search for in tool names, descriptions, and tags.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forge_tool",
            "description": (
                "Create a new dynamic tool mid-run by writing a .py file to the project's "
                "tools/ directory. The tool is immediately available for use in subsequent turns "
                "without restarting. The file must export TOOL_SCHEMA (an OpenAI function schema "
                "dict) and a callable with the same name as the tool. Auto-executes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Tool name in snake_case, must match the callable name.",
                    },
                    "code": {
                        "type": "string",
                        "description": "Full Python source. Must define TOOL_SCHEMA and a callable.",
                    },
                },
                "required": ["name", "code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch the content of a URL and return the response body as text. "
                "Useful for reading documentation, APIs, raw files, or any web resource. "
                "Responses larger than 100 KB are truncated. Auto-executes — no confirmation needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch.",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "HEAD"],
                        "description": "HTTP method (default: GET).",
                    },
                    "headers": {
                        "type": "object",
                        "description": "Optional HTTP headers as key-value pairs.",
                    },
                    "body": {
                        "type": "string",
                        "description": "Optional request body (for POST).",
                    },
                    "timeout_s": {
                        "type": "integer",
                        "description": "Request timeout in seconds (default: 15).",
                    },
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_model",
            "description": (
                "Load a structured AppModel for a known target. Returns JSON with "
                "base_url, auth pattern, endpoint signatures, stack, and notes "
                "synthesized from recon. Auto-executes — no confirmation needed. "
                "Call this as soon as you identify the target system, before writing "
                "any HTTP calls or target-specific tools. If no model exists, the "
                "response tells you what models are available and how to build one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "App or API name, e.g. 'cloudflare-api' or 'wordpress'.",
                    },
                    "version": {
                        "type": "string",
                        "description": "Version string, e.g. 'v4' or '6.4.2'.",
                    },
                },
                "required": ["name", "version"],
            },
        },
    },
]

# Sentinel returned by dispatch_tool_call when commit_changes fires, so the
# agent loop knows to exit immediately.
COMMIT_SENTINEL = "__COMMITTED__"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _resolve_in_project(project: Project, rel: str) -> pathlib.Path:
    root = pathlib.Path(project.path).resolve()
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise G.MichaelError(f"path escapes project root: {rel}") from e
    permissions.assert_not_central(candidate, "access")
    return candidate


def _summary_for(name: str, args: dict[str, Any]) -> str:
    if name == "write_file":
        return f"write_file({args.get('path', '?')}, {len(args.get('content', ''))}b)"
    if name == "read_file":
        return f"read_file({args.get('path', '?')})"
    if name == "list_dir":
        return f"list_dir({args.get('path', '.')})"
    if name == "apply_patch":
        return f"apply_patch({args.get('path', '?')})"
    if name == "run_in_sandbox":
        return f"run_in_sandbox({len(args.get('python_code', ''))}b)"
    if name == "run_shell":
        cmd = str(args.get("cmd", "?"))
        return f"run_shell({cmd[:80]}{'...' if len(cmd) > 80 else ''})"
    return f"{name}(?)"


def _format_proc_result(cp: subprocess.CompletedProcess) -> str:
    out = [f"rc={cp.returncode}"]
    if cp.stdout:
        out.append(f"stdout (first 2000 chars):\n{cp.stdout[:2000]}")
    if cp.stderr:
        out.append(f"stderr (first 500 chars):\n{cp.stderr[:500]}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Staging helpers
# ---------------------------------------------------------------------------


def _stage_ignore(directory: str, contents: list[str]) -> list[str]:
    out: list[str] = []
    for c in contents:
        p = pathlib.Path(directory) / c
        if p.is_dir() and (c.startswith(".") or c in G.SKIP_DIRS):
            out.append(c)
    return out


def _stage_project(project: Project) -> pathlib.Path:
    src = pathlib.Path(project.path).resolve()
    if not src.is_dir():
        raise G.MichaelError(f"project root does not exist: {src}")
    parent = pathlib.Path(tempfile.mkdtemp(prefix="michael-stage-"))
    dst = parent / src.name
    shutil.copytree(src, dst, ignore=_stage_ignore, symlinks=False)
    return dst


def _file_hashes(root: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    root = root.resolve()
    if not root.is_dir():
        return out
    for dp, dirs, files in os.walk(root):
        dp_path = pathlib.Path(dp)
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in G.SKIP_DIRS]
        for fn in files:
            fp = dp_path / fn
            try:
                data = fp.read_bytes()
            except OSError:
                continue
            rel = str(fp.relative_to(root))
            out[rel] = hashlib.sha256(data).hexdigest()
    return out


def _combined_hashes(
    stage_root: pathlib.Path, ext_root: pathlib.Path
) -> dict[str, str]:
    """Hash every staged file.

    Keys for project files are project-relative strings (e.g. ``"src/main.py"``).
    Keys for external files are absolute path strings (e.g. ``"/tmp/foo.py"``).
    """
    out = _file_hashes(stage_root)
    if ext_root.is_dir():
        for dp, dirs, files in os.walk(ext_root):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in G.SKIP_DIRS]
            for fn in files:
                fp = pathlib.Path(dp) / fn
                try:
                    data = fp.read_bytes()
                except OSError:
                    continue
                rel_in_ext = fp.relative_to(ext_root)
                abs_key = "/" + str(rel_in_ext)
                out[abs_key] = hashlib.sha256(data).hexdigest()
    return out


def _diff_hashes(before: dict[str, str], after: dict[str, str]) -> dict[str, list[str]]:
    added = sorted(p for p in after if p not in before)
    removed = sorted(p for p in before if p not in after)
    modified = sorted(p for p in after if p in before and after[p] != before[p])
    return {"added": added, "removed": removed, "modified": modified}


def _check_expected(expected: list[str], delta: dict[str, list[str]]) -> str:
    actual = set(delta["added"]) | set(delta["modified"]) | set(delta["removed"])
    expected_set = set(expected)
    extra = sorted(actual - expected_set)
    missing = sorted(expected_set - actual)
    if not extra and not missing:
        return ""
    parts: list[str] = []
    if extra:
        parts.append(f"extra={extra}")
    if missing:
        parts.append(f"missing={missing}")
    return "; ".join(parts)


def _run_verify(cmd: str, cwd: pathlib.Path, *, timeout_s: int = 60) -> tuple[int, str]:
    try:
        cp = subprocess.run(
            ["bash", "-c", cmd],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "")[-1000:] if isinstance(e.stdout, str) else ""
        errs = (e.stderr or "")[-500:] if isinstance(e.stderr, str) else ""
        return 124, f"verify timed out after {timeout_s}s\nstdout:\n{out}\nstderr:\n{errs}"
    out = ""
    if cp.stdout:
        out += f"stdout (truncated):\n{cp.stdout[-1500:]}\n"
    if cp.stderr:
        out += f"stderr (truncated):\n{cp.stderr[-500:]}"
    return cp.returncode, out


def _staging_target(
    path_str: str,
    stage_root: pathlib.Path,
    ext_root: pathlib.Path,
    real_project_root: pathlib.Path,
) -> pathlib.Path:
    """Resolve *path_str* to its location inside the staging area.

    Project-relative and project-absolute paths land under *stage_root*.
    External absolute paths land under *ext_root* mirroring the absolute path
    (e.g. ``/tmp/foo.py`` → ``<ext_root>/tmp/foo.py``).
    Central FS paths are rejected before reaching here, but we double-check.
    """
    abs_path = permissions.resolve_any(path_str, real_project_root)
    permissions.assert_not_central(abs_path, "write")
    if permissions.is_project_path(abs_path, real_project_root):
        rel = abs_path.relative_to(real_project_root)
        return (stage_root / rel).resolve()
    # External path — mirror under ext_root
    stripped = str(abs_path).lstrip("/")
    return (ext_root / stripped).resolve()


def _apply_in_staging(
    name: str,
    args: dict[str, Any],
    stage_root: pathlib.Path,
    real_project_root: pathlib.Path,
    ext_root: pathlib.Path,
) -> None:
    path_str = str(args.get("path", ""))
    target = _staging_target(path_str, stage_root, ext_root, real_project_root)
    target.parent.mkdir(parents=True, exist_ok=True)

    if name == "write_file":
        target.write_text(str(args["content"]))
        return

    if name == "apply_patch":
        if not target.is_file():
            # For external files not yet copied into staging, seed from real FS.
            abs_path = permissions.resolve_any(path_str, real_project_root)
            if abs_path.is_file():
                shutil.copy2(abs_path, target)
            else:
                raise G.MichaelError(f"apply_patch target does not exist: {path_str}")
        if not shutil.which("patch"):
            raise G.MichaelError("`patch` not installed on host (apt install patch)")
        diff = str(args["unified_diff"])
        cp = subprocess.run(
            ["patch", "--no-backup-if-mismatch", "-u", str(target)],
            input=diff,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if cp.returncode != 0:
            raise G.MichaelError(
                f"patch failed in staging (rc={cp.returncode}): "
                f"{(cp.stderr or '')[-500:]}"
            )
        return

    raise G.MichaelError(f"_apply_in_staging: unknown tool {name}")


def _save_trash(
    project: Project,
    op_name: str,
    args: dict[str, Any],
    delta: dict[str, list[str]],
    real_root: pathlib.Path,
    *,
    verify_rc: Optional[int],
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    trash_id = f"{ts}-{uuid.uuid4().hex[:6]}"
    trash_dir = G.PROJECTS_DIR / project.slug / "trash" / trash_id
    trash_dir.mkdir(parents=True, exist_ok=True)
    before_dir = trash_dir / "before"
    before_dir.mkdir(exist_ok=True)
    for rel in delta["modified"] + delta["removed"]:
        if rel.startswith("/"):
            # External file — snapshot from its real absolute path.
            src = pathlib.Path(rel)
            dst = before_dir / "_ext" / rel.lstrip("/")
        else:
            src = real_root / rel
            dst = before_dir / rel
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except OSError:
            continue
    metadata = {
        "trash_id": trash_id,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool": op_name,
        "summary": _summary_for(op_name, args),
        "args": args,
        "delta": delta,
        "verify_rc": verify_rc,
    }
    (trash_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    return trash_id


def _sync_to_real(
    stage_root: pathlib.Path,
    real_root: pathlib.Path,
    delta: dict[str, list[str]],
    ext_root: Optional[pathlib.Path] = None,
) -> None:
    for rel in delta["added"] + delta["modified"]:
        if rel.startswith("/"):
            # External path — source is mirrored under ext_root.
            if ext_root is None:
                continue
            src = ext_root / rel.lstrip("/")
            dst = pathlib.Path(rel)
        else:
            src = stage_root / rel
            dst = real_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for rel in delta["removed"]:
        dst = pathlib.Path(rel) if rel.startswith("/") else real_root / rel
        if dst.is_file():
            try:
                dst.unlink()
            except OSError:
                pass


def _list_trash(project: Project) -> list[dict[str, Any]]:
    trash_root = G.PROJECTS_DIR / project.slug / "trash"
    if not trash_root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(trash_root.iterdir()):
        if not d.is_dir():
            continue
        meta = d / "metadata.json"
        if not meta.is_file():
            continue
        try:
            out.append(json.loads(meta.read_text()))
        except json.JSONDecodeError:
            continue
    return out


def _undo_one(project: Project, trash_id: Optional[str] = None) -> dict[str, Any]:
    trash_root = G.PROJECTS_DIR / project.slug / "trash"
    if not trash_root.is_dir():
        raise G.MichaelError("no trash entries to undo")
    entries = sorted([d for d in trash_root.iterdir() if d.is_dir()])
    if not entries:
        raise G.MichaelError("no trash entries to undo")
    if trash_id:
        target = trash_root / trash_id
        if not target.is_dir():
            raise G.MichaelError(f"unknown trash id: {trash_id}")
    else:
        target = entries[-1]
    metadata = json.loads((target / "metadata.json").read_text())
    delta = metadata.get("delta", {}) or {}
    real_root = pathlib.Path(project.path).resolve()
    for rel in delta.get("modified", []) + delta.get("removed", []):
        if rel.startswith("/"):
            src = target / "before" / "_ext" / rel.lstrip("/")
            dst = pathlib.Path(rel)
        else:
            src = target / "before" / rel
            dst = real_root / rel
        if not src.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    for rel in delta.get("added", []):
        dst = pathlib.Path(rel) if rel.startswith("/") else real_root / rel
        if dst.is_file():
            try:
                dst.unlink()
            except OSError:
                pass
    shutil.rmtree(target, ignore_errors=True)
    return metadata


# ---------------------------------------------------------------------------
# Tool execution (read/list/search/sandbox/shell)
# ---------------------------------------------------------------------------


def _search_memory(project: Project, query: str, cfg: Config) -> str:
    if not cfg.log_responses:
        return (
            "search_memory: responses are not stored in this installation. "
            "Set log_responses=true in config and run new sessions to build memory."
        )
    query = query.strip()
    if not query:
        return "search_memory: query must not be empty"
    q = query.lower()
    hits: list[str] = []
    for ev in iter_events(project.events_path):
        if len(hits) >= 8:
            break
        ev_type = ev.get("type", "")
        payload = ev.get("payload") or {}
        ts = ev.get("ts", "?")

        if ev_type == "assistant.message":
            text = payload.get("text") or ""
            if not text or q not in text.lower():
                continue
            turn = payload.get("turn", "?")
            excerpt = text[:500] + ("…" if len(text) > 500 else "")
            hits.append(f"[{ts} turn={turn} type=reasoning]\n{excerpt}")

        elif ev_type == "tool.executed" and payload.get("brief_result"):
            brief = payload["brief_result"]
            summary = payload.get("summary", "")
            if q not in brief.lower() and q not in summary.lower():
                continue
            hits.append(
                f"[{ts} tool={payload.get('tool')} type=result]\n"
                f"{summary}\n{brief[:400]}"
            )

    if not hits:
        return f"search_memory: no matches for {query!r} in this project's history"
    return (
        f"search_memory: {len(hits)} match(es) for {query!r}\n\n"
        + "\n\n---\n\n".join(hits)
    )


def execute_tool(
    name: str,
    args: dict[str, Any],
    project: Project,
    cfg: Config,
    backend: "SandboxBackend",
) -> str:
    if name == "read_file":
        project_root = pathlib.Path(project.path).resolve()
        target = permissions.resolve_any(str(args["path"]), project_root)
        permissions.assert_not_central(target, "read")
        if not target.is_file():
            return "error: not a file"
        try:
            text = target.read_text(errors="replace")
        except OSError as e:
            return f"error: {e}"
        if len(text) > 200_000:
            return f"file too large ({len(text)}b) — refusing to read full content"
        return text

    if name == "list_dir":
        project_root = pathlib.Path(project.path).resolve()
        target = permissions.resolve_any(str(args.get("path", ".")), project_root)
        permissions.assert_not_central(target, "read")
        if not target.is_dir():
            return "error: not a directory"
        rows = []
        for child in sorted(target.iterdir()):
            try:
                size = child.stat().st_size
            except OSError:
                size = -1
            kind = "dir" if child.is_dir() else "file"
            rows.append(f"{kind}\t{child.name}\t{size}")
        return "\n".join(rows) or "(empty)"

    if name == "run_in_sandbox":
        cp = backend.run(
            str(args["python_code"]),
            network=False,
            timeout_s=cfg.sandbox.timeout_s,
            project=project,
        )
        return _format_proc_result(cp)

    if name == "run_shell":
        if block_msg := permissions.check_shell_cmd(str(args.get("cmd", ""))):
            return f"error: {block_msg}"
        timeout_s = int(args.get("timeout_s", 60))
        cwd = pathlib.Path(project.path).resolve()
        try:
            cp = subprocess.run(
                ["bash", "-c", str(args["cmd"])],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            return f"timed out after {timeout_s}s; partial stdout:\n{(e.stdout or '')[-1000:]}"
        return _format_proc_result(cp)

    if name == "search_memory":
        return _search_memory(project, str(args.get("query", "")), cfg)

    if name == "search_tools":
        from michael.project import load_catalog
        catalog = load_catalog()
        if not catalog:
            return "search_tools: catalog is empty — no tools have been delivered yet"
        q = str(args.get("query", "")).lower().strip()
        if not q:
            return "search_tools: query must not be empty"
        hits: list[str] = []
        for slug, entry in sorted(catalog.items()):
            desc = str(entry.get("description", "")).lower()
            tags = " ".join(entry.get("tags", [])).lower()
            if q in slug.lower() or q in desc or q in tags:
                run_cmd = entry.get("run_cmd", "—")
                installed = entry.get("installed_as")
                display_cmd = installed if installed else run_cmd
                hits.append(
                    f"  {slug}: {entry.get('description', '(no description)')}\n"
                    f"    run: {display_cmd}"
                )
        if not hits:
            return f"search_tools: no matches for {q!r} in {len(catalog)} catalog entries"
        return f"search_tools: {len(hits)} match(es) for {q!r}:\n" + "\n".join(hits)

    if name == "forge_tool":
        tool_name = str(args.get("name", "")).strip()
        code = str(args.get("code", ""))
        if not tool_name:
            return "forge_tool: name is required"
        if not code:
            return "forge_tool: code is required"
        tools_dir = pathlib.Path(project.path) / "tools"
        tools_dir.mkdir(exist_ok=True)
        target = tools_dir / f"{tool_name}.py"
        try:
            target.write_text(code)
        except OSError as exc:
            return f"forge_tool: failed to write {target}: {exc}"
        import importlib.util as _ilu
        try:
            spec = _ilu.spec_from_file_location(tool_name, target)
            mod = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            if not hasattr(mod, "TOOL_SCHEMA"):
                return f"forge_tool: wrote {target} but TOOL_SCHEMA not found — tool will not auto-load"
            if not callable(getattr(mod, tool_name, None)):
                return f"forge_tool: wrote {target} but callable {tool_name!r} not found — dispatch will fail"
        except Exception as exc:
            return f"forge_tool: wrote {target} but validation failed: {exc}"
        return f"forge_tool: {tool_name} created at {target} — available immediately in this run"

    if name == "fetch_url":
        import httpx

        url = str(args.get("url", ""))
        if not url:
            return "error: url is required"
        method = str(args.get("method", "GET")).upper()
        headers = args.get("headers") or {}
        body = args.get("body")
        timeout_s = int(args.get("timeout_s", 15))
        _MAX_RESPONSE_BYTES = 100_000
        try:
            with httpx.Client(follow_redirects=True, timeout=timeout_s) as client:
                resp = client.request(
                    method,
                    url,
                    headers=headers,
                    content=body.encode() if isinstance(body, str) else None,
                )
        except httpx.TimeoutException:
            return f"error: request timed out after {timeout_s}s"
        except httpx.RequestError as exc:
            return f"error: {exc}"
        content_type = resp.headers.get("content-type", "")
        try:
            text = resp.text
        except Exception:
            text = resp.content.decode("utf-8", errors="replace")
        truncated = len(text) > _MAX_RESPONSE_BYTES
        body_out = text[:_MAX_RESPONSE_BYTES]
        parts = [f"status: {resp.status_code}", f"content-type: {content_type}"]
        if truncated:
            parts.append(f"(truncated to {_MAX_RESPONSE_BYTES} bytes)")
        parts.append(body_out)
        return "\n".join(parts)

    if name == "load_model":
        from dataclasses import asdict as _asdict
        from michael.appmodel import load_model as _load_model, list_models as _list_models
        model_name_arg = str(args.get("name", ""))
        model_ver_arg = str(args.get("version", ""))
        if not model_name_arg or not model_ver_arg:
            return "load_model: both 'name' and 'version' are required"
        try:
            model = _load_model(project, model_name_arg, model_ver_arg)
            return json.dumps(_asdict(model), indent=2)
        except G.MichaelError:
            available = [f"{mo.name} {mo.version}" for mo in _list_models(project)]
            hint = (
                f"Available models: {available}" if available
                else "No models built yet — run recon (recon_passive/explore_service), "
                     "then write models/<name>-<version>.json with base_url, auth, endpoints, stack, notes."
            )
            return f"No model for {model_name_arg!r} {model_ver_arg!r}. {hint}"

    return f"error: unknown tool {name}"


# ---------------------------------------------------------------------------
# Verify-before-apply flow (write_file, apply_patch)
# ---------------------------------------------------------------------------


def _format_delta(delta: dict[str, list[str]]) -> str:
    parts: list[str] = [
        f"files added:    {len(delta['added'])}",
        f"files modified: {len(delta['modified'])}",
        f"files removed:  {len(delta['removed'])}",
    ]
    if delta["added"]:
        parts.append("  + " + "\n  + ".join(delta["added"]))
    if delta["modified"]:
        parts.append("  ~ " + "\n  ~ ".join(delta["modified"]))
    if delta["removed"]:
        parts.append("  - " + "\n  - ".join(delta["removed"]))
    return "\n".join(parts)


def _format_review(
    name: str,
    args: dict[str, Any],
    project: Project,
    stage_root: pathlib.Path,
    delta: dict[str, list[str]],
    verify_rc: Optional[int],
    verify_out: str,
    expected: list[str],
    mismatch: str,
) -> str:
    sections: list[str] = []
    sections.append(f"tool: {name}({args.get('path', '?')})")
    sections.append(
        f"predicted: added/modified/removed = {sorted(expected)}"
    )
    sections.append(
        f"actual:    added={delta['added']}  "
        f"modified={delta['modified']}  removed={delta['removed']}"
    )
    if mismatch:
        sections.append(f"prediction-vs-reality: {mismatch}")
    else:
        sections.append("prediction-vs-reality: match")

    if name == "write_file":
        rel = str(args.get("path", "?"))
        try:
            real_target = _resolve_in_project(project, rel)
            old = real_target.read_text(errors="replace") if real_target.is_file() else ""
        except G.MichaelError:
            old = ""
        diff = "".join(difflib.unified_diff(
            old.splitlines(keepends=True),
            str(args.get("content", "")).splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )) or "(no changes)"
        sections.append("diff (vs. real workspace):")
        sections.append(diff)
    elif name == "apply_patch":
        sections.append("patch applied:")
        sections.append(str(args.get("unified_diff", "")))

    if verify_rc is not None:
        tail = (verify_out or "")[-1200:]
        sections.append(f"verify rc={verify_rc}\n{tail}")

    sections.append(
        f"staging committed at {stage_root}; this change is pending. "
        "Continue iterating or end your message with the Ja passcode to "
        "surface to the user."
    )
    return "\n\n".join(sections)


@dataclass
class PendingChanges:
    """Per-agent-loop staging state."""

    stage_root: Optional[pathlib.Path] = None
    change_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ext_root(self) -> Optional[pathlib.Path]:
        """Staging area for external (non-project) file writes."""
        return self.stage_root.parent / "_ext" if self.stage_root is not None else None

    def ensure_stage(self, project: Project) -> pathlib.Path:
        if self.stage_root is None:
            self.stage_root = _stage_project(project)
            (self.stage_root.parent / "_ext").mkdir(exist_ok=True)
        return self.stage_root

    def discard(self) -> None:
        if self.stage_root is not None:
            shutil.rmtree(self.stage_root.parent, ignore_errors=True)
            self.stage_root = None
        self.change_log.clear()


def _snapshot_file(
    stage_root: pathlib.Path,
    ext_root: pathlib.Path,
    path_str: str,
    real_project_root: pathlib.Path,
) -> tuple[bool, Optional[bytes]]:
    try:
        target = _staging_target(path_str, stage_root, ext_root, real_project_root)
    except G.MichaelError:
        return False, None
    if not target.is_file():
        # External file not yet staged — try the real filesystem.
        abs_path = permissions.resolve_any(path_str, real_project_root)
        if not permissions.is_project_path(abs_path, real_project_root) and abs_path.is_file():
            try:
                return True, abs_path.read_bytes()
            except OSError:
                return True, None
        return False, None
    try:
        return True, target.read_bytes()
    except OSError:
        return True, None


def _restore_file(
    stage_root: pathlib.Path,
    ext_root: pathlib.Path,
    path_str: str,
    real_project_root: pathlib.Path,
    existed: bool,
    blob: Optional[bytes],
) -> None:
    try:
        target = _staging_target(path_str, stage_root, ext_root, real_project_root)
    except G.MichaelError:
        return
    if not existed:
        if target.is_file():
            try:
                target.unlink()
            except OSError:
                pass
        return
    if blob is None:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(blob)


def execute_with_staging(
    name: str,
    args: dict[str, Any],
    project: Project,
    cfg: Config,
    pending: PendingChanges,
) -> str:
    """Apply the LLM's proposal in the persistent staging dir and return a review."""
    expected_raw = args.get("expected_changes")
    expected_list: list[str] = (
        [str(x) for x in expected_raw]
        if isinstance(expected_raw, list) else []
    )
    if not expected_list:
        append_event(
            "tool.delta_missing",
            {"tool": name, "summary": _summary_for(name, args)},
            project=project,
        )
        return (
            "error: expected_changes is required and must be a non-empty "
            "list of project-relative paths you predict will be added, "
            "modified, or removed. Predict the delta, then re-propose."
        )

    try:
        stage_root = pending.ensure_stage(project)
    except G.MichaelError as e:
        return f"error: staging failed: {e}"

    ext_root = pending.ext_root
    real_project_root = pathlib.Path(project.path).resolve()
    path_str = str(args.get("path", ""))
    existed, blob = _snapshot_file(stage_root, ext_root, path_str, real_project_root)
    before = _combined_hashes(stage_root, ext_root)
    try:
        _apply_in_staging(name, args, stage_root, real_project_root, ext_root)
    except G.MichaelError as e:
        _restore_file(stage_root, ext_root, path_str, real_project_root, existed, blob)
        return f"error applying in staging: {e}"
    after = _combined_hashes(stage_root, ext_root)
    delta = _diff_hashes(before, after)

    verify_rc: Optional[int] = None
    verify_out = ""
    verify_cmd = args.get("verify")
    if isinstance(verify_cmd, str) and verify_cmd.strip():
        verify_rc, verify_out = _run_verify(verify_cmd, stage_root, timeout_s=60)
        if verify_rc != 0:
            _restore_file(stage_root, ext_root, path_str, real_project_root, existed, blob)
            append_event(
                "tool.verify_failed",
                {
                    "tool": name,
                    "summary": _summary_for(name, args),
                    "verify_cmd": verify_cmd,
                    "verify_rc": verify_rc,
                    "delta": delta,
                },
                project=project,
            )
            return (
                f"verify failed in staging (rc={verify_rc}); this call was "
                f"rolled back. Prior pending changes are intact.\n"
                f"delta this call would have made: {delta}\n"
                f"verify output:\n{verify_out[-1500:]}"
            )

    mismatch = _check_expected(expected_list, delta)
    if mismatch:
        _restore_file(stage_root, ext_root, path_str, real_project_root, existed, blob)
        append_event(
            "tool.delta_mismatch",
            {
                "tool": name,
                "summary": _summary_for(name, args),
                "expected": expected_list,
                "delta": delta,
                "mismatch": mismatch,
            },
            project=project,
        )
        return (
            f"mismatch: prediction and reality diverge — {mismatch}.\n"
            f"predicted: {sorted(expected_list)}\n"
            f"actual:    added={delta['added']}  "
            f"modified={delta['modified']}  removed={delta['removed']}\n"
            "This call was rolled back. Correct expected_changes and re-propose."
        )

    append_event(
        "tool.staged",
        {
            "tool": name,
            "summary": _summary_for(name, args),
            "delta": delta,
            "verify_rc": verify_rc,
            "mismatch": mismatch,
        },
        project=project,
    )
    pending.change_log.append({
        "tool": name,
        "args": args,
        "delta": delta,
        "verify_rc": verify_rc,
        "expected": expected_list,
        "mismatch": mismatch,
    })
    return _format_review(
        name, args, project, stage_root, delta,
        verify_rc, verify_out, expected_list, mismatch,
    )


def commit_pending(project: Project, pending: PendingChanges) -> list[dict[str, Any]]:
    """At Ja-time + user-yes: sync pending stage to real workspace and discard."""
    if pending.stage_root is None or not pending.change_log:
        return []
    real_root = pathlib.Path(project.path).resolve()
    ext_root = pending.ext_root
    summaries: list[dict[str, Any]] = []
    for entry in pending.change_log:
        delta = entry["delta"]
        trash_id = _save_trash(
            project, entry["tool"], entry["args"], delta, real_root,
            verify_rc=entry.get("verify_rc"),
        )
        _sync_to_real(pending.stage_root, real_root, delta, ext_root=ext_root)
        summary = (
            f"{_summary_for(entry['tool'], entry['args'])} → applied "
            f"+{len(delta['added'])} ~{len(delta['modified'])} "
            f"-{len(delta['removed'])} trash_id={trash_id}"
        )
        append_event(
            "tool.executed",
            {
                "tool": entry["tool"],
                "args": entry["args"],
                "summary": summary[:240],
                "trash_id": trash_id,
                "delta": delta,
                "verify_rc": entry.get("verify_rc"),
            },
            project=project,
        )
        summaries.append({"trash_id": trash_id, "summary": summary})
    pending.discard()
    return summaries


# ---------------------------------------------------------------------------
# Confirmation (Y/n/Edit) for run_in_sandbox / run_shell
# ---------------------------------------------------------------------------


def _render_for_confirmation(name: str, args: dict[str, Any], project: Project) -> tuple[str, str]:
    if name == "run_in_sandbox":
        return str(args.get("python_code", "")), "python"
    if name == "run_shell":
        return f"cmd: {args.get('cmd', '?')}\ncwd: {project.path}", "bash"
    return json.dumps(args, indent=2), "json"


def _edit_args(name: str, args: dict[str, Any]) -> Optional[dict[str, Any]]:
    if name == "write_file":
        edited = typer.edit(str(args["content"]))
        return {**args, "content": edited} if edited is not None else None
    if name == "apply_patch":
        edited = typer.edit(str(args["unified_diff"]))
        return {**args, "unified_diff": edited} if edited is not None else None
    if name == "run_in_sandbox":
        edited = typer.edit(str(args["python_code"]))
        return {**args, "python_code": edited} if edited is not None else None
    if name == "run_shell":
        new_cmd = typer.prompt("new cmd", default=str(args.get("cmd", "")))
        return {**args, "cmd": new_cmd}
    edited = typer.edit(json.dumps(args, indent=2))
    if edited is None:
        return None
    try:
        return json.loads(edited)
    except json.JSONDecodeError:
        return None


def confirm_tool_call(
    name: str,
    args: dict[str, Any],
    project: Project,
) -> tuple[str, dict[str, Any]]:
    from rich.panel import Panel
    from rich.syntax import Syntax
    while True:
        rendered, lexer = _render_for_confirmation(name, args, project)
        G.console.print(
            Panel(
                Syntax(rendered, lexer, theme="ansi_dark", word_wrap=True),
                title=f"[cyan]propose[/] {name}",
                border_style="cyan",
            )
        )
        choice = (typer.prompt("Apply? [Y]es / [n]o / [e]dit", default="y") or "").strip().lower()
        if choice in ("", "y", "yes"):
            return "yes", args
        if choice in ("n", "no"):
            return "no", args
        if choice in ("e", "edit"):
            edited = _edit_args(name, args)
            if edited is None:
                G.err.print("editor returned no content; try again")
                continue
            args = edited
            continue
        G.err.print(f"unknown choice: {choice!r}")


# ---------------------------------------------------------------------------
# Dynamic tool dispatch (for tools written by the agent to tools/<name>.py)
# ---------------------------------------------------------------------------


def _dispatch_dynamic_tool_from_path(name: str, args: dict[str, Any], py_file: pathlib.Path, project: Project) -> str:
    """Load and call a tool script from the given py_file path."""
    import importlib.util as _ilu
    token = _wb._set_context(project)
    try:
        spec = _ilu.spec_from_file_location(name, py_file)
        mod = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        fn = getattr(mod, name)
        return str(fn(**args))
    except Exception as exc:
        return f"error running dynamic tool {name!r}: {exc}"
    finally:
        _wb._reset_context(token)


# ---------------------------------------------------------------------------
# Master dispatch
# ---------------------------------------------------------------------------


def dispatch_tool_call(
    name: str,
    args: dict[str, Any],
    project: Project,
    cfg: Config,
    backend: "SandboxBackend",
    pending: PendingChanges,
) -> str:
    """Route one LLM tool call to the right handler."""
    summary = _summary_for(name, args)

    if name in G.AUTO_EXEC_TOOLS:
        try:
            result = execute_tool(name, args, project, cfg, backend)
        except G.MichaelError as e:
            result = f"error: {e}"
        first = (result.splitlines()[0] if result else "ok")[:120]
        append_event(
            "tool.executed",
            {
                "tool": name,
                "args": args,
                "summary": f"{summary} → {first}",
                "result_chars": len(result),
            },
            project=project,
        )
        return result

    if name in ("write_file", "apply_patch"):
        return execute_with_staging(name, args, project, cfg, pending)

    if name == "commit_changes":
        summaries = commit_pending(project, pending)
        if summaries:
            for s in summaries:
                G.console.print(f"[green]applied[/] {s['summary']}")
        else:
            G.console.print("[dim]no file changes staged[/]")
        append_event(
            "tool.executed",
            {"tool": "commit_changes", "args": args, "summary": args.get("summary", "")},
            project=project,
        )
        return COMMIT_SENTINEL

    # Dynamic tool invented by the agent (tools/<name>.py) — auto-executed, no confirmation.
    dynamic_result = _try_dynamic_dispatch(name, args, project)
    if dynamic_result is not None:
        first = (dynamic_result.splitlines()[0] if dynamic_result else "ok")[:120]
        append_event(
            "tool.executed",
            {"tool": name, "args": args, "summary": f"{summary} → {first}",
             "result_chars": len(dynamic_result), "dynamic": True},
            project=project,
        )
        return dynamic_result

    try:
        decision, final_args = confirm_tool_call(name, args, project)
    except (KeyboardInterrupt, typer.Abort):
        decision, final_args = "no", args
    if decision == "no":
        append_event(
            "tool.rejected",
            {"tool": name, "args": args, "summary": summary},
            project=project,
        )
        return "[user rejected this tool call]"
    try:
        result = execute_tool(name, final_args, project, cfg, backend)
    except G.MichaelError as e:
        result = f"error: {e}"
    first = (result.splitlines()[0] if result else "ok")[:120]
    payload: dict[str, Any] = {
        "tool": name,
        "args": final_args,
        "summary": f"{_summary_for(name, final_args)} → {first}",
        "result_chars": len(result),
    }
    if name in ("run_in_sandbox", "run_shell"):
        payload["brief_result"] = result[:600]
    append_event("tool.executed", payload, project=project)
    return result


def _try_dynamic_dispatch(name: str, args: dict[str, Any], project: Project) -> Optional[str]:
    """Return dynamic tool result if <name>.py exists in any dynamic tool dir, else None.

    Search order (first found wins): project-local > user global > bundled.
    """
    search_dirs = [
        pathlib.Path(project.path) / "tools",
        pathlib.Path(G.GLOBAL_TOOLS_DIR),
        pathlib.Path(__file__).parent.parent / "toolbox",
    ]
    for d in search_dirs:
        py_file = d / f"{name}.py"
        if py_file.exists():
            return _dispatch_dynamic_tool_from_path(name, args, py_file, project)
    return None
