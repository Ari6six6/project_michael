"""Structured knowledge artifact bridging recon output to code generation.

An AppModel is synthesized from recon findings (targets/*.md, source_map output)
and stored as <project.path>/models/<name>-<version>.json. The H2 filesystem
snapshot captures this directory automatically, making models available in every
subsequent run without manual injection.

Michael calls load_model(name, version) as soon as it identifies the target system.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import michael.globals as G

if TYPE_CHECKING:
    from michael.project import Project


@dataclass
class AppModel:
    name: str
    version: str
    discovered_at: str          # ISO 8601 timestamp
    base_url: str               # e.g. "https://api.cloudflare.com/client/v4"
    auth: dict                  # {"type": "apikey", "headers": ["X-Auth-Email", "X-Auth-Key"]}
    endpoints: list[dict]       # [{"method": "POST", "path": "/zones/{id}/dns_records", ...}]
    stack: list[str]            # ["nginx", "Go", "Cloudflare"]
    notes: str                  # free-form synthesis of interesting findings
    source_files: list[str] = field(default_factory=list)  # recon files consumed


def _models_dir(project: "Project") -> pathlib.Path:
    return pathlib.Path(project.path) / G.MODELS_SUBDIR


def _model_path(project: "Project", name: str, version: str) -> pathlib.Path:
    safe_name = name.replace("/", "-").replace(" ", "-")
    safe_ver = version.replace("/", "-").replace(" ", "-")
    return _models_dir(project) / f"{safe_name}-{safe_ver}.json"


def save_model(project: "Project", model: AppModel) -> pathlib.Path:
    """Persist model to <project>/models/<name>-<version>.json."""
    d = _models_dir(project)
    d.mkdir(parents=True, exist_ok=True)
    path = _model_path(project, model.name, model.version)
    path.write_text(json.dumps(asdict(model), indent=2, sort_keys=True))
    return path


def load_model(project: "Project", name: str, version: str) -> AppModel:
    """Load a model; raises MichaelError if not found."""
    path = _model_path(project, name, version)
    if not path.is_file():
        raise G.MichaelError(
            f"no model for {name!r} {version!r} — "
            f"run recon then write models/{name}-{version}.json"
        )
    data = json.loads(path.read_text())
    return AppModel(**{k: data.get(k, AppModel.__dataclass_fields__[k].default
                                   if hasattr(AppModel.__dataclass_fields__[k].default, '__class__')
                                   else [])
                       for k in AppModel.__dataclass_fields__})


def list_models(project: "Project") -> list[AppModel]:
    """Return all models stored in the project, sorted by name+version."""
    d = _models_dir(project)
    if not d.is_dir():
        return []
    out: list[AppModel] = []
    for f in sorted(d.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            out.append(AppModel(**{k: data.get(k, [] if k == "source_files" else
                                               ({} if k in ("auth",) else ""))
                                   for k in AppModel.__dataclass_fields__}))
        except Exception:
            continue
    return out


def make_model(name: str, version: str, **kwargs: object) -> AppModel:
    """Convenience constructor with sensible defaults."""
    return AppModel(
        name=name,
        version=version,
        discovered_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        base_url=str(kwargs.get("base_url", "")),
        auth=dict(kwargs.get("auth", {})),  # type: ignore[arg-type]
        endpoints=list(kwargs.get("endpoints", [])),  # type: ignore[arg-type]
        stack=list(kwargs.get("stack", [])),  # type: ignore[arg-type]
        notes=str(kwargs.get("notes", "")),
        source_files=list(kwargs.get("source_files", [])),  # type: ignore[arg-type]
    )
