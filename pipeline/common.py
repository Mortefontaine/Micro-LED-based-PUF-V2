"""Shared helpers for the self-contained M1-M6 closed-loop example."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "work" / "m1_m6_closed_loop"
MANIFEST_SCHEMA = "microled-m1-m6-stage-manifest-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: Sequence[str | Path]) -> None:
    printable = " ".join(str(value) for value in command)
    print(f"\n$ {printable}", flush=True)
    subprocess.run([str(value) for value in command], check=True, cwd=REPO_ROOT)


def prepare_stage_dir(stage_dir: Path, output_root: Path, force: bool) -> None:
    stage = stage_dir.resolve()
    root = output_root.resolve()
    try:
        stage.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Stage directory must stay inside output root: {stage}") from error
    if stage.exists():
        if not force:
            raise FileExistsError(f"{stage} exists; use --force to rebuild this stage")
        shutil.rmtree(stage)
    stage.mkdir(parents=True)


def write_manifest(
    stage_dir: Path,
    stage: str,
    artifacts: dict[str, Path],
    inputs: dict[str, Path] | None = None,
    metrics: dict[str, Any] | None = None,
) -> Path:
    encoded_artifacts: dict[str, dict[str, Any]] = {}
    for name, path in artifacts.items():
        resolved = path.resolve()
        if not resolved.is_file() and not resolved.is_dir():
            raise FileNotFoundError(f"Missing stage artifact {name}: {resolved}")
        entry: dict[str, Any] = {
            "path": resolved.relative_to(stage_dir.resolve()).as_posix(),
            "kind": "directory" if resolved.is_dir() else "file",
        }
        if resolved.is_file():
            entry["bytes"] = resolved.stat().st_size
            entry["sha256"] = sha256(resolved)
        encoded_artifacts[name] = entry
    encoded_inputs: dict[str, dict[str, Any]] = {}
    for name, path in (inputs or {}).items():
        resolved = path.resolve()
        entry = {
            "path": (
                resolved.relative_to(REPO_ROOT.resolve()).as_posix()
                if resolved.is_relative_to(REPO_ROOT.resolve())
                else str(resolved)
            ),
            "kind": "directory" if resolved.is_dir() else "file",
        }
        if resolved.is_file():
            entry["sha256"] = sha256(resolved)
        encoded_inputs[name] = entry
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "stage": stage,
        "python": sys.version,
        "inputs": encoded_inputs,
        "artifacts": encoded_artifacts,
        "metrics": metrics or {},
    }
    path = stage_dir / "stage_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def read_artifact(manifest_path: Path, name: str) -> Path:
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported stage manifest: {manifest_path}")
    try:
        entry = payload["artifacts"][name]
    except KeyError as error:
        raise KeyError(f"Artifact {name!r} not found in {manifest_path}") from error
    path = (manifest_path.parent / entry["path"]).resolve()
    if entry["kind"] == "file":
        if not path.is_file():
            raise FileNotFoundError(path)
        if entry.get("sha256") != sha256(path):
            raise ValueError(f"Artifact digest mismatch: {path}")
    elif not path.is_dir():
        raise FileNotFoundError(path)
    return path
