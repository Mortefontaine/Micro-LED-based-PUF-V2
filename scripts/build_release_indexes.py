"""Build the compact STN pair table and reproducibility indexes."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXCLUDED_TOP_LEVEL = {
    ".git",
    ".venv",
    "__pycache__",
    "local_state",
    "private_audit",
    "work",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_release_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if relative.parts[0] in EXCLUDED_TOP_LEVEL:
            continue
        if "__pycache__" in relative.parts or path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        files.append(path)
    return sorted(files)


def build_stn_pairs() -> Path:
    pair_root = DATA / "02_stn_pairs_M1_M6"
    input_root = pair_root / "input"
    target_root = pair_root / "target"
    output = pair_root / "alignment_pairs_M1_M6.csv"
    rows: list[dict[str, str]] = []
    for crop in sorted(input_root.rglob("*")):
        if not crop.is_file():
            continue
        relative = crop.relative_to(input_root)
        target = target_root / relative
        if not target.is_file():
            raise FileNotFoundError(f"Missing STN target for {relative.as_posix()}")
        condition = relative.parts[0] if len(relative.parts) > 1 else crop.stem
        rows.append(
            {
                "condition": condition,
                "frame": crop.stem,
                "crop_relative": (Path("input") / relative).as_posix(),
                "target_relative": (Path("target") / relative).as_posix(),
                "crop_path": "",
                "target_path": "",
                "error": "",
            }
        )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output


def build_dataset_index() -> Path:
    output = DATA / "DATASET_INDEX.csv"
    rows: list[dict[str, str | int]] = []
    for stage in sorted(path for path in DATA.iterdir() if path.is_dir()):
        files = [path for path in stage.rglob("*") if path.is_file()]
        rows.append(
            {
                "stage": stage.name,
                "file_count": len(files),
                "bytes": sum(path.stat().st_size for path in files),
                "scope": "compact M1-M6 sample; not full reported M1-M9 corpus",
            }
        )
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output


def build_sample_hashes() -> Path:
    output = DATA / "SAMPLE_FILE_SHA256.csv"
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in sorted(DATA.rglob("*")):
            if path.is_file() and path != output:
                writer.writerow(
                    [path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)]
                )
    return output


def build_release_hashes() -> Path:
    output = ROOT / "RELEASE_FILE_SHA256.csv"
    excluded = {output, ROOT / "PACKAGE_CONTENTS.csv"}
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in iter_release_files():
            if path.is_file() and path not in excluded:
                writer.writerow(
                    [path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)]
                )
    return output


def build_package_contents() -> Path:
    output = ROOT / "PACKAGE_CONTENTS.csv"
    excluded = {output, ROOT / "RELEASE_FILE_SHA256.csv"}
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["relative_path", "bytes"])
        for path in iter_release_files():
            if path.is_file() and path not in excluded:
                writer.writerow([path.relative_to(ROOT).as_posix(), path.stat().st_size])
    return output


def main() -> None:
    for generated in (
        build_stn_pairs(),
        build_dataset_index(),
        build_sample_hashes(),
        build_release_hashes(),
        build_package_contents(),
    ):
        print(generated.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
