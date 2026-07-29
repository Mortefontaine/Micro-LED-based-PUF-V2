"""Apply the stage-1 YOLO and stage-2 STN to all compact M1-M6 raw images."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from common import (
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    prepare_stage_dir,
    read_artifact,
    run,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--detector-manifest", type=Path, default=None)
    parser.add_argument("--stn-manifest", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--min-valid-fraction", type=float, default=0.98)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    detector_manifest = args.detector_manifest or args.output_root / "01_yolo" / "stage_manifest.json"
    stn_manifest = args.stn_manifest or args.output_root / "02_stn" / "stage_manifest.json"
    detector = read_artifact(detector_manifest, "detector_model")
    stn = read_artifact(stn_manifest, "stn_model")
    raw_root = REPO_ROOT / "data" / "00_raw_rgb_M1_M6"
    stage_dir = args.output_root / "03_alignment"
    prepare_stage_dir(stage_dir, args.output_root, args.force)
    command: list[str | Path] = [
        sys.executable,
        REPO_ROOT / "code" / "microled_align.py",
        "--input",
        raw_root,
        "--output",
        stage_dir,
        "--detector-model",
        detector,
        "--stn-model",
        stn,
        "--device",
        args.device,
        "--min-valid-fraction",
        str(args.min_valid_fraction),
        "--preview-count",
        "12",
    ]
    if args.limit:
        command.extend(["--limit", str(args.limit)])
    run(command)
    aligned_root = stage_dir / "aligned_rgb_256"
    files = sorted(aligned_root.rglob("*.png"))
    if not files:
        raise RuntimeError("Alignment stage produced no images")
    devices = sorted({path.parent.name.split("_", 1)[0] for path in files})
    conditions = sorted({path.parent.name for path in files})
    summary_csv = stage_dir / "metrics.csv"
    passed = len(files)
    if summary_csv.is_file():
        with summary_csv.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        passed = sum(str(row.get("alignment_pass", "")).lower() in {"true", "1"} for row in rows)
    manifest = write_manifest(
        stage_dir,
        "03_align_raw",
        {
            "aligned_images": aligned_root,
            "alignment_summary": summary_csv,
        },
        {
            "raw_sample": raw_root,
            "detector_manifest": detector_manifest,
            "stn_manifest": stn_manifest,
        },
        {
            "aligned_image_count": len(files),
            "alignment_pass_count": passed,
            "devices": devices,
            "condition_count": len(conditions),
            "min_valid_fraction": args.min_valid_fraction,
        },
    )
    print(f"Stage manifest: {manifest}")


if __name__ == "__main__":
    main()
