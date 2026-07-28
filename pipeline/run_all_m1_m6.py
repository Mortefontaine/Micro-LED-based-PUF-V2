"""Run the complete M1-M6 detector -> STN -> alignment -> PUF/FE loop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from common import DEFAULT_OUTPUT_ROOT, REPO_ROOT, read_artifact, run


PIPELINE_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--yolo-epochs", type=int, default=1)
    parser.add_argument("--stn-epochs", type=int, default=1)
    parser.add_argument("--yolo-batch", type=int, default=4)
    parser.add_argument("--stn-batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quality-corr-min", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=0, help="Debug only; zero processes all 108 raw sample images.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    force = ["--force"] if args.force else []
    run(
        [
            sys.executable,
            PIPELINE_DIR / "stage_01_train_yolo.py",
            "--output-root",
            args.output_root,
            "--epochs",
            str(args.yolo_epochs),
            "--batch",
            str(args.yolo_batch),
            *force,
        ]
    )
    run(
        [
            sys.executable,
            PIPELINE_DIR / "stage_02_train_stn.py",
            "--output-root",
            args.output_root,
            "--epochs",
            str(args.stn_epochs),
            "--batch-size",
            str(args.stn_batch),
            "--device",
            args.device,
            *force,
        ]
    )
    align_command: list[str | Path] = [
        sys.executable,
        PIPELINE_DIR / "stage_03_align_raw.py",
        "--output-root",
        args.output_root,
        "--device",
        args.device,
        *force,
    ]
    if args.limit:
        align_command.extend(["--limit", str(args.limit)])
    run(align_command)
    run(
        [
            sys.executable,
            PIPELINE_DIR / "stage_04_puf_fuzzy.py",
            "--output-root",
            args.output_root,
            "--quality-corr-min",
            str(args.quality_corr_min),
            *force,
        ]
    )
    manifests = [
        args.output_root / f"{stage:02d}_{name}" / "stage_manifest.json"
        for stage, name in [
            (1, "yolo"),
            (2, "stn"),
            (3, "alignment"),
            (4, "puf_fuzzy"),
        ]
    ]
    for path in manifests:
        if not path.is_file():
            raise FileNotFoundError(path)
    final_summary = read_artifact(manifests[-1], "summary")
    payload = {
        "schema": "microled-m1-m6-closed-loop-v1",
        "repository_root": str(REPO_ROOT),
        "stage_manifests": [str(path.resolve()) for path in manifests],
        "final_summary": json.loads(final_summary.read_text(encoding="utf-8")),
    }
    path = args.output_root / "pipeline_manifest.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nClosed loop complete: {path}")


if __name__ == "__main__":
    main()
