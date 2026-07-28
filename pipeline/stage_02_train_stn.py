"""Fine-tune the packaged STN on the compact M1-M6 crop/target pairs."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from common import DEFAULT_OUTPUT_ROOT, REPO_ROOT, prepare_stage_dir, run, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage_dir = args.output_root / "02_stn"
    prepare_stage_dir(stage_dir, args.output_root, args.force)
    pair_csv = REPO_ROOT / "data" / "02_stn_pairs_M1_M6" / "alignment_pairs_M1_M6.csv"
    initial_model = REPO_ROOT / "models" / "luma_spatial_head_stn.pt"
    training_root = stage_dir / "training"
    run(
        [
            sys.executable,
            REPO_ROOT / "training" / "microled_train_luma_spatial_head_stn.py",
            "--pair-csv",
            pair_csv,
            "--out-dir",
            training_root,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--device",
            args.device,
            "--predictor-source",
            "blue",
            "--init-checkpoint",
            initial_model,
        ]
    )
    trained = training_root / "best_luma_spatial_head_stn.pt"
    if not trained.is_file():
        raise FileNotFoundError(f"STN training did not create {trained}")
    artifact = stage_dir / "luma_spatial_head_stn.pt"
    shutil.copy2(trained, artifact)
    manifest = write_manifest(
        stage_dir,
        "02_train_stn",
        {
            "stn_model": artifact,
            "aligned_training_pairs": training_root / "aligned_rgb_256",
            "training_report": training_root / "luma_spatial_head_stn_report.md",
        },
        {"pair_csv": pair_csv, "initial_model": initial_model},
        {"epochs": args.epochs, "batch_size": args.batch_size, "device": args.device},
    )
    print(f"Stage manifest: {manifest}")


if __name__ == "__main__":
    main()
