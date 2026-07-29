"""Fine-tune the packaged detector on the compact M1-M6 YOLO sample."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

from common import DEFAULT_OUTPUT_ROOT, REPO_ROOT, prepare_stage_dir, run, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    stage_dir = args.output_root / "01_yolo"
    prepare_stage_dir(stage_dir, args.output_root, args.force)
    data_yaml = REPO_ROOT / "data" / "01_yolo_detector_sample" / "microled.yaml"
    dataset_root = data_yaml.parent.resolve()
    runtime_yaml = stage_dir / "microled_runtime.yaml"
    runtime_yaml.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 1",
                "names: ['microled']",
                "",
            ]
        ),
        encoding="utf-8",
    )
    initial_model = REPO_ROOT / "models" / "yolo11n_microled_best.pt"
    training_root = stage_dir / "training"
    run(
        [
            sys.executable,
            REPO_ROOT / "training" / "train_yolo.py",
            "--data",
            runtime_yaml,
            "--model",
            initial_model,
            "--project",
            training_root,
            "--name",
            "m1_m6_finetune",
            "--epochs",
            str(args.epochs),
            "--batch",
            str(args.batch),
            "--imgsz",
            str(args.imgsz),
            "--workers",
            "0",
        ]
    )
    trained = training_root / "m1_m6_finetune" / "weights" / "best.pt"
    if not trained.is_file():
        raise FileNotFoundError(f"YOLO training did not create {trained}")
    artifact = stage_dir / "yolo11n_microled_best.pt"
    shutil.copy2(trained, artifact)
    metrics: dict[str, object] = {"epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz}
    results_csv = training_root / "m1_m6_finetune" / "results.csv"
    if results_csv.is_file():
        with results_csv.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if rows:
            metrics["last_epoch_metrics"] = {key.strip(): value for key, value in rows[-1].items()}
    manifest = write_manifest(
        stage_dir,
        "01_train_yolo",
        {"detector_model": artifact, "training_output": training_root / "m1_m6_finetune"},
        {"dataset_yaml": data_yaml, "runtime_dataset_yaml": runtime_yaml, "initial_model": initial_model},
        metrics,
    )
    print(f"Stage manifest: {manifest}")


if __name__ == "__main__":
    main()
