"""Train an ultralight YOLO detector on the micro-LED pseudo-label dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib"))

# Keep conda's CUDA torch first; use .yolo_deps only for missing packages.
import torch  # noqa: E402

from ultralytics import YOLO  # noqa: E402


DATA_YAML = REPO_ROOT / "data" / "01_yolo_detector_sample" / "microled.yaml"
YOLO11N = REPO_ROOT / "models" / "yolo11n_microled_best.pt"
PROJECT_DIR = REPO_ROOT / "work" / "detector_training"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_YAML)
    parser.add_argument("--model", type=Path, default=YOLO11N)
    parser.add_argument("--project", type=Path, default=PROJECT_DIR)
    parser.add_argument("--name", type=str, default="yolo11n_microled_320")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using torch {torch.__version__}, cuda={torch.cuda.is_available()}, device={device}")
    print(f"Data: {args.data}")
    print(f"Model: {args.model}")
    model = YOLO(str(args.model))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        device=device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        single_cls=True,
        cos_lr=True,
        plots=True,
        cache=False,
        verbose=True,
    )
    if hasattr(results, "results_dict"):
        print(f"Validation metrics: {results.results_dict}")
    print(f"Training output: {args.project / args.name}")


if __name__ == "__main__":
    main()
