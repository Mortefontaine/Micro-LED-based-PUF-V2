"""Prepare a lightweight detector training dataset from old YOLOv8 pseudo-labels.

The previous YOLOv8 model is used as a teacher. This script runs inference on
raw micro-LED frames, writes YOLO-format labels, and hard-links images into a
standard Ultralytics dataset layout:

    microled_light_detector_dataset/
      images/train/...jpg
      labels/train/...txt
      images/val/...jpg
      labels/val/...txt
      microled.yaml

The same label format can also be reused by Darknet/Yolo-Fastest training.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))

# Import torch from the conda environment before exposing the local dependency
# target, otherwise the CPU torch wheel in .yolo_deps may shadow it.
import torch  # noqa: E402

from ultralytics import YOLO  # noqa: E402

from microled_register_raw_frames_level2 import RAW_ROOT, natural_key  # noqa: E402


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
MODEL_PATH = REPO_ROOT / "models" / "yolo11n_microled_best.pt"
OUT_DIR = REPO_ROOT / "work" / "detector_training_dataset"


def list_images(root: Path) -> List[Path]:
    return sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=lambda p: natural_key(str(p)))


def safe_link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "exists"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def yolo_label_from_box(box_xywh, img_w: int, img_h: int) -> str:
    x, y, w, h = [float(v) for v in box_xywh]
    return f"0 {x / img_w:.8f} {y / img_h:.8f} {w / img_w:.8f} {h / img_h:.8f}\n"


def split_name(index: int, val_ratio: float) -> str:
    # Deterministic split independent of folder ordering after shuffling.
    return "val" if index < val_ratio else "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--max-total", type=int, default=3000)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_images = list_images(args.raw_root)
    rng = random.Random(args.seed)
    if args.max_total and len(all_images) > args.max_total:
        selected = rng.sample(all_images, args.max_total)
    else:
        selected = list(all_images)
    selected = sorted(selected, key=lambda p: natural_key(str(p)))
    split_flags = ["val"] * int(round(len(selected) * args.val_ratio)) + ["train"] * (len(selected) - int(round(len(selected) * args.val_ratio)))
    rng.shuffle(split_flags)

    model = YOLO(str(args.model_path))
    rows = []
    counts = {"train": 0, "val": 0, "failed": 0}
    for idx, (src, split) in enumerate(zip(selected, split_flags), 1):
        rel = src.relative_to(args.raw_root)
        image_dst = args.out_dir / "images" / split / rel
        label_dst = (args.out_dir / "labels" / split / rel).with_suffix(".txt")
        try:
            result = model(str(src), conf=args.conf, verbose=False)[0]
            boxes = result.boxes.xywh.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy() if result.boxes is not None and result.boxes.conf is not None else []
            if len(boxes) == 0:
                raise RuntimeError("no detection")
            # Single object task: use highest-confidence box.
            best_idx = int(max(range(len(boxes)), key=lambda i: float(confs[i]) if len(confs) else 0.0))
            img_h, img_w = result.orig_shape
            label_dst.parent.mkdir(parents=True, exist_ok=True)
            label_dst.write_text(yolo_label_from_box(boxes[best_idx], img_w, img_h), encoding="utf-8")
            link_mode = safe_link_or_copy(src, image_dst)
            rows.append({
                "src": str(src),
                "split": split,
                "image": str(image_dst),
                "label": str(label_dst),
                "conf": float(confs[best_idx]) if len(confs) else "",
                "num_boxes": int(len(boxes)),
                "link_mode": link_mode,
                "error": "",
            })
            counts[split] += 1
        except Exception as exc:
            rows.append({"src": str(src), "split": split, "image": "", "label": "", "conf": "", "num_boxes": "", "link_mode": "", "error": str(exc)})
            counts["failed"] += 1
        if idx % 100 == 0 or idx == len(selected):
            print(f"processed {idx}/{len(selected)} train={counts['train']} val={counts['val']} failed={counts['failed']}")

    yaml_path = args.out_dir / "microled.yaml"
    yaml_path.write_text(
        "\n".join([
            f"path: {args.out_dir.as_posix()}",
            "train: images/train",
            "val: images/val",
            "nc: 1",
            "names: ['microled']",
            "",
        ]),
        encoding="utf-8",
    )
    names_path = args.out_dir / "microled.names"
    names_path.write_text("microled\n", encoding="utf-8")
    (args.out_dir / "train.txt").write_text("\n".join(row["image"] for row in rows if row.get("split") == "train" and row.get("image")) + "\n", encoding="utf-8")
    (args.out_dir / "val.txt").write_text("\n".join(row["image"] for row in rows if row.get("split") == "val" and row.get("image")) + "\n", encoding="utf-8")
    (args.out_dir / "microled.data").write_text(
        "\n".join([
            "classes = 1",
            f"train = {(args.out_dir / 'train.txt').as_posix()}",
            f"valid = {(args.out_dir / 'val.txt').as_posix()}",
            f"names = {names_path.as_posix()}",
            f"backup = {(args.out_dir / 'darknet_backup').as_posix()}",
            "",
        ]),
        encoding="utf-8",
    )

    with (args.out_dir / "pseudo_label_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "split", "image", "label", "conf", "num_boxes", "link_mode", "error"])
        writer.writeheader()
        writer.writerows(rows)

    report = args.out_dir / "dataset_report.md"
    report.write_text(
        "\n".join([
            "# Micro-LED Lightweight Detector Dataset",
            "",
            f"- Teacher model: `{args.model_path}`",
            f"- Raw root: `{args.raw_root}`",
            f"- Selected images: {len(selected)}",
            f"- Train labels: {counts['train']}",
            f"- Val labels: {counts['val']}",
            f"- Failed pseudo-labels: {counts['failed']}",
            f"- Ultralytics YAML: `{yaml_path}`",
            f"- Darknet data file: `{args.out_dir / 'microled.data'}`",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"Dataset: {args.out_dir}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
