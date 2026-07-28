"""Prepare paired YOLO11n crops for training a 256-preserving affine aligner.

The output keeps the same condition-folder structure as the downstream bit
extraction pipeline:

    aligned_root/M1_10mA_20C_0/frame_0001.png

Pairs are generated as:

    raw augmented frame -> YOLO11n square crop -> target crop from frames_selected_by_YOLO
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
os.environ.setdefault("YOLO_CONFIG_DIR", str(REPO_ROOT / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib"))

import torch

import cv2
import numpy as np
from ultralytics import YOLO

from microled_register_raw_frames_level2 import list_images, natural_key


RAW_ROOT = DATA_ROOT / "02_stn_pairs_M1_M6" / "input"
TARGET_ROOT = DATA_ROOT / "02_stn_pairs_M1_M6" / "target"
MODEL_PATH = REPO_ROOT / "models" / "yolo11n_microled_best.pt"
OUT_DIR = REPO_ROOT / "work" / "stn_training_dataset"
BEST9 = tuple(f"M{i}" for i in range(1, 10))
IMAGE_SIZE = 256


def condition_device(condition: str) -> str:
    return condition.split("_", 1)[0].upper()


def find_pairs(raw_root: Path, target_root: Path) -> list[tuple[Path, Path]]:
    target_set = {p.relative_to(target_root).as_posix(): p for p in list_images(target_root)}
    pairs = []
    for raw in list_images(raw_root):
        rel = raw.relative_to(raw_root).as_posix()
        target = target_set.get(rel)
        if target is not None:
            pairs.append((raw, target))
    return sorted(pairs, key=lambda pair: natural_key(str(pair[0])))


def select_pairs(
    pairs: list[tuple[Path, Path]],
    raw_root: Path,
    devices: set[str] | None,
    max_per_condition: int,
    seed: int,
) -> list[tuple[Path, Path]]:
    grouped: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for raw, target in pairs:
        condition = raw.relative_to(raw_root).parent.as_posix()
        if devices is not None and condition_device(condition) not in devices:
            continue
        grouped[condition].append((raw, target))

    rng = random.Random(seed)
    selected = []
    for condition, items in sorted(grouped.items(), key=lambda kv: natural_key(kv[0])):
        items = sorted(items, key=lambda pair: natural_key(pair[0].name))
        if max_per_condition > 0 and len(items) > max_per_condition:
            items = sorted(rng.sample(items, max_per_condition), key=lambda pair: natural_key(pair[0].name))
        selected.extend(items)
    return selected


def yolo_crop(
    model: YOLO,
    img_path: Path,
    conf: float,
    image_size: int,
    device: str,
    crop_margin: float,
) -> tuple[np.ndarray, dict[str, float]]:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise RuntimeError(f"Unable to read image: {img_path}")
    result = model.predict(img_bgr, conf=conf, verbose=False, device=device)[0]
    boxes = result.boxes.xywh.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy() if result.boxes is not None and result.boxes.conf is not None else np.asarray([])
    if len(boxes) == 0:
        raise RuntimeError("No YOLO target detected")
    x_center, y_center, w, h = boxes[0]
    side = float(max(w, h) * crop_margin)
    side = max(side, float(image_size))
    x1 = int(x_center - side / 2)
    y1 = int(y_center - side / 2)
    x2 = int(x_center + side / 2)
    y2 = int(y_center + side / 2)
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_bgr.shape[1], x2)
    y2 = min(img_bgr.shape[0], y2)
    crop = cv2.resize(img_bgr[y1:y2, x1:x2], (image_size, image_size), interpolation=cv2.INTER_AREA)
    info = {
        "conf": float(confs[0]) if len(confs) else float("nan"),
        "x_center": float(x_center),
        "y_center": float(y_center),
        "box_w": float(w),
        "box_h": float(h),
        "crop_side": float(side),
        "crop_margin": float(crop_margin),
        "num_boxes": float(len(boxes)),
    }
    return crop, info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--target-root", type=Path, default=TARGET_ROOT)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--devices", nargs="*", default=list(BEST9))
    parser.add_argument("--max-per-condition", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--crop-margin", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    crop_root = args.out_dir / "yolo11n_crops_256"
    device_set = {d.upper() for d in args.devices} if args.devices else None

    pairs = select_pairs(
        find_pairs(args.raw_root, args.target_root),
        args.raw_root,
        device_set,
        args.max_per_condition,
        args.seed,
    )
    print(f"selected pairs={len(pairs)} device={args.device}")
    model = YOLO(str(args.model_path))

    rows = []
    failed = 0
    for idx, (raw_path, target_path) in enumerate(pairs, 1):
        rel = raw_path.relative_to(args.raw_root)
        crop_path = crop_root / rel.with_suffix(".png")
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            crop_bgr, info = yolo_crop(model, raw_path, args.conf, args.image_size, args.device, args.crop_margin)
            cv2.imwrite(str(crop_path), crop_bgr)
            rows.append(
                {
                    "raw_path": str(raw_path),
                    "target_path": str(target_path),
                    "crop_path": str(crop_path),
                    "condition": rel.parent.as_posix(),
                    "frame": rel.name,
                    "error": "",
                    **info,
                }
            )
        except Exception as exc:
            failed += 1
            rows.append(
                {
                    "raw_path": str(raw_path),
                    "target_path": str(target_path),
                    "crop_path": "",
                    "condition": rel.parent.as_posix(),
                    "frame": rel.name,
                    "error": str(exc),
                }
            )
        if idx % 100 == 0 or idx == len(pairs):
            print(f"processed {idx}/{len(pairs)} failed={failed}")

    fieldnames = [
        "raw_path",
        "target_path",
        "crop_path",
        "condition",
        "frame",
        "conf",
        "x_center",
        "y_center",
        "box_w",
        "box_h",
        "crop_side",
        "crop_margin",
        "num_boxes",
        "error",
    ]
    with (args.out_dir / "alignment_pairs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    with (args.out_dir / "alignment_dataset_report.md").open("w", encoding="utf-8") as f:
        f.write("# Micro-LED Affine Alignment Dataset\n\n")
        f.write(f"- Raw root: `{args.raw_root}`\n")
        f.write(f"- Target root: `{args.target_root}`\n")
        f.write(f"- YOLO11n model: `{args.model_path}`\n")
        f.write(f"- Crop root: `{crop_root}`\n")
        f.write(f"- Pair CSV: `{args.out_dir / 'alignment_pairs.csv'}`\n")
        f.write(f"- Selected pairs: {len(pairs)}\n")
        f.write(f"- Failed crops: {failed}\n")
        f.write(f"- Devices: {', '.join(args.devices) if args.devices else 'all'}\n")
        f.write(f"- Max per condition: {args.max_per_condition}\n")
        f.write(f"- Crop margin: {args.crop_margin}\n")
    print(f"Report: {args.out_dir / 'alignment_dataset_report.md'}")


if __name__ == "__main__":
    main()
