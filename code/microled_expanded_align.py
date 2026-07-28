"""YOLO11n localization plus expanded-background RGB STN alignment."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LOCAL_YOLO_DEPS = PACKAGE_ROOT / ".yolo_deps"
if LOCAL_YOLO_DEPS.is_dir():
    # Keep the environment's compatible torch/torchvision ahead of the local
    # target directory, while still making Ultralytics available.
    sys.path.append(str(LOCAL_YOLO_DEPS))
os.environ.setdefault("YOLO_CONFIG_DIR", str(PACKAGE_ROOT / ".ultralytics_config"))

import cv2  # noqa: E402

from microled_align import (  # noqa: E402
    IMAGE_SIZE,
    blue_predictor_features,
    load_stn,
)


def _pure_torch_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """CPU fallback for environments whose torchvision NMS binary is unavailable."""
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    x1, y1, x2, y2 = boxes.unbind(1)
    areas = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    order = scores.argsort(descending=True)
    keep: list[torch.Tensor] = []
    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        xx1 = torch.maximum(x1[current], x1[remaining])
        yy1 = torch.maximum(y1[current], y1[remaining])
        xx2 = torch.minimum(x2[current], x2[remaining])
        yy2 = torch.minimum(y2[current], y2[remaining])
        intersection = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[current] + areas[remaining] - intersection
        iou = intersection / union.clamp(min=torch.finfo(boxes.dtype).eps)
        order = remaining[iou <= iou_threshold]
    return torch.stack(keep).to(dtype=torch.long)


def detect_box(
    path: Path,
    detector,
    device: str,
    early_intensity: bool = False,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"Unable to read {path}")
    if early_intensity:
        values = bgr.astype(np.float32)
        luma = 0.0722 * values[:, :, 0] + 0.7152 * values[:, :, 1] + 0.2126 * values[:, :, 2]
        gray = np.uint8(np.clip(np.rint(luma), 0, 255))
        bgr = np.repeat(gray[:, :, None], 3, axis=2)
    result = detector.predict(bgr, conf=0.25, verbose=False, device=device)[0]
    boxes = result.boxes.xywh.cpu().numpy()
    if len(boxes) == 0:
        raise RuntimeError(f"No detection for {path}")
    return bgr, tuple(float(value) for value in boxes[0])


def square_bounds(
    center_x: float,
    center_y: float,
    requested_side: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    # Match the released YOLO crop exactly: truncate each floating boundary
    # independently instead of rounding the center/side first. A one-pixel
    # shift can change the STN angle branch for the near-fourfold cross.
    x1 = max(0, int(center_x - requested_side / 2))
    y1 = max(0, int(center_y - requested_side / 2))
    x2 = min(image_width, int(center_x + requested_side / 2))
    y2 = min(image_height, int(center_y + requested_side / 2))
    return x1, y1, x2, y2


def rgb_crop(bgr: np.ndarray, bounds: tuple[int, int, int, int], size: int) -> Image.Image:
    x1, y1, x2, y2 = bounds
    crop = cv2.resize(bgr[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))


@torch.no_grad()
def align_pair(
    model,
    bgr: np.ndarray,
    box: tuple[float, float, float, float],
    device: torch.device,
    expanded_margin: float,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image, dict[str, float]]:
    image_height, image_width = bgr.shape[:2]
    center_x, center_y, width, height = box
    tight_side = max(width, height, float(IMAGE_SIZE))
    tight_bounds = square_bounds(center_x, center_y, tight_side, image_width, image_height)
    expanded_bounds = square_bounds(
        center_x,
        center_y,
        tight_side * expanded_margin,
        image_width,
        image_height,
    )

    tight = rgb_crop(bgr, tight_bounds, IMAGE_SIZE)
    expanded_size = max(IMAGE_SIZE, int(round(IMAGE_SIZE * expanded_margin)))
    expanded = rgb_crop(bgr, expanded_bounds, expanded_size)

    predictor = blue_predictor_features(tight)[None].to(device)
    theta_tight, params = model(predictor)

    tight_rgb = np.asarray(tight, dtype=np.float32) / 255.0
    tight_source = torch.from_numpy(tight_rgb.transpose(2, 0, 1))[None].to(device)
    baseline_grid = F.affine_grid(theta_tight, tight_source.shape, align_corners=False)
    baseline = F.grid_sample(
        tight_source,
        baseline_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    tx1, ty1, tx2, ty2 = tight_bounds
    ex1, ey1, ex2, ey2 = expanded_bounds
    tight_w = float(tx2 - tx1)
    tight_h = float(ty2 - ty1)
    expanded_w = float(ex2 - ex1)
    expanded_h = float(ey2 - ey1)
    tight_cx = (tx1 + tx2) / 2.0
    tight_cy = (ty1 + ty2) / 2.0
    expanded_cx = (ex1 + ex2) / 2.0
    expanded_cy = (ey1 + ey2) / 2.0

    theta_expanded = theta_tight.clone()
    theta_expanded[:, 0, :2] *= tight_w / expanded_w
    theta_expanded[:, 1, :2] *= tight_h / expanded_h
    theta_expanded[:, 0, 2] = (
        (tight_cx - expanded_cx) / (expanded_w / 2.0)
        + theta_tight[:, 0, 2] * tight_w / expanded_w
    )
    theta_expanded[:, 1, 2] = (
        (tight_cy - expanded_cy) / (expanded_h / 2.0)
        + theta_tight[:, 1, 2] * tight_h / expanded_h
    )

    expanded_rgb = np.asarray(expanded, dtype=np.float32) / 255.0
    expanded_source = torch.from_numpy(expanded_rgb.transpose(2, 0, 1))[None].to(device)
    output_shape = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    expanded_grid = F.affine_grid(theta_expanded, output_shape, align_corners=False)
    corrected = F.grid_sample(
        expanded_source,
        expanded_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    valid = F.grid_sample(
        torch.ones((1, 1, expanded_size, expanded_size), dtype=torch.float32, device=device),
        expanded_grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    def to_image(tensor: torch.Tensor) -> Image.Image:
        array = tensor[0].detach().cpu().numpy().transpose(1, 2, 0)
        return Image.fromarray(np.uint8(np.clip(array, 0, 1) * 255))

    baseline_image = to_image(baseline)
    corrected_image = to_image(corrected)
    valid_array = valid[0, 0].detach().cpu().numpy()
    valid_image = Image.fromarray(np.uint8(np.clip(valid_array, 0, 1) * 255))
    p = params[0].detach().cpu().numpy()
    baseline_array = np.asarray(baseline_image)
    corrected_array = np.asarray(corrected_image)
    metrics = {
        "angle_deg": float(np.degrees(p[0])),
        "scale": float(p[1]),
        "tx_norm": float(p[2]),
        "ty_norm": float(p[3]),
        "tight_side_px": tight_w,
        "expanded_side_px": expanded_w,
        "baseline_exact_zero_fraction": float(np.all(baseline_array == 0, axis=2).mean()),
        "corrected_exact_zero_fraction": float(np.all(corrected_array == 0, axis=2).mean()),
        "corrected_valid_fraction": float((valid_array > 0.999).mean()),
    }
    return tight, baseline_image, corrected_image, valid_image, metrics


def panel(images: list[Image.Image], labels: list[str]) -> Image.Image:
    label_height = 28
    canvas = Image.new("RGB", (IMAGE_SIZE * len(images), IMAGE_SIZE + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (img, label) in enumerate(zip(images, labels)):
        canvas.paste(img.convert("RGB"), (index * IMAGE_SIZE, label_height))
        draw.text((index * IMAGE_SIZE + 6, 7), label, fill="black")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--detector-model",
        type=Path,
        default=PACKAGE_ROOT / "models" / "yolo11n_microled_best.pt",
        help="YOLO checkpoint produced by the detector-training stage.",
    )
    parser.add_argument(
        "--stn-model",
        type=Path,
        default=PACKAGE_ROOT / "models" / "luma_spatial_head_stn.pt",
        help="STN checkpoint produced by the alignment-training stage.",
    )
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=None,
        help="Process only relative paths represented by PNG files under this root.",
    )
    parser.add_argument("--margin", type=float, default=1.70)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.98,
        help="Fail closed instead of emitting an aligned frame when geometric source coverage is lower.",
    )
    parser.add_argument("--save-baseline", action="store_true")
    parser.add_argument(
        "--early-intensity",
        action="store_true",
        help="Convert raw RGB to Rec.709 intensity before both YOLO and STN prediction.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.margin < 1.0:
        parser.error("--margin must be at least 1.0.")
    if not 0.0 <= args.min_valid_fraction <= 1.0:
        parser.error("--min-valid-fraction must be within [0, 1].")

    from ultralytics import YOLO

    if args.device == "cpu" or not torch.cuda.is_available():
        # The packaged torchvision build can lack its compiled CPU NMS kernel.
        # Ultralytics still supplies all filtering/class logic; only the final
        # mathematically equivalent suppression primitive is replaced.
        import torchvision

        torchvision.ops.nms = _pure_torch_nms

    if args.selection_root is not None:
        selected = sorted(args.selection_root.rglob("*.png"))
        paths = [args.input / path.relative_to(args.selection_root).with_suffix(".jpg") for path in selected]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} selected raw images; first: {missing[0]}")
    else:
        paths = [args.input] if args.input.is_file() else sorted(args.input.rglob("*.jpg"))
    if args.limit:
        paths = paths[: args.limit]
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    detector_device = "0" if device.type == "cuda" else "cpu"
    if not args.detector_model.is_file():
        raise FileNotFoundError(f"Detector checkpoint not found: {args.detector_model}")
    if not args.stn_model.is_file():
        raise FileNotFoundError(f"STN checkpoint not found: {args.stn_model}")
    detector = YOLO(str(args.detector_model))
    stn = load_stn(args.stn_model, device)

    rows: list[dict[str, object]] = []
    for path in paths:
        bgr, box = detect_box(path, detector, detector_device, early_intensity=args.early_intensity)
        tight, baseline, corrected, valid, metrics = align_pair(stn, bgr, box, device, args.margin)
        relative = path.relative_to(args.input) if args.input.is_dir() else Path(path.name)
        aligned_path = args.output / "aligned_rgb_256" / relative.with_suffix(".png")
        valid_path = args.output / "valid_masks_256" / relative.with_suffix(".png")
        valid_path.parent.mkdir(parents=True, exist_ok=True)
        valid.save(valid_path)
        alignment_pass = float(metrics["corrected_valid_fraction"]) >= args.min_valid_fraction
        metrics["alignment_pass"] = alignment_pass
        metrics["aligned_path"] = str(aligned_path) if alignment_pass else ""
        metrics["valid_mask_path"] = str(valid_path)
        if alignment_pass:
            aligned_path.parent.mkdir(parents=True, exist_ok=True)
            corrected.save(aligned_path)
        if args.save_baseline:
            baseline_path = args.output / "baseline_zero_pad_rgb_256" / relative.with_suffix(".png")
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline.save(baseline_path)
        if len(rows) < args.preview_count:
            stem = f"{path.parent.name}__{path.stem}"
            preview_root = args.output / "previews"
            preview_root.mkdir(parents=True, exist_ok=True)
            tight.save(preview_root / f"{stem}__tight_crop.png")
            baseline.save(preview_root / f"{stem}__baseline.png")
            corrected.save(preview_root / f"{stem}__expanded.png")
            valid.save(preview_root / f"{stem}__valid_mask.png")
            panel(
                [tight, baseline, corrected, valid],
                ["YOLO tight crop", "Current zero-pad", "Expanded real bg", "Valid mask"],
            ).save(preview_root / f"{stem}__comparison.png")
        rows.append({"path": str(path), **metrics})

    if not rows:
        raise SystemExit("No input JPG images were found.")
    with (args.output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"processed={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
