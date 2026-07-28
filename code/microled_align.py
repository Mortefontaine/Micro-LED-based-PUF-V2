"""YOLO crop + STN alignment for the release micro-LED PUF pipeline."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


IMAGE_SIZE = 256
PREDICTOR_SIZE = 96
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def natural_key(text: str) -> list[object]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def iter_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(
        [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: natural_key(str(p)),
    )


def relative_output_path(path: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root / path.with_suffix(".png").name
    return output_root / path.relative_to(input_root).with_suffix(".png")


def pil_rgb(path: Path, size: int = IMAGE_SIZE) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return img


def radial_residual_np(x: np.ndarray) -> np.ndarray:
    h, w = x.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    center = (h - 1) / 2.0
    bins = np.floor(np.sqrt((xx - center) ** 2 + (yy - center) ** 2)).astype(np.int32).ravel()
    flat = x.ravel()
    sums = np.bincount(bins, weights=flat)
    counts = np.bincount(bins)
    means = sums / np.maximum(counts, 1)
    return (flat - means[bins]).reshape(h, w).astype(np.float32)


def sobel_mag_np(x: np.ndarray) -> np.ndarray:
    xp = np.pad(x, 1, mode="edge")
    gx = -xp[:-2, :-2] - 2 * xp[1:-1, :-2] - xp[2:, :-2] + xp[:-2, 2:] + 2 * xp[1:-1, 2:] + xp[2:, 2:]
    gy = -xp[:-2, :-2] - 2 * xp[:-2, 1:-1] - xp[:-2, 2:] + xp[2:, :-2] + 2 * xp[2:, 1:-1] + xp[2:, 2:]
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    return ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)


def blue_predictor_features(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.resize((PREDICTOR_SIZE, PREDICTOR_SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    blue = arr[:, :, 2]
    residual = radial_residual_np(blue)
    dark = np.clip(-residual, 0, None)
    edge = sobel_mag_np(residual)
    return torch.from_numpy(np.stack([zscore(blue), zscore(residual), zscore(dark), zscore(edge)], axis=0).astype(np.float32))


class SpatialHeadSimilaritySTN(nn.Module):
    def __init__(self, max_angle_deg: float = 180.0, max_translate: float = 0.20, max_log_scale: float = 0.35) -> None:
        super().__init__()
        self.max_angle = math.radians(max_angle_deg)
        self.max_translate = max_translate
        self.max_log_scale = max_log_scale
        self.features = nn.Sequential(
            nn.Conv2d(4, 24, 5, stride=2, padding=2),
            nn.GroupNorm(6, 24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GroupNorm(12, 96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.05),
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 4),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.head(self.features(x))
        angle = self.max_angle * torch.tanh(raw[:, 0])
        scale = torch.exp(self.max_log_scale * torch.tanh(raw[:, 1]))
        tx = self.max_translate * torch.tanh(raw[:, 2])
        ty = self.max_translate * torch.tanh(raw[:, 3])
        c = torch.cos(angle) * scale
        s = torch.sin(angle) * scale
        theta = torch.zeros((x.shape[0], 2, 3), dtype=x.dtype, device=x.device)
        theta[:, 0, 0] = c
        theta[:, 0, 1] = -s
        theta[:, 1, 0] = s
        theta[:, 1, 1] = c
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        return theta, torch.stack([angle, scale, tx, ty], dim=1)


def load_stn(checkpoint_path: Path, device: torch.device) -> SpatialHeadSimilaritySTN:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = SpatialHeadSimilaritySTN(
        checkpoint.get("max_angle_deg", 180.0),
        checkpoint.get("max_translate", 0.20),
        checkpoint.get("max_log_scale", 0.35),
    ).to(device)
    state = checkpoint.get("model_state", checkpoint)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def stn_align_image(model: nn.Module, crop: Image.Image, device: torch.device) -> tuple[Image.Image, dict[str, float]]:
    crop = crop.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    predictor = blue_predictor_features(crop)[None].to(device)
    rgb = np.asarray(crop, dtype=np.float32) / 255.0
    source = torch.from_numpy(np.transpose(rgb, (2, 0, 1)))[None].to(device)
    theta, params = model(predictor)
    grid = F.affine_grid(theta, source.shape, align_corners=False)
    warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    arr = np.transpose(warped[0].detach().cpu().numpy(), (1, 2, 0))
    out = Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255))
    p = params[0].detach().cpu().numpy()
    return out, {
        "angle_deg": float(np.degrees(p[0])),
        "scale": float(p[1]),
        "tx_norm": float(p[2]),
        "ty_norm": float(p[3]),
    }


def yolo_square_crop(path: Path, model_path: Path, conf: float, device: str, crop_margin: float) -> Image.Image:
    from ultralytics import YOLO

    import cv2

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise RuntimeError(f"Unable to read image: {path}")
    result = YOLO(str(model_path)).predict(img_bgr, conf=conf, verbose=False, device=device)[0]
    boxes = result.boxes.xywh.cpu().numpy()
    if len(boxes) == 0:
        raise RuntimeError(f"No YOLO detection: {path}")
    x_center, y_center, w, h = boxes[0]
    side = max(float(max(w, h) * crop_margin), float(IMAGE_SIZE))
    x1 = max(0, int(x_center - side / 2))
    y1 = max(0, int(y_center - side / 2))
    x2 = min(img_bgr.shape[1], int(x_center + side / 2))
    y2 = min(img_bgr.shape[0], int(y_center + side / 2))
    crop_bgr = cv2.resize(img_bgr[y1:y2, x1:x2], (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


def default_models_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw image file/folder or already YOLO-cropped image file/folder.")
    parser.add_argument("--output", type=Path, required=True, help="Output folder for aligned 256x256 RGB PNGs.")
    parser.add_argument("--models-dir", type=Path, default=default_models_dir())
    parser.add_argument("--skip-yolo", action="store_true", help="Use input images as 256x256 crops and only apply STN.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--crop-margin", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--yolo-device", default="0" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    stn = load_stn(args.models_dir / "luma_spatial_head_stn.pt", device)
    yolo_model = args.models_dir / "yolo11n_microled_best.pt"
    rows = []
    for idx, path in enumerate(iter_images(args.input), 1):
        crop = pil_rgb(path) if args.skip_yolo else yolo_square_crop(path, yolo_model, args.conf, args.yolo_device, args.crop_margin)
        aligned, info = stn_align_image(stn, crop, device)
        out_path = relative_output_path(path, args.input, args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        aligned.save(out_path)
        rows.append((path, out_path, info))
        if idx % 100 == 0:
            print(f"aligned {idx}")
    print(f"aligned={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
