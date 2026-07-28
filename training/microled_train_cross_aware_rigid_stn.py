"""Train a blue-derived cross-aware rigid STN for micro-LED alignment.

This is a conservative follow-up to the similarity STN:

- input features are derived only from blue intensity
- output remains a 256x256 blue/intensity image
- transform is restricted to rotation + small translation, no scale
- loss includes blue intensity, radial residual patch, and dark-residual maps
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from microled_train_similarity_stn_256 import (
    IMAGE_SIZE,
    PAIR_CSV,
    PairRow,
    blue_tensor,
    load_pairs,
    split_pairs,
    warp_with_theta,
    zncc,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "work" / "cross_aware_rigid_stn"
CROP_ROOT = REPO_ROOT / "data" / "02_stn_pairs_M1_M6" / "input"
PREDICTOR_SIZE = 96


def get_font(size: int) -> ImageFont.ImageFont:
    for candidate in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int = 14, fill=(30, 30, 30)) -> None:
    draw.text(xy, text, font=get_font(size), fill=fill)


def pil_blue(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)[:, :, 2] / 255.0


def radial_residual_np(blue: np.ndarray) -> np.ndarray:
    h, w = blue.shape
    yy, xx = np.indices((h, w), dtype=np.float32)
    center = (h - 1) / 2.0
    bins = np.floor(np.sqrt((xx - center) ** 2 + (yy - center) ** 2)).astype(np.int32).ravel()
    flat = blue.ravel()
    sums = np.bincount(bins, weights=flat)
    counts = np.bincount(bins)
    means = sums / np.maximum(counts, 1)
    return (flat - means[bins]).reshape(h, w).astype(np.float32)


def sobel_mag_np(x: np.ndarray) -> np.ndarray:
    # Small manual Sobel implementation to avoid extra image-processing deps.
    xp = np.pad(x, 1, mode="edge")
    gx = (
        -xp[:-2, :-2]
        - 2 * xp[1:-1, :-2]
        - xp[2:, :-2]
        + xp[:-2, 2:]
        + 2 * xp[1:-1, 2:]
        + xp[2:, 2:]
    )
    gy = (
        -xp[:-2, :-2]
        - 2 * xp[:-2, 1:-1]
        - xp[:-2, 2:]
        + xp[2:, :-2]
        + 2 * xp[2:, 1:-1]
        + xp[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def zscore(x: np.ndarray) -> np.ndarray:
    return ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)


def predictor_features(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((PREDICTOR_SIZE, PREDICTOR_SIZE), Image.Resampling.BILINEAR)
    blue = np.asarray(img, dtype=np.float32)[:, :, 2] / 255.0
    residual = radial_residual_np(blue)
    dark = np.clip(-residual, 0, None)
    edge = sobel_mag_np(residual)
    arr = np.stack([zscore(blue), zscore(residual), zscore(dark), zscore(edge)], axis=0)
    return torch.from_numpy(arr)


class CrossAwareDataset(Dataset):
    def __init__(self, rows: Sequence[PairRow]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows[idx]
        return {
            "predictor": predictor_features(row.crop_path),
            "source_blue": blue_tensor(row.crop_path),
            "target_blue": blue_tensor(row.target_path),
            "condition": row.condition,
            "frame": row.frame,
            "crop_path": str(row.crop_path),
            "target_path": str(row.target_path),
        }


class CrossAwareRigidSTN(nn.Module):
    def __init__(self, max_angle_deg: float = 180.0, max_translate: float = 0.16) -> None:
        super().__init__()
        self.max_angle = math.radians(max_angle_deg)
        self.max_translate = max_translate
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
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 96),
            nn.SiLU(inplace=True),
            nn.Linear(96, 3),
        )
        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.head(self.features(x))
        angle = self.max_angle * torch.tanh(raw[:, 0])
        tx = self.max_translate * torch.tanh(raw[:, 1])
        ty = self.max_translate * torch.tanh(raw[:, 2])
        c = torch.cos(angle)
        s = torch.sin(angle)
        theta = torch.zeros((x.shape[0], 2, 3), dtype=x.dtype, device=x.device)
        theta[:, 0, 0] = c
        theta[:, 0, 1] = -s
        theta[:, 1, 0] = s
        theta[:, 1, 1] = c
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        params = torch.stack([angle, tx, ty], dim=1)
        return theta, params


class CrossAwareSimilaritySTN(nn.Module):
    def __init__(self, max_angle_deg: float = 180.0, max_translate: float = 0.16, max_log_scale: float = 0.28) -> None:
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
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 96),
            nn.SiLU(inplace=True),
            nn.Linear(96, 4),
        )
        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

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
        params = torch.stack([angle, scale, tx, ty], dim=1)
        return theta, params


def radial_residual_patch_tensor(x: torch.Tensor, patch_grid: int = 32) -> torch.Tensor:
    b, c, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=torch.float32),
        torch.arange(w, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    center = (h - 1) / 2.0
    bins = torch.floor(torch.sqrt((yy - center).square() + (xx - center).square())).long().flatten()
    n_bins = int(bins.max().item()) + 1
    flat = x[:, 0].flatten(1)
    sums = torch.zeros((b, n_bins), dtype=x.dtype, device=x.device)
    sums.scatter_add_(1, bins.expand(b, -1), flat)
    counts = torch.bincount(bins, minlength=n_bins).to(device=x.device, dtype=x.dtype).clamp_min(1.0)
    means = sums / counts[None, :]
    residual = flat - means.gather(1, bins.expand(b, -1))
    block = h // patch_grid
    patch = residual.reshape(b, patch_grid, block, patch_grid, block).mean(dim=(2, 4))
    return patch.flatten(1)


def dark_residual_tensor(x: torch.Tensor) -> torch.Tensor:
    # Full-resolution radial residual dark map for cross-aware alignment loss.
    b, c, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=torch.float32),
        torch.arange(w, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    center = (h - 1) / 2.0
    bins = torch.floor(torch.sqrt((yy - center).square() + (xx - center).square())).long().flatten()
    n_bins = int(bins.max().item()) + 1
    flat = x[:, 0].flatten(1)
    sums = torch.zeros((b, n_bins), dtype=x.dtype, device=x.device)
    sums.scatter_add_(1, bins.expand(b, -1), flat)
    counts = torch.bincount(bins, minlength=n_bins).to(device=x.device, dtype=x.dtype).clamp_min(1.0)
    means = sums / counts[None, :]
    residual = flat - means.gather(1, bins.expand(b, -1))
    return torch.relu(-residual).reshape(b, 1, h, w)


def alignment_loss(warped: torch.Tensor, target: torch.Tensor, params: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    corr = zncc(warped, target)
    patch_corr = zncc(radial_residual_patch_tensor(warped), radial_residual_patch_tensor(target))
    dark_corr = zncc(dark_residual_tensor(warped), dark_residual_tensor(target))
    l1 = F.smooth_l1_loss(warped, target)
    angle = params[:, 0]
    if params.shape[1] == 4:
        scale = params[:, 1]
        tx = params[:, 2]
        ty = params[:, 3]
        scale_reg = 0.001 * torch.log(scale).square().mean()
    else:
        tx = params[:, 1]
        ty = params[:, 2]
        scale_reg = torch.zeros((), dtype=params.dtype, device=params.device)
    reg = 0.001 * ((angle / math.pi) ** 2).mean() + 0.002 * (tx.square().mean() + ty.square().mean())
    reg = reg + scale_reg
    loss = (1 - corr.mean()) + 0.45 * (1 - patch_corr.mean()) + 0.25 * (1 - dark_corr.mean()) + 0.10 * l1 + reg
    return loss, {
        "corr": float(corr.mean().detach().cpu()),
        "patch_corr": float(patch_corr.mean().detach().cpu()),
        "dark_corr": float(dark_corr.mean().detach().cpu()),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    before, after, angles, scales, txs, tys = [], [], [], [], [], []
    for batch in loader:
        predictor = batch["predictor"].to(device)
        source = batch["source_blue"].to(device)
        target = batch["target_blue"].to(device)
        theta, params = model(predictor)
        warped = warp_with_theta(source, theta)
        before.extend(zncc(source, target).detach().cpu().tolist())
        after.extend(zncc(warped, target).detach().cpu().tolist())
        angles.extend(torch.rad2deg(params[:, 0]).detach().cpu().tolist())
        if params.shape[1] == 4:
            scales.extend(params[:, 1].detach().cpu().tolist())
            txs.extend(params[:, 2].detach().cpu().tolist())
            tys.extend(params[:, 3].detach().cpu().tolist())
        else:
            scales.extend([1.0] * params.shape[0])
            txs.extend(params[:, 1].detach().cpu().tolist())
            tys.extend(params[:, 2].detach().cpu().tolist())
    return {
        "before": float(np.mean(before)),
        "after": float(np.mean(after)),
        "improvement": float(np.mean(after) - np.mean(before)),
        "angle_abs_mean": float(np.mean(np.abs(angles))),
        "angle_mean": float(np.mean(angles)),
        "angle_std": float(np.std(angles)),
        "scale_mean": float(np.mean(scales)),
        "scale_std": float(np.std(scales)),
        "tx_px_mean": float(np.mean(txs) * IMAGE_SIZE / 2.0),
        "ty_px_mean": float(np.mean(tys) * IMAGE_SIZE / 2.0),
    }


def blue_tensor_to_l_image(x: torch.Tensor) -> Image.Image:
    arr = np.clip(x.detach().cpu().numpy(), 0, 1)
    return Image.fromarray(np.uint8(arr * 255), "L")


@torch.no_grad()
def write_outputs(
    model: nn.Module,
    rows: Sequence[PairRow],
    out_root: Path,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, object]]:
    out_root.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(CrossAwareDataset(rows), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    summary = []
    for batch in loader:
        predictor = batch["predictor"].to(device)
        source = batch["source_blue"].to(device)
        target = batch["target_blue"].to(device)
        theta, params = model(predictor)
        warped = warp_with_theta(source, theta)
        before = zncc(source, target).detach().cpu().tolist()
        after = zncc(warped, target).detach().cpu().tolist()
        angles = torch.rad2deg(params[:, 0]).detach().cpu().tolist()
        if params.shape[1] == 4:
            scales = params[:, 1].detach().cpu().tolist()
            txs = params[:, 2].detach().cpu().tolist()
            tys = params[:, 3].detach().cpu().tolist()
        else:
            scales = [1.0] * params.shape[0]
            txs = params[:, 1].detach().cpu().tolist()
            tys = params[:, 2].detach().cpu().tolist()
        for i in range(warped.shape[0]):
            crop_path = Path(batch["crop_path"][i])
            try:
                rel = crop_path.relative_to(CROP_ROOT)
            except ValueError:
                rel = Path(str(batch["condition"][i])) / Path(str(batch["frame"][i])).with_suffix(".png")
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            blue_tensor_to_l_image(warped[i, 0]).save(out_path)
            summary.append(
                {
                    "aligned_path": str(out_path),
                    "crop_path": str(crop_path),
                    "target_path": batch["target_path"][i],
                    "condition": batch["condition"][i],
                    "frame": batch["frame"][i],
                    "before_corr": before[i],
                    "after_corr": after[i],
                    "angle_deg": angles[i],
                    "scale": scales[i],
                    "tx_px": txs[i] * IMAGE_SIZE / 2.0,
                    "ty_px": tys[i] * IMAGE_SIZE / 2.0,
                }
            )
    return summary


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def make_preview(rows: Sequence[dict[str, object]], out_path: Path) -> None:
    rows = list(rows)[:24]
    cols = 3
    tile_w, tile_h = 900, 286
    canvas = Image.new("RGB", (cols * tile_w, math.ceil(len(rows) / cols) * tile_h + 76), "white")
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (28, 20), "Cross-aware rigid STN: YOLO blue / aligned blue / target blue", 25)
    for i, row in enumerate(rows):
        crop = Image.open(row["crop_path"]).convert("RGB")
        crop_b = Image.fromarray(np.asarray(crop)[:, :, 2]).convert("RGB").resize((150, 150), Image.Resampling.NEAREST)
        aligned = Image.open(row["aligned_path"]).convert("L").convert("RGB").resize((150, 150), Image.Resampling.NEAREST)
        target = Image.open(row["target_path"]).convert("RGB")
        target_b = Image.fromarray(np.asarray(target)[:, :, 2]).convert("RGB").resize((150, 150), Image.Resampling.NEAREST)
        x = (i % cols) * tile_w + 18
        y = (i // cols) * tile_h + 78
        for j, (label, img) in enumerate((("YOLO blue", crop_b), ("aligned blue", aligned), ("target blue", target_b))):
            canvas.paste(img, (x + j * 180, y))
            draw_text(draw, (x + j * 180, y + 156), label, 12)
        draw_text(
            draw,
            (x, y + 180),
            f"corr {float(row['before_corr']):.3f}->{float(row['after_corr']):.3f}, angle={float(row['angle_deg']):.1f}",
            11,
            (70, 70, 70),
        )
        draw_text(draw, (x, y + 200), f"{row['condition']}/{row['frame']}", 10, (80, 80, 80))
    canvas.save(out_path, quality=95)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-csv", type=Path, default=PAIR_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-angle-deg", type=float, default=180.0)
    parser.add_argument("--max-translate", type=float, default=0.16)
    parser.add_argument("--allow-scale", action="store_true")
    parser.add_argument("--max-log-scale", type=float, default=0.28)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows = load_pairs(args.pair_csv)
    train_rows, val_rows = split_pairs(rows, args.val_ratio, args.seed)
    train_loader = DataLoader(CrossAwareDataset(train_rows), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(CrossAwareDataset(val_rows), batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.allow_scale:
        model: nn.Module = CrossAwareSimilaritySTN(args.max_angle_deg, args.max_translate, args.max_log_scale).to(device)
        model_name = "CrossAwareSimilaritySTN"
    else:
        model = CrossAwareRigidSTN(args.max_angle_deg, args.max_translate).to(device)
        model_name = "CrossAwareRigidSTN"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_after = -1e9
    best_path = args.out_dir / "best_cross_aware_rigid_stn.pt"
    history = []
    print(f"device={device} train={len(train_rows)} val={len(val_rows)} params={parameter_count(model)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            predictor = batch["predictor"].to(device)
            source = batch["source_blue"].to(device)
            target = batch["target_blue"].to(device)
            theta, params = model(predictor)
            warped = warp_with_theta(source, theta)
            loss, parts = alignment_loss(warped, target, params)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['loss']:.4f} val {metrics['before']:.4f}->{metrics['after']:.4f} "
            f"gain={metrics['improvement']:.4f} angle_abs={metrics['angle_abs_mean']:.2f}"
        )
        if metrics["after"] > best_after:
            best_after = metrics["after"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model": model_name,
                    "params": parameter_count(model),
                    "epoch": epoch,
                    "val_metrics": metrics,
                    "max_angle_deg": args.max_angle_deg,
                    "max_translate": args.max_translate,
                    "max_log_scale": args.max_log_scale if args.allow_scale else 0.0,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    aligned_root = args.out_dir / "aligned_blue_256"
    summary = write_outputs(model, rows, aligned_root, device, args.batch_size)
    write_csv(
        args.out_dir / "cross_aware_alignment_summary.csv",
        summary,
        ["aligned_path", "crop_path", "target_path", "condition", "frame", "before_corr", "after_corr", "angle_deg", "scale", "tx_px", "ty_px"],
    )
    write_csv(
        args.out_dir / "training_history.csv",
        history,
        ["epoch", "loss", "before", "after", "improvement", "angle_abs_mean", "angle_mean", "angle_std", "scale_mean", "scale_std", "tx_px_mean", "ty_px_mean"],
    )
    make_preview(summary, args.out_dir / "cross_aware_alignment_preview.jpg")

    before = np.asarray([float(row["before_corr"]) for row in summary], dtype=np.float32)
    after = np.asarray([float(row["after_corr"]) for row in summary], dtype=np.float32)
    angles = np.asarray([float(row["angle_deg"]) for row in summary], dtype=np.float32)
    report = args.out_dir / "cross_aware_rigid_stn_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write(f"# {model_name} Report\n\n")
        f.write("## Method\n\n")
        f.write("- Input features: blue, radial residual, dark residual, Sobel magnitude.\n")
        f.write("- Transform: rotation + small translation")
        f.write(" + limited scale.\n" if args.allow_scale else ", no scale.\n")
        f.write("- Output: 256x256 blue-only image, condition folders preserved.\n\n")
        f.write("## Training\n\n")
        f.write(f"- Train/val pairs: {len(train_rows)} / {len(val_rows)}\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Parameters: {parameter_count(model)}\n")
        f.write(f"- Best checkpoint: `{best_path}`\n\n")
        f.write("## Alignment\n\n")
        f.write(f"- Before correlation mean/std: {before.mean():.4f} / {before.std():.4f}\n")
        f.write(f"- After correlation mean/std: {after.mean():.4f} / {after.std():.4f}\n")
        f.write(f"- Mean improvement: {(after - before).mean():.4f}\n")
        f.write(f"- Angle mean/std/min/max: {angles.mean():.3f} / {angles.std():.3f} / {angles.min():.3f} / {angles.max():.3f} deg\n")
        f.write(f"- Aligned root: `{aligned_root}`\n")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
