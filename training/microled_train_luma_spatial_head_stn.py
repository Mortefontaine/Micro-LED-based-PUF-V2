"""Train a luma-aware spatial-head STN and preserve RGB aligned output.

The previous spatial-head STN predicted a good similarity transform from
blue-derived features, but the final PUF extractor uses Rec.709 luma. This
variant aligns the training objective with the final extractor by using luma
features and luma residual losses while still warping the full RGB crop.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from microled_train_cross_aware_rigid_stn import (
    dark_residual_tensor,
    parameter_count,
    predictor_features,
    radial_residual_np,
    radial_residual_patch_tensor,
    sobel_mag_np,
    zscore,
)
from microled_train_similarity_stn_256 import IMAGE_SIZE, PairRow, load_pairs, split_pairs, warp_with_theta, zncc
from microled_train_spatial_head_stn import SpatialHeadSimilaritySTN


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_CSV = REPO_ROOT / "data" / "02_stn_pairs_M1_M6" / "alignment_pairs_M1_M6.csv"
OUT_DIR = REPO_ROOT / "work" / "luma_spatial_head_stn"
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


def rgb01(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def luma01(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]).astype(np.float32)


def rgb_tensor(path: Path) -> torch.Tensor:
    arr = rgb01(path, IMAGE_SIZE)
    return torch.from_numpy(np.transpose(arr, (2, 0, 1)))


def luma_tensor(path: Path) -> torch.Tensor:
    return torch.from_numpy(luma01(rgb01(path, IMAGE_SIZE)))[None, :, :]


def luma_predictor_features(path: Path) -> torch.Tensor:
    rgb = rgb01(path, PREDICTOR_SIZE)
    y = luma01(rgb)
    residual = radial_residual_np(y)
    dark = np.clip(-residual, 0, None)
    edge = sobel_mag_np(residual)
    arr = np.stack([zscore(y), zscore(residual), zscore(dark), zscore(edge)], axis=0)
    return torch.from_numpy(arr.astype(np.float32))


class LumaAlignmentDataset(Dataset):
    def __init__(self, rows: Sequence[PairRow], predictor_source: str = "luma") -> None:
        self.rows = list(rows)
        self.predictor_source = predictor_source

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows[idx]
        predictor = predictor_features(row.crop_path) if self.predictor_source == "blue" else luma_predictor_features(row.crop_path)
        return {
            "predictor": predictor,
            "source_rgb": rgb_tensor(row.crop_path),
            "source_luma": luma_tensor(row.crop_path),
            "target_luma": luma_tensor(row.target_path),
            "condition": row.condition,
            "frame": row.frame,
            "crop_path": str(row.crop_path),
            "target_path": str(row.target_path),
        }


def warp_rgb_with_theta(rgb: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    grid = F.affine_grid(theta, rgb.shape, align_corners=False)
    return F.grid_sample(rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def luma_alignment_loss(warped_luma: torch.Tensor, target_luma: torch.Tensor, params: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    corr = zncc(warped_luma, target_luma)
    patch_corr = zncc(radial_residual_patch_tensor(warped_luma), radial_residual_patch_tensor(target_luma))
    dark_corr = zncc(dark_residual_tensor(warped_luma), dark_residual_tensor(target_luma))
    l1 = F.smooth_l1_loss(warped_luma, target_luma)
    angle = params[:, 0]
    scale = params[:, 1]
    tx = params[:, 2]
    ty = params[:, 3]
    reg = (
        0.001 * ((angle / math.pi) ** 2).mean()
        + 0.0015 * torch.log(scale).square().mean()
        + 0.002 * (tx.square().mean() + ty.square().mean())
    )
    loss = (1 - corr.mean()) + 0.55 * (1 - patch_corr.mean()) + 0.35 * (1 - dark_corr.mean()) + 0.08 * l1 + reg
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
        source_luma = batch["source_luma"].to(device)
        target_luma = batch["target_luma"].to(device)
        theta, params = model(predictor)
        warped = warp_with_theta(source_luma, theta)
        before.extend(zncc(source_luma, target_luma).detach().cpu().tolist())
        after.extend(zncc(warped, target_luma).detach().cpu().tolist())
        angles.extend(torch.rad2deg(params[:, 0]).detach().cpu().tolist())
        scales.extend(params[:, 1].detach().cpu().tolist())
        txs.extend((params[:, 2] * IMAGE_SIZE / 2.0).detach().cpu().tolist())
        tys.extend((params[:, 3] * IMAGE_SIZE / 2.0).detach().cpu().tolist())
    return {
        "before": float(np.mean(before)),
        "after": float(np.mean(after)),
        "improvement": float(np.mean(after) - np.mean(before)),
        "after_p01": float(np.percentile(after, 1)),
        "after_p05": float(np.percentile(after, 5)),
        "angle_abs_mean": float(np.mean(np.abs(angles))),
        "angle_mean": float(np.mean(angles)),
        "angle_std": float(np.std(angles)),
        "scale_mean": float(np.mean(scales)),
        "scale_std": float(np.std(scales)),
        "tx_px_mean": float(np.mean(txs)),
        "ty_px_mean": float(np.mean(tys)),
    }


def tensor_to_rgb_image(x: torch.Tensor) -> Image.Image:
    arr = np.transpose(x.detach().cpu().numpy(), (1, 2, 0))
    return Image.fromarray(np.uint8(np.clip(arr, 0, 1) * 255), "RGB")


@torch.no_grad()
def write_outputs(
    model: nn.Module,
    rows: Sequence[PairRow],
    out_root: Path,
    device: torch.device,
    batch_size: int,
    predictor_source: str = "luma",
) -> list[dict[str, object]]:
    out_root.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(LumaAlignmentDataset(rows, predictor_source), batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    summary = []
    for batch in loader:
        predictor = batch["predictor"].to(device)
        source_rgb = batch["source_rgb"].to(device)
        source_luma = batch["source_luma"].to(device)
        target_luma = batch["target_luma"].to(device)
        theta, params = model(predictor)
        warped_rgb = warp_rgb_with_theta(source_rgb, theta)
        warped_luma = (
            0.2126 * warped_rgb[:, 0:1]
            + 0.7152 * warped_rgb[:, 1:2]
            + 0.0722 * warped_rgb[:, 2:3]
        )
        before = zncc(source_luma, target_luma).detach().cpu().tolist()
        after = zncc(warped_luma, target_luma).detach().cpu().tolist()
        angles = torch.rad2deg(params[:, 0]).detach().cpu().tolist()
        scales = params[:, 1].detach().cpu().tolist()
        txs = (params[:, 2] * IMAGE_SIZE / 2.0).detach().cpu().tolist()
        tys = (params[:, 3] * IMAGE_SIZE / 2.0).detach().cpu().tolist()
        for i in range(warped_rgb.shape[0]):
            condition = str(batch["condition"][i])
            frame = Path(str(batch["frame"][i])).with_suffix(".png").name
            out_path = out_root / condition / frame
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tensor_to_rgb_image(warped_rgb[i]).save(out_path)
            summary.append(
                {
                    "aligned_path": str(out_path),
                    "crop_path": batch["crop_path"][i],
                    "target_path": batch["target_path"][i],
                    "condition": condition,
                    "frame": batch["frame"][i],
                    "before_corr": before[i],
                    "after_corr": after[i],
                    "angle_deg": angles[i],
                    "scale": scales[i],
                    "tx_px": txs[i],
                    "ty_px": tys[i],
                }
            )
    return summary


def write_csv(path: Path, rows: Sequence[dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def make_preview(summary: Sequence[dict[str, object]], out_path: Path) -> None:
    rows = list(summary)[:24]
    cols = 3
    tile_w, tile_h = 900, 286
    canvas = Image.new("RGB", (cols * tile_w, math.ceil(len(rows) / cols) * tile_h + 76), "white")
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (28, 20), "Luma-aware spatial-head STN: YOLO crop / aligned RGB / target RGB", 25)
    for i, row in enumerate(rows):
        crop = Image.open(row["crop_path"]).convert("RGB").resize((150, 150), Image.Resampling.NEAREST)
        aligned = Image.open(row["aligned_path"]).convert("RGB").resize((150, 150), Image.Resampling.NEAREST)
        target = Image.open(row["target_path"]).convert("RGB").resize((150, 150), Image.Resampling.NEAREST)
        x = (i % cols) * tile_w + 18
        y = (i // cols) * tile_h + 78
        for j, (label, img) in enumerate((("YOLO crop", crop), ("aligned RGB", aligned), ("target RGB", target))):
            canvas.paste(img, (x + j * 180, y))
            draw_text(draw, (x + j * 180, y + 156), label, 12)
        draw_text(
            draw,
            (x, y + 180),
            f"luma corr {float(row['before_corr']):.3f}->{float(row['after_corr']):.3f}, angle={float(row['angle_deg']):.1f}",
            11,
            (70, 70, 70),
        )
        draw_text(draw, (x, y + 200), f"{row['condition']}/{row['frame']}", 10, (80, 80, 80))
    canvas.save(out_path, quality=95)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-csv", type=Path, default=PAIR_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=26)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-angle-deg", type=float, default=180.0)
    parser.add_argument("--max-translate", type=float, default=0.20)
    parser.add_argument("--max-log-scale", type=float, default=0.35)
    parser.add_argument("--predictor-source", choices=["luma", "blue"], default="luma")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
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
    train_loader = DataLoader(LumaAlignmentDataset(train_rows, args.predictor_source), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(LumaAlignmentDataset(val_rows, args.predictor_source), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SpatialHeadSimilaritySTN(args.max_angle_deg, args.max_translate, args.max_log_scale).to(device)
    if args.init_checkpoint is not None:
        init = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(init["model_state"])
        print(f"initialized from {args.init_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_score = -1e9
    best_path = args.out_dir / "best_luma_spatial_head_stn.pt"
    history = []
    print(f"device={device} train={len(train_rows)} val={len(val_rows)} params={parameter_count(model)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            predictor = batch["predictor"].to(device)
            source_luma = batch["source_luma"].to(device)
            target_luma = batch["target_luma"].to(device)
            theta, params = model(predictor)
            warped_luma = warp_with_theta(source_luma, theta)
            loss, _ = luma_alignment_loss(warped_luma, target_luma, params)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **metrics}
        history.append(row)
        score = metrics["after"] + 0.10 * metrics["after_p05"] + 0.05 * metrics["after_p01"]
        print(
            f"epoch {epoch:03d} loss={row['loss']:.4f} val {metrics['before']:.4f}->{metrics['after']:.4f} "
            f"p05={metrics['after_p05']:.4f} gain={metrics['improvement']:.4f}"
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model": "LumaSpatialHeadSimilaritySTN",
                    "params": parameter_count(model),
                    "epoch": epoch,
                    "val_metrics": metrics,
                    "max_angle_deg": args.max_angle_deg,
                    "max_translate": args.max_translate,
                    "max_log_scale": args.max_log_scale,
                    "predictor_source": args.predictor_source,
                    "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else "",
                    "luma": "Rec.709 Y = 0.2126R + 0.7152G + 0.0722B",
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    aligned_root = args.out_dir / "aligned_rgb_256"
    summary = write_outputs(model, rows, aligned_root, device, args.batch_size, args.predictor_source)
    fields = ["aligned_path", "crop_path", "target_path", "condition", "frame", "before_corr", "after_corr", "angle_deg", "scale", "tx_px", "ty_px"]
    write_csv(args.out_dir / "luma_spatial_head_alignment_summary.csv", summary, fields)
    write_csv(
        args.out_dir / "training_history.csv",
        history,
        ["epoch", "loss", "before", "after", "improvement", "after_p01", "after_p05", "angle_abs_mean", "angle_mean", "angle_std", "scale_mean", "scale_std", "tx_px_mean", "ty_px_mean"],
    )
    make_preview(summary, args.out_dir / "luma_spatial_head_alignment_preview.jpg")

    before = np.asarray([float(row["before_corr"]) for row in summary], dtype=np.float32)
    after = np.asarray([float(row["after_corr"]) for row in summary], dtype=np.float32)
    report = args.out_dir / "luma_spatial_head_stn_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Luma-Aware Spatial-Head STN Report\n\n")
        f.write(f"- Pair CSV: `{args.pair_csv}`\n")
        f.write(f"- Parameters: {parameter_count(model)}\n")
        f.write(f"- Predictor source: {args.predictor_source}\n")
        f.write(f"- Init checkpoint: `{args.init_checkpoint}`\n")
        f.write(f"- Best checkpoint: `{best_path}`\n")
        f.write(f"- Aligned RGB root: `{aligned_root}`\n")
        f.write(f"- Before luma-correlation mean/std: {before.mean():.4f} / {before.std():.4f}\n")
        f.write(f"- After luma-correlation mean/std: {after.mean():.4f} / {after.std():.4f}\n")
        f.write(f"- After luma-correlation p01/p05: {np.percentile(after, 1):.4f} / {np.percentile(after, 5):.4f}\n")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
