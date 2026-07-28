"""Train a lightweight similarity-STN aligner for 256x256 micro-LED crops.

This module is intentionally constrained for compatibility with the existing
512-bit PUF extraction pipeline:

- input: YOLO-cropped RGB images, 256x256
- output: aligned RGB images, 256x256
- folder layout: condition folders are preserved
- alignment loss: blue-channel based, matching the downstream extractor
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_CSV = REPO_ROOT / "data" / "02_stn_pairs_M1_M6" / "alignment_pairs_M1_M6.csv"
OUT_DIR = REPO_ROOT / "work" / "similarity_stn"
IMAGE_SIZE = 256
PREDICTOR_SIZE = 80


@dataclass(frozen=True)
class PairRow:
    crop_path: Path
    target_path: Path
    condition: str
    frame: str


def get_font(size: int) -> ImageFont.ImageFont:
    for candidate in [r"C:\Windows\Fonts\arial.ttf", r"C:\Windows\Fonts\segoeui.ttf"]:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int = 14, fill=(30, 30, 30)) -> None:
    draw.text(xy, text, font=get_font(size), fill=fill)


def load_pairs(pair_csv: Path) -> list[PairRow]:
    rows = []
    with pair_csv.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("error") or (not row.get("crop_path") and not row.get("crop_relative")):
                continue
            if row.get("crop_relative"):
                crop_path = pair_csv.parent / row["crop_relative"]
                target_path = pair_csv.parent / row["target_relative"]
            else:
                crop_path = Path(row["crop_path"])
                target_path = Path(row["target_path"])
            if crop_path.exists() and target_path.exists():
                rows.append(PairRow(crop_path, target_path, row["condition"], row["frame"]))
    return rows


def split_pairs(rows: Sequence[PairRow], val_ratio: float, seed: int) -> tuple[list[PairRow], list[PairRow]]:
    grouped: dict[str, list[PairRow]] = defaultdict(list)
    for row in rows:
        grouped[row.condition].append(row)
    rng = random.Random(seed)
    train, val = [], []
    for _, items in sorted(grouped.items()):
        items = list(items)
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def pil_rgb(path: Path, size: int = IMAGE_SIZE) -> Image.Image:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BILINEAR)
    return img


def rgb_tensor(path: Path, size: int = IMAGE_SIZE) -> torch.Tensor:
    arr = np.asarray(pil_rgb(path, size), dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def blue_tensor(path: Path, size: int = IMAGE_SIZE) -> torch.Tensor:
    arr = np.asarray(pil_rgb(path, size), dtype=np.float32) / 255.0
    return torch.from_numpy(arr[:, :, 2])[None, :, :]


def predictor_tensor(path: Path) -> torch.Tensor:
    img = pil_rgb(path, IMAGE_SIZE).resize((PREDICTOR_SIZE, PREDICTOR_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    blue = arr[:, :, 2]
    blue = (blue - blue.mean()) / (blue.std() + 1e-6)
    return torch.from_numpy(blue.astype(np.float32))[None, :, :]


class AlignmentDataset(Dataset):
    def __init__(self, rows: Sequence[PairRow]) -> None:
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, object]:
        row = self.rows[idx]
        return {
            "predictor": predictor_tensor(row.crop_path),
            "source_blue": blue_tensor(row.crop_path),
            "target_blue": blue_tensor(row.target_path),
            "source_rgb": rgb_tensor(row.crop_path),
            "condition": row.condition,
            "frame": row.frame,
            "crop_path": str(row.crop_path),
            "target_path": str(row.target_path),
        }


class SimilaritySTN(nn.Module):
    def __init__(self, max_angle_deg: float = 90.0, max_translate: float = 0.22, max_log_scale: float = 0.25) -> None:
        super().__init__()
        self.max_angle = math.radians(max_angle_deg)
        self.max_translate = max_translate
        self.max_log_scale = max_log_scale
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GroupNorm(12, 96),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.SiLU(inplace=True),
            nn.Linear(64, 4),
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
        cos = torch.cos(angle) * scale
        sin = torch.sin(angle) * scale
        theta = torch.zeros((x.shape[0], 2, 3), dtype=x.dtype, device=x.device)
        theta[:, 0, 0] = cos
        theta[:, 0, 1] = -sin
        theta[:, 1, 0] = sin
        theta[:, 1, 1] = cos
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        params = torch.stack([angle, scale, tx, ty], dim=1)
        return theta, params


def warp_with_theta(x: torch.Tensor, theta: torch.Tensor, padding_mode: str = "zeros") -> torch.Tensor:
    grid = F.affine_grid(theta, x.size(), align_corners=True)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)


def zncc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af = a.flatten(1)
    bf = b.flatten(1)
    af = af - af.mean(dim=1, keepdim=True)
    bf = bf - bf.mean(dim=1, keepdim=True)
    return (af * bf).mean(dim=1) / (af.std(dim=1) * bf.std(dim=1) + 1e-6)


def radial_residual_patch_tensor(x: torch.Tensor, patch_grid: int = 32) -> torch.Tensor:
    b, c, h, w = x.shape
    if c != 1 or h != IMAGE_SIZE or w != IMAGE_SIZE:
        raise ValueError(f"expected Bx1x{IMAGE_SIZE}x{IMAGE_SIZE}, got {tuple(x.shape)}")
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=torch.float32),
        torch.arange(w, device=x.device, dtype=torch.float32),
        indexing="ij",
    )
    center = (IMAGE_SIZE - 1) / 2.0
    bins = torch.floor(torch.sqrt((yy - center).square() + (xx - center).square())).long().flatten()
    n_bins = int(bins.max().item()) + 1
    flat = x[:, 0].flatten(1)
    sums = torch.zeros((b, n_bins), dtype=x.dtype, device=x.device)
    sums.scatter_add_(1, bins.expand(b, -1), flat)
    counts = torch.bincount(bins, minlength=n_bins).to(device=x.device, dtype=x.dtype).clamp_min(1.0)
    means = sums / counts[None, :]
    residual = flat - means.gather(1, bins.expand(b, -1))
    block = IMAGE_SIZE // patch_grid
    patch = residual.reshape(b, patch_grid, block, patch_grid, block).mean(dim=(2, 4))
    return patch.flatten(1)


def alignment_loss(
    warped: torch.Tensor,
    target: torch.Tensor,
    params: torch.Tensor,
    residual_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    corr = zncc(warped, target)
    ncc_loss = 1.0 - corr.mean()
    residual_corr = zncc(radial_residual_patch_tensor(warped), radial_residual_patch_tensor(target))
    residual_loss = 1.0 - residual_corr.mean()
    l1 = F.smooth_l1_loss(warped, target)
    angle = params[:, 0]
    scale = params[:, 1]
    tx = params[:, 2]
    ty = params[:, 3]
    reg = 0.002 * ((angle / math.pi) ** 2).mean() + 0.002 * ((torch.log(scale) / 0.25) ** 2).mean()
    reg = reg + 0.001 * (tx.square().mean() + ty.square().mean())
    loss = ncc_loss + residual_weight * residual_loss + 0.15 * l1 + reg
    return loss, {
        "corr": float(corr.mean().detach().cpu()),
        "residual_corr": float(residual_corr.mean().detach().cpu()),
        "l1": float(l1.detach().cpu()),
        "reg": float(reg.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model: SimilaritySTN, loader: DataLoader, device: torch.device) -> dict[str, float]:
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
        scales.extend(params[:, 1].detach().cpu().tolist())
        txs.extend(params[:, 2].detach().cpu().tolist())
        tys.extend(params[:, 3].detach().cpu().tolist())
    return {
        "before": float(np.mean(before)),
        "after": float(np.mean(after)),
        "improvement": float(np.mean(after) - np.mean(before)),
        "angle_mean": float(np.mean(angles)),
        "angle_std": float(np.std(angles)),
        "angle_abs_mean": float(np.mean(np.abs(angles))),
        "scale_mean": float(np.mean(scales)),
        "tx_px_mean": float(np.mean(txs) * IMAGE_SIZE / 2.0),
        "ty_px_mean": float(np.mean(tys) * IMAGE_SIZE / 2.0),
    }


def tensor_to_rgb(x: torch.Tensor) -> Image.Image:
    arr = x.detach().cpu().numpy()
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0, 1)
    return Image.fromarray(np.uint8(arr * 255), "RGB")


@torch.no_grad()
def write_aligned_outputs(
    model: SimilaritySTN,
    rows: Sequence[PairRow],
    out_root: Path,
    crop_root: Path,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, object]]:
    out_root.mkdir(parents=True, exist_ok=True)
    ds = AlignmentDataset(rows)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    summary = []
    for batch in loader:
        predictor = batch["predictor"].to(device)
        source_rgb = batch["source_rgb"].to(device)
        source_blue = batch["source_blue"].to(device)
        target_blue = batch["target_blue"].to(device)
        theta, params = model(predictor)
        warped_rgb = warp_with_theta(source_rgb, theta)
        warped_blue = warp_with_theta(source_blue, theta)
        before = zncc(source_blue, target_blue).detach().cpu().tolist()
        after = zncc(warped_blue, target_blue).detach().cpu().tolist()
        angles = torch.rad2deg(params[:, 0]).detach().cpu().tolist()
        scales = params[:, 1].detach().cpu().tolist()
        txs = params[:, 2].detach().cpu().tolist()
        tys = params[:, 3].detach().cpu().tolist()
        for i in range(warped_rgb.shape[0]):
            crop_path = Path(batch["crop_path"][i])
            rel = crop_path.relative_to(crop_root)
            out_path = out_root / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tensor_to_rgb(warped_rgb[i]).save(out_path)
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
    draw_text(draw, (28, 20), "Similarity-STN alignment: YOLO crop / aligned 256 / target", 25)
    for i, row in enumerate(rows):
        crop = Image.open(row["crop_path"]).convert("RGB").resize((150, 150), Image.Resampling.NEAREST)
        aligned = Image.open(row["aligned_path"]).convert("RGB").resize((150, 150), Image.Resampling.NEAREST)
        target = Image.open(row["target_path"]).convert("RGB").resize((150, 150), Image.Resampling.NEAREST)
        x = (i % cols) * tile_w + 18
        y = (i // cols) * tile_h + 78
        for j, (label, img) in enumerate((("YOLO crop", crop), ("aligned 256", aligned), ("target", target))):
            canvas.paste(img, (x + j * 180, y))
            draw_text(draw, (x + j * 180, y + 156), label, 12)
        draw_text(
            draw,
            (x, y + 180),
            f"corr {float(row['before_corr']):.3f}->{float(row['after_corr']):.3f}, angle={float(row['angle_deg']):.1f}, scale={float(row['scale']):.3f}",
            11,
            (70, 70, 70),
        )
        draw_text(
            draw,
            (x, y + 200),
            f"shift=({float(row['tx_px']):.1f},{float(row['ty_px']):.1f}) px  {row['condition']}/{row['frame']}",
            10,
            (80, 80, 80),
        )
    canvas.save(out_path, quality=95)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-csv", type=Path, default=PAIR_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument("--residual-weight", type=float, default=0.0)
    parser.add_argument("--max-angle-deg", type=float, default=90.0)
    parser.add_argument("--max-translate", type=float, default=0.22)
    parser.add_argument("--max-log-scale", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    rows = load_pairs(args.pair_csv)
    if not rows:
        raise RuntimeError(f"No valid pairs found in {args.pair_csv}")
    train_rows, val_rows = split_pairs(rows, args.val_ratio, args.seed)
    crop_root = args.pair_csv.parent / "yolo11n_crops_256"

    train_loader = DataLoader(AlignmentDataset(train_rows), batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(AlignmentDataset(val_rows), batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SimilaritySTN(
        max_angle_deg=args.max_angle_deg,
        max_translate=args.max_translate,
        max_log_scale=args.max_log_scale,
    ).to(device)
    if args.init_checkpoint is not None:
        checkpoint = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_metric = -1e9
    best_path = args.out_dir / "best_similarity_stn_256.pt"
    history = []
    print(f"device={device} train={len(train_rows)} val={len(val_rows)} params={parameter_count(model)}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []
        for batch in train_loader:
            predictor = batch["predictor"].to(device)
            source = batch["source_blue"].to(device)
            target = batch["target_blue"].to(device)
            theta, params = model(predictor)
            warped = warp_with_theta(source, theta)
            loss, parts = alignment_loss(warped, target, params, args.residual_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running.append(float(loss.detach().cpu()))
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        train_loss = float(np.mean(running))
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={train_loss:.4f} val {metrics['before']:.4f}->{metrics['after']:.4f} "
            f"gain={metrics['improvement']:.4f} angle_abs={metrics['angle_abs_mean']:.2f}"
        )
        if metrics["after"] > best_metric:
            best_metric = metrics["after"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model": "SimilaritySTN",
                    "image_size": IMAGE_SIZE,
                    "predictor_size": PREDICTOR_SIZE,
                    "params": parameter_count(model),
                    "epoch": epoch,
                    "val_metrics": metrics,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    aligned_root = args.out_dir / "aligned_256"
    summary = write_aligned_outputs(model, rows, aligned_root, crop_root, device, args.batch_size)
    summary_path = args.out_dir / "alignment_apply_summary.csv"
    write_csv(
        summary_path,
        summary,
        [
            "aligned_path",
            "crop_path",
            "target_path",
            "condition",
            "frame",
            "before_corr",
            "after_corr",
            "angle_deg",
            "scale",
            "tx_px",
            "ty_px",
        ],
    )
    preview_rows = sorted(summary, key=lambda r: float(r["after_corr"]) - float(r["before_corr"]), reverse=True)
    make_preview(preview_rows, args.out_dir / "similarity_stn_best_improvements_preview.jpg")
    make_preview(summary, args.out_dir / "similarity_stn_preview.jpg")

    hist_path = args.out_dir / "training_history.csv"
    write_csv(
        hist_path,
        history,
        ["epoch", "train_loss", "before", "after", "improvement", "angle_mean", "angle_std", "angle_abs_mean", "scale_mean", "tx_px_mean", "ty_px_mean"],
    )

    before_vals = np.asarray([float(row["before_corr"]) for row in summary], dtype=np.float32)
    after_vals = np.asarray([float(row["after_corr"]) for row in summary], dtype=np.float32)
    angle_vals = np.asarray([float(row["angle_deg"]) for row in summary], dtype=np.float32)
    scale_vals = np.asarray([float(row["scale"]) for row in summary], dtype=np.float32)
    report = args.out_dir / "similarity_stn_256_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Similarity-STN 256x256 Alignment Report\n\n")
        f.write("## Pipeline Compatibility\n\n")
        f.write("- Input: YOLO11n-cropped RGB images, 256x256\n")
        f.write("- Output: aligned RGB images, 256x256\n")
        f.write("- Folder layout: condition folders and frame names preserved\n")
        f.write("- Downstream bit extraction: compatible with blue-channel radial-residual 512-bit pipeline\n\n")
        f.write("## Training\n\n")
        f.write(f"- Pair CSV: `{args.pair_csv}`\n")
        f.write(f"- Train/val pairs: {len(train_rows)} / {len(val_rows)}\n")
        f.write(f"- Epochs: {args.epochs}\n")
        f.write(f"- Batch size: {args.batch_size}\n")
        f.write(f"- Residual patch loss weight: {args.residual_weight}\n")
        f.write(f"- Max angle: +/-{args.max_angle_deg} deg\n")
        f.write(f"- Max normalized translation: +/-{args.max_translate}\n")
        f.write(f"- Max log scale: +/-{args.max_log_scale}\n")
        if args.init_checkpoint is not None:
            f.write(f"- Initialized from: `{args.init_checkpoint}`\n")
        f.write(f"- Model parameters: {parameter_count(model)}\n")
        f.write(f"- Best checkpoint: `{best_path}`\n\n")
        f.write("## Alignment Quality On All Prepared Pairs\n\n")
        f.write(f"- Before correlation mean/std: {before_vals.mean():.4f} / {before_vals.std():.4f}\n")
        f.write(f"- After correlation mean/std: {after_vals.mean():.4f} / {after_vals.std():.4f}\n")
        f.write(f"- Mean improvement: {(after_vals - before_vals).mean():.4f}\n")
        f.write(f"- Improved frames (>0.02): {int(np.sum(after_vals - before_vals > 0.02))}\n")
        f.write(f"- Worse frames (<-0.02): {int(np.sum(after_vals - before_vals < -0.02))}\n")
        f.write(f"- Angle mean/std/min/max: {angle_vals.mean():.3f} / {angle_vals.std():.3f} / {angle_vals.min():.3f} / {angle_vals.max():.3f} deg\n")
        f.write(f"- Scale mean/std/min/max: {scale_vals.mean():.4f} / {scale_vals.std():.4f} / {scale_vals.min():.4f} / {scale_vals.max():.4f}\n\n")
        f.write("## Files\n\n")
        f.write(f"- Aligned root for bit extraction: `{aligned_root}`\n")
        f.write(f"- Apply summary: `{summary_path}`\n")
        f.write(f"- Training history: `{hist_path}`\n")
        f.write("- `similarity_stn_preview.jpg`: regular side-by-side preview\n")
        f.write("- `similarity_stn_best_improvements_preview.jpg`: frames with largest correlation gain\n")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
