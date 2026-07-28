"""Train a spatial-head STN that preserves pose information before regression."""

from __future__ import annotations

import argparse
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from microled_train_cross_aware_rigid_stn import (
    CROP_ROOT,
    CrossAwareDataset,
    alignment_loss,
    evaluate,
    make_preview,
    parameter_count,
    write_outputs,
)
from microled_train_similarity_stn_256 import PAIR_CSV, load_pairs, split_pairs, warp_with_theta


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "work" / "spatial_head_similarity_stn"


class SpatialHeadSimilaritySTN(nn.Module):
    def __init__(self, max_angle_deg: float = 180.0, max_translate: float = 0.20, max_log_scale: float = 0.35) -> None:
        super().__init__()
        self.max_angle = math.radians(max_angle_deg)
        self.max_translate = max_translate
        self.max_log_scale = max_log_scale
        self.features = nn.Sequential(
            nn.Conv2d(4, 24, 5, stride=2, padding=2),  # 96 -> 48
            nn.GroupNorm(6, 24),
            nn.SiLU(inplace=True),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),  # 48 -> 24
            nn.GroupNorm(8, 48),
            nn.SiLU(inplace=True),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),  # 24 -> 12
            nn.GroupNorm(12, 96),
            nn.SiLU(inplace=True),
            nn.Conv2d(96, 128, 3, stride=2, padding=1),  # 12 -> 6
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


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-csv", type=Path, default=PAIR_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-angle-deg", type=float, default=180.0)
    parser.add_argument("--max-translate", type=float, default=0.20)
    parser.add_argument("--max-log-scale", type=float, default=0.35)
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

    model = SpatialHeadSimilaritySTN(args.max_angle_deg, args.max_translate, args.max_log_scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    best_after = -1e9
    best_path = args.out_dir / "best_spatial_head_similarity_stn.pt"
    history: list[dict[str, object]] = []
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
            loss, _ = alignment_loss(warped, target, params)
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
            f"gain={metrics['improvement']:.4f} angle_abs={metrics['angle_abs_mean']:.2f} scale={metrics['scale_mean']:.3f}"
        )
        if metrics["after"] > best_after:
            best_after = metrics["after"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model": "SpatialHeadSimilaritySTN",
                    "params": parameter_count(model),
                    "epoch": epoch,
                    "val_metrics": metrics,
                    "max_angle_deg": args.max_angle_deg,
                    "max_translate": args.max_translate,
                    "max_log_scale": args.max_log_scale,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    aligned_root = args.out_dir / "aligned_blue_256"
    summary = write_outputs(model, rows, aligned_root, device, args.batch_size)
    write_csv(
        args.out_dir / "spatial_head_alignment_summary.csv",
        summary,
        ["aligned_path", "crop_path", "target_path", "condition", "frame", "before_corr", "after_corr", "angle_deg", "scale", "tx_px", "ty_px"],
    )
    write_csv(
        args.out_dir / "training_history.csv",
        history,
        ["epoch", "loss", "before", "after", "improvement", "angle_abs_mean", "angle_mean", "angle_std", "scale_mean", "scale_std", "tx_px_mean", "ty_px_mean"],
    )
    make_preview(summary, args.out_dir / "spatial_head_alignment_preview.jpg")
    before = np.asarray([float(row["before_corr"]) for row in summary], dtype=np.float32)
    after = np.asarray([float(row["after_corr"]) for row in summary], dtype=np.float32)
    angles = np.asarray([float(row["angle_deg"]) for row in summary], dtype=np.float32)
    scales = np.asarray([float(row["scale"]) for row in summary], dtype=np.float32)
    report = args.out_dir / "spatial_head_similarity_stn_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Spatial-head Similarity STN Report\n\n")
        f.write("## Method\n\n")
        f.write("- Input: blue-derived 4-channel features.\n")
        f.write("- Head: flatten 6x6 spatial feature map, preserving pose information.\n")
        f.write("- Transform: rotation + translation + limited scale.\n")
        f.write("- Output: 256x256 blue-only image.\n\n")
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
        f.write(f"- Scale mean/std/min/max: {scales.mean():.4f} / {scales.std():.4f} / {scales.min():.4f} / {scales.max():.4f}\n")
        f.write(f"- Aligned root: `{aligned_root}`\n")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
