"""STN architecture and input-feature functions used by training and inference."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


IMAGE_SIZE = 256
PREDICTOR_SIZE = 96


def radial_residual_np(image: np.ndarray) -> np.ndarray:
    height, width = image.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    center = (height - 1) / 2.0
    bins = np.floor(
        np.sqrt((xx - center) ** 2 + (yy - center) ** 2)
    ).astype(np.int32).ravel()
    flat = image.ravel()
    sums = np.bincount(bins, weights=flat)
    counts = np.bincount(bins)
    means = sums / np.maximum(counts, 1)
    return (flat - means[bins]).reshape(height, width).astype(np.float32)


def sobel_mag_np(image: np.ndarray) -> np.ndarray:
    padded = np.pad(image, 1, mode="edge")
    gx = (
        -padded[:-2, :-2]
        - 2 * padded[1:-1, :-2]
        - padded[2:, :-2]
        + padded[:-2, 2:]
        + 2 * padded[1:-1, 2:]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def zscore(image: np.ndarray) -> np.ndarray:
    return (
        (image - image.mean()) / (image.std() + 1e-6)
    ).astype(np.float32)


def predictor_features(image: Image.Image | Path) -> torch.Tensor:
    source = (
        Image.open(image).convert("RGB")
        if isinstance(image, Path)
        else image.convert("RGB")
    )
    resized = source.resize(
        (PREDICTOR_SIZE, PREDICTOR_SIZE),
        Image.Resampling.BILINEAR,
    )
    rgb = np.asarray(resized, dtype=np.float32) / 255.0
    blue = rgb[:, :, 2]
    residual = radial_residual_np(blue)
    dark = np.clip(-residual, 0, None)
    edge = sobel_mag_np(residual)
    channels = np.stack(
        [
            zscore(blue),
            zscore(residual),
            zscore(dark),
            zscore(edge),
        ],
        axis=0,
    )
    return torch.from_numpy(channels.astype(np.float32))


class SpatialHeadSimilaritySTN(nn.Module):
    def __init__(
        self,
        max_angle_deg: float = 180.0,
        max_translate: float = 0.20,
        max_log_scale: float = 0.35,
    ) -> None:
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
        final = self.head[-1]
        assert isinstance(final, nn.Linear)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.head(self.features(inputs))
        angle = self.max_angle * torch.tanh(raw[:, 0])
        scale = torch.exp(self.max_log_scale * torch.tanh(raw[:, 1]))
        tx = self.max_translate * torch.tanh(raw[:, 2])
        ty = self.max_translate * torch.tanh(raw[:, 3])
        cosine = torch.cos(angle) * scale
        sine = torch.sin(angle) * scale
        theta = torch.zeros(
            (inputs.shape[0], 2, 3),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = -sine
        theta[:, 1, 0] = sine
        theta[:, 1, 1] = cosine
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        return theta, torch.stack([angle, scale, tx, ty], dim=1)


def load_stn(
    checkpoint_path: Path,
    device: torch.device,
) -> SpatialHeadSimilaritySTN:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model = SpatialHeadSimilaritySTN(
        checkpoint.get("max_angle_deg", 180.0),
        checkpoint.get("max_translate", 0.20),
        checkpoint.get("max_log_scale", 0.35),
    ).to(device)
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    model.eval()
    return model
