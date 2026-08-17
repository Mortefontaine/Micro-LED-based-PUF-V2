"""Candidate-bank micro-LED PUF extraction from aligned RGB images.

This release script is intentionally inference-oriented. It expects images that
are already cropped/aligned to 256x256 RGB, then applies the final lightweight
handcrafted extractor:

RGB -> Rec.709 luma -> radial residual -> common-template removal ->
fixed sparse-projection candidates for per-device 2048-bit enrollment.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import cv2
from PIL import Image


IMAGE_SIZE = 256
PATCH_GRID = 32
PATCH_COUNT = PATCH_GRID * PATCH_GRID
POSE_MIN_COMMON_CORR_GAIN = 0.05
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


def condition_name(path: Path, root: Path | None = None) -> str:
    if root is not None and root.is_dir():
        try:
            rel = path.relative_to(root)
            if len(rel.parts) > 1:
                return rel.parts[0]
        except ValueError:
            pass
    return path.parent.name


def device_name(condition: str) -> str:
    return condition.split("_", 1)[0].upper()


def read_rgb01(path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def rec709_luma(rgb: np.ndarray) -> np.ndarray:
    return (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]).astype(np.float32)


def make_radial_bins(image_size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.indices((image_size, image_size), dtype=np.float32)
    center = (image_size - 1) / 2.0
    bins = np.floor(np.sqrt((xx - center) ** 2 + (yy - center) ** 2)).astype(np.int32).ravel()
    counts = np.bincount(bins)
    return bins, counts


def radial_residual_patch(image: np.ndarray, radial_bins: np.ndarray, radial_counts: np.ndarray) -> np.ndarray:
    flat = image.ravel()
    sums = np.bincount(radial_bins, weights=flat, minlength=radial_counts.size)
    radial_mean = sums / np.maximum(radial_counts, 1)
    residual = flat - radial_mean[radial_bins]
    block = IMAGE_SIZE // PATCH_GRID
    patch = residual.reshape(PATCH_GRID, block, PATCH_GRID, block).mean(axis=(1, 3))
    return patch.astype(np.float32).ravel()


def normalized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float32).ravel()
    y = np.asarray(b, dtype=np.float32).ravel()
    x = x - float(x.mean())
    y = y - float(y.mean())
    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))


def pose_feature_from_rgb(rgb: np.ndarray) -> np.ndarray:
    luma = rec709_luma(rgb)
    block = IMAGE_SIZE // PATCH_GRID
    return luma.reshape(PATCH_GRID, block, PATCH_GRID, block).mean(axis=(1, 3)).astype(np.float32).ravel()


def search_pose_correction(observed: np.ndarray, template: np.ndarray) -> dict[str, float]:
    image = np.asarray(observed, dtype=np.float32).reshape(PATCH_GRID, PATCH_GRID)
    best = {"score": normalized_correlation(image, template), "angle_deg": 0.0, "scale": 1.0, "tx32": 0.0, "ty32": 0.0}
    for angle in range(-30, 31, 3):
        for scale in np.arange(0.95, 1.051, 0.025):
            base = cv2.getRotationMatrix2D(((PATCH_GRID - 1) / 2, (PATCH_GRID - 1) / 2), float(angle), float(scale))
            for tx in range(-1, 2):
                for ty in range(-1, 2):
                    matrix = base.copy()
                    matrix[0, 2] += tx
                    matrix[1, 2] += ty
                    warped = cv2.warpAffine(
                        image, matrix, (PATCH_GRID, PATCH_GRID), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
                    )
                    score = normalized_correlation(warped, template)
                    if score > best["score"]:
                        best = {"score": score, "angle_deg": float(angle), "scale": float(scale), "tx32": float(tx), "ty32": float(ty)}
    return best


def apply_pose_correction(rgb: np.ndarray, pose: dict[str, float]) -> np.ndarray:
    matrix = cv2.getRotationMatrix2D(
        ((IMAGE_SIZE - 1) / 2, (IMAGE_SIZE - 1) / 2), pose["angle_deg"], pose["scale"]
    )
    block = IMAGE_SIZE / PATCH_GRID
    matrix[0, 2] += block * pose["tx32"]
    matrix[1, 2] += block * pose["ty32"]
    return cv2.warpAffine(
        np.asarray(rgb, dtype=np.float32), matrix, (IMAGE_SIZE, IMAGE_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )


@dataclass(frozen=True)
class PUFExtractor:
    selected: np.ndarray
    pos_idx: np.ndarray
    neg_idx: np.ndarray
    common_template: np.ndarray
    channels: tuple[str, ...]
    pose_template_luma32: np.ndarray | None
    pose_trigger_corr: float | None
    pose_refined_min_common_corr: float | None

    @classmethod
    def from_payload(cls, payload_path: Path) -> "PUFExtractor":
        data = np.load(payload_path, allow_pickle=True)
        channels = tuple(str(x) for x in data["channels"].tolist())
        if channels != ("Y",):
            raise ValueError(f"This release extractor expects luma-only payload, got {channels}")
        return cls(
            selected=np.asarray(data["selected"], dtype=np.int64),
            pos_idx=np.asarray(data["pos_idx"], dtype=np.int32),
            neg_idx=np.asarray(data["neg_idx"], dtype=np.int32),
            common_template=np.asarray(data["common_template"], dtype=np.float32),
            channels=channels,
            pose_template_luma32=(np.asarray(data["pose_template_luma32"], dtype=np.float32) if "pose_template_luma32" in data.files else None),
            pose_trigger_corr=(float(np.asarray(data["pose_trigger_corr"]).item()) if "pose_trigger_corr" in data.files else None),
            pose_refined_min_common_corr=(
                float(np.asarray(data["pose_refined_min_common_corr"]).item())
                if "pose_refined_min_common_corr" in data.files
                else None
            ),
        )

    def feature_from_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Extract the canonical first-pass feature without pose refinement."""
        prepared, _ = self.prepare_rgb(rgb, pose_mode="disabled")
        return self.feature_from_prepared_rgb(prepared)

    def feature_from_prepared_rgb(self, rgb: np.ndarray) -> np.ndarray:
        radial_bins, radial_counts = make_radial_bins(IMAGE_SIZE)
        y = rec709_luma(rgb)
        return radial_residual_patch(y, radial_bins, radial_counts)

    def prepare_rgb(
        self,
        rgb: np.ndarray,
        pose_mode: str = "disabled",
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Use no pose correction first; allow one explicit failure retry."""
        if pose_mode not in {"disabled", "forced", "triggered"}:
            raise ValueError(f"Unsupported pose mode: {pose_mode}")
        default = {
            "pose_initial_corr": None, "pose_final_corr": None, "pose_refined": False,
            "pose_attempted": False,
            "pose_gate_passed": True, "pose_angle_deg": 0.0, "pose_scale": 1.0,
            "pose_tx32": 0.0, "pose_ty32": 0.0,
        }
        if self.pose_template_luma32 is None:
            return rgb, default
        initial_feature = self.feature_from_prepared_rgb(rgb)
        initial = self.common_template_correlation(initial_feature)
        initial_state = {**default, "pose_initial_corr": initial, "pose_final_corr": initial}
        if pose_mode == "disabled":
            return rgb, initial_state
        if (
            pose_mode == "triggered"
            and self.pose_trigger_corr is not None
            and np.isfinite(initial)
            and initial >= self.pose_trigger_corr
        ):
            return rgb, initial_state
        observed = pose_feature_from_rgb(rgb)
        pose = search_pose_correction(observed, self.pose_template_luma32)
        corrected = apply_pose_correction(rgb, pose)
        final = self.common_template_correlation(self.feature_from_prepared_rgb(corrected))
        if not np.isfinite(final) or final - initial < POSE_MIN_COMMON_CORR_GAIN:
            return rgb, {
                **default,
                "pose_initial_corr": initial,
                "pose_final_corr": initial,
                "pose_attempted": True,
            }
        return corrected, {
            "pose_initial_corr": initial,
            "pose_final_corr": final,
            "pose_refined": True,
            "pose_attempted": True,
            "pose_gate_passed": bool(
                np.isfinite(final)
                and (
                    self.pose_refined_min_common_corr is None
                    or final >= self.pose_refined_min_common_corr
                )
            ),
            "pose_angle_deg": pose["angle_deg"],
            "pose_scale": pose["scale"],
            "pose_tx32": pose["tx32"],
            "pose_ty32": pose["ty32"],
        }

    def feature_and_pose_from_rgb(
        self,
        rgb: np.ndarray,
        pose_mode: str = "disabled",
    ) -> tuple[np.ndarray, dict[str, object]]:
        prepared, pose = self.prepare_rgb(rgb, pose_mode=pose_mode)
        return self.feature_from_prepared_rgb(prepared), pose

    def common_template_correlation(self, feature: np.ndarray) -> float:
        """Score global structural alignment without producing response bits."""
        observed = np.asarray(feature, dtype=np.float32) - float(np.mean(feature))
        template = self.common_template - float(np.mean(self.common_template))
        return float(np.dot(observed, template) / (np.linalg.norm(observed) * np.linalg.norm(template) + 1e-9))

    def bits_from_feature(self, feature: np.ndarray) -> np.ndarray:
        return (self.margins_from_feature(feature) > 0).astype(np.uint8)

    def margins_from_feature(self, feature: np.ndarray) -> np.ndarray:
        """Return signed margins for the payload's legacy selected projections.

        The sign is the PUF response bit; the magnitude is available to the
        key-regeneration layer as a local confidence signal.
        """
        return self.candidate_margins_from_feature(feature)[self.selected]

    def candidate_margins_from_feature(self, feature: np.ndarray) -> np.ndarray:
        """Return signed margins for all fixed sparse-projection candidates."""
        adjusted = feature.astype(np.float32) - self.common_template
        return (adjusted[self.pos_idx].sum(axis=1) - adjusted[self.neg_idx].sum(axis=1)).astype(np.float32)

    def selected_candidate_margins_from_feature(
        self, feature: np.ndarray, candidate_indices: Sequence[int]
    ) -> np.ndarray:
        """Evaluate only enrolled candidates for lower edge-runtime cost."""
        selected = np.asarray(candidate_indices, dtype=np.int64)
        adjusted = feature.astype(np.float32) - self.common_template
        return (
            adjusted[self.pos_idx[selected]].sum(axis=1)
            - adjusted[self.neg_idx[selected]].sum(axis=1)
        ).astype(np.float32)

    def bits_from_image(self, image_path: Path) -> np.ndarray:
        return self.bits_from_feature(self.feature_from_rgb(read_rgb01(image_path)))

    def response_from_image(self, image_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Return the payload-selected response and its signed projection margins."""
        feature = self.feature_from_rgb(read_rgb01(image_path))
        margins = self.margins_from_feature(feature)
        return (margins > 0).astype(np.uint8), margins


def bits_to_string(bits: np.ndarray) -> str:
    return "".join("1" if int(x) else "0" for x in bits)


def majority_vote(bit_rows: Sequence[np.ndarray]) -> np.ndarray:
    stack = np.stack(bit_rows, axis=0).astype(np.uint8)
    return (stack.sum(axis=0) >= ((stack.shape[0] + 1) // 2)).astype(np.uint8)


def hamming(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    return float(np.mean(a != b))


def extract_images(input_path: Path, payload_path: Path) -> list[dict[str, object]]:
    extractor = PUFExtractor.from_payload(payload_path)
    root = input_path if input_path.is_dir() else input_path.parent
    rows: list[dict[str, object]] = []
    for path in iter_images(input_path):
        bits = extractor.bits_from_image(path)
        condition = condition_name(path, root)
        rows.append(
            {
                "path": str(path),
                "condition": condition,
                "device": device_name(condition),
                "bits": bits,
                "bitstring": bits_to_string(bits),
            }
        )
    return rows


def write_bit_csv(rows: Sequence[dict[str, object]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "condition", "device", "bitstring"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in ["path", "condition", "device", "bitstring"]})


def write_majority_csv(rows: Sequence[dict[str, object]], out_csv: Path, vote_sizes: Sequence[int]) -> None:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "device", "vote_size", "n_used", "bitstring"])
        writer.writeheader()
        for condition, items in sorted(grouped.items(), key=lambda kv: natural_key(kv[0])):
            items = sorted(items, key=lambda r: natural_key(str(r["path"])))
            for vote_size in vote_sizes:
                chosen = items[: min(vote_size, len(items))]
                bits = majority_vote([row["bits"] for row in chosen])
                writer.writerow(
                    {
                        "condition": condition,
                        "device": device_name(condition),
                        "vote_size": vote_size,
                        "n_used": len(chosen),
                        "bitstring": bits_to_string(bits),
                    }
                )


def default_payload_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "expanded_luma_support_payload.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Aligned RGB image file or folder.")
    parser.add_argument("--payload", type=Path, default=default_payload_path())
    parser.add_argument("--out-csv", type=Path, default=Path("release_bits.csv"))
    parser.add_argument("--majority-csv", type=Path, default=None)
    parser.add_argument("--vote-sizes", type=int, nargs="*", default=[1, 3, 9])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = extract_images(args.input, args.payload)
    write_bit_csv(rows, args.out_csv)
    if args.majority_csv:
        write_majority_csv(rows, args.majority_csv, args.vote_sizes)
    bit_count = len(rows[0]["bits"]) if rows else 0
    print(f"images={len(rows)} bits={bit_count} out={args.out_csv}")


if __name__ == "__main__":
    main()
