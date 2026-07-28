"""Shared response-extraction and serialization helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from microled_puf import PUFExtractor, condition_name, iter_images, read_rgb01


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def pack_bits(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").tobytes()


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 HKDF using the Python standard library."""
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    output = bytearray()
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.new(
            prk,
            previous + info + bytes([counter]),
            hashlib.sha256,
        ).digest()
        output.extend(previous)
        counter += 1
    return bytes(output[:length])


def payload_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_rows(input_path: Path, payload_path: Path) -> list[dict[str, Any]]:
    """Extract bit responses and margins from aligned 256 x 256 RGB images."""
    extractor = PUFExtractor.from_payload(payload_path)
    root = input_path if input_path.is_dir() else input_path.parent
    rows: list[dict[str, Any]] = []
    for path in iter_images(input_path):
        rgb = read_rgb01(path)
        feature, pose = extractor.feature_and_pose_from_rgb(rgb)
        candidate_margins = extractor.candidate_margins_from_feature(feature)
        rows.append(
            {
                "path": str(path),
                "condition": condition_name(path, root),
                "candidate_bits": (candidate_margins > 0).astype(np.uint8),
                "candidate_margins": candidate_margins,
                "quality_template_corr": extractor.common_template_correlation(
                    feature
                ),
                **pose,
                "bits": (
                    candidate_margins[extractor.selected] > 0
                ).astype(np.uint8),
            }
        )
    return rows


def select_stable_candidates(
    rows: Sequence[dict[str, Any]],
    response_bits: int,
    candidate_pool: Sequence[int] | None = None,
    require_unanimous: bool = True,
) -> np.ndarray:
    """Select candidates using within-device stability and projection margin."""
    candidate_bits = np.stack(
        [
            np.asarray(row["candidate_bits"], dtype=np.uint8)
            for row in rows
        ],
        axis=0,
    )
    candidate_margins = np.stack(
        [
            np.asarray(row["candidate_margins"], dtype=np.float32)
            for row in rows
        ],
        axis=0,
    )
    reference = (candidate_bits.mean(axis=0) >= 0.5).astype(np.uint8)
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["condition"]), []).append(index)
    condition_agreement = np.stack(
        [
            (candidate_bits[indexes] == reference).mean(axis=0)
            for indexes in grouped.values()
        ],
        axis=0,
    )
    min_agreement = condition_agreement.min(axis=0)
    mean_agreement = (candidate_bits == reference).mean(axis=0)
    mean_margin = np.abs(candidate_margins).mean(axis=0)
    allowed = (
        np.arange(candidate_bits.shape[1], dtype=np.int64)
        if candidate_pool is None
        else np.asarray(candidate_pool, dtype=np.int64)
    )
    if (
        allowed.ndim != 1
        or np.any(allowed < 0)
        or np.any(allowed >= candidate_bits.shape[1])
    ):
        raise ValueError("Candidate pool contains out-of-range indices.")
    if np.unique(allowed).size != allowed.size:
        raise ValueError("Candidate pool contains duplicate indices.")
    if allowed.size < response_bits:
        raise ValueError(
            "The candidate pool is smaller than the required response size."
        )
    if not np.all(np.isfinite(candidate_margins[:, allowed])):
        raise ValueError(
            "Enrollment candidate margins contain non-finite values."
        )
    unanimous_count = int(
        np.count_nonzero(min_agreement[allowed] == 1.0)
    )
    if require_unanimous and unanimous_count < response_bits:
        raise ValueError(
            f"Only {unanimous_count} candidates are unanimous across all "
            f"enrollment conditions; {response_bits} are required."
        )
    if allowed.size == response_bits:
        return allowed.copy()
    order = allowed[
        np.lexsort(
            (
                -mean_margin[allowed],
                -mean_agreement[allowed],
                -min_agreement[allowed],
            )
        )
    ]
    selected = order[:response_bits].astype(np.int64)
    if len(selected) != response_bits:
        raise ValueError(
            "Not enough candidate projections for the requested response size."
        )
    return selected


def apply_candidate_selection(
    rows: Sequence[dict[str, Any]],
    indices: Sequence[int],
) -> list[dict[str, Any]]:
    selected = np.asarray(indices, dtype=np.int64)
    return [
        {
            **row,
            "bits": np.asarray(
                row["candidate_bits"],
                dtype=np.uint8,
            )[selected],
        }
        for row in rows
    ]


def load_candidate_pool(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    indices = np.asarray(data["eligible_indices"], dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError(
            "Candidate profile does not contain eligible indices."
        )
    if np.any(indices < 0) or np.unique(indices).size != indices.size:
        raise ValueError(
            "Candidate profile contains negative or duplicate indices."
        )
    return indices
