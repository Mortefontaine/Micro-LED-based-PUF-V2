"""Local fuzzy-extractor key regeneration for the micro-LED weak PUF.

This module implements a compact code-offset helper-data construction on top
of the released 512-bit response extractor.  It is intentionally local: the
enrollment manifest contains public helper data and quality thresholds, never
the enrolled raw PUF response, an image, or the generated root key.

The inner error-correction code is a public interleaved repetition code.  It
is simple enough for an edge-node reference implementation and, together with
the key check, makes an accepted reconstructed key exact.  It is not a claim
that the 512 response bits contain 512 independent bits of entropy; production
key length must be justified by a separate conditional min-entropy study.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import itertools
import json
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from microled_puf import PUFExtractor, condition_name, hamming, iter_images, majority_vote, read_rgb01


PROTOCOL = "microled-puf-code-offset-r1"
KEY_CHECK_CONTEXT = b"microled-puf-root-key-check-r1"
IDENTITY_CONTEXT = b"microled-puf-device-identity-seed-r1"


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def pack_bits(bits: np.ndarray) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").tobytes()


def unpack_bits(data: bytes, count: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="big")[:count].astype(np.uint8)


def hkdf_sha256(ikm: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """RFC 5869 HKDF using only the Python standard library."""
    prk = hmac.new(salt, ikm, hashlib.sha256).digest()
    output = bytearray()
    previous = b""
    counter = 1
    while len(output) < length:
        previous = hmac.new(prk, previous + info + bytes([counter]), hashlib.sha256).digest()
        output.extend(previous)
        counter += 1
    return bytes(output[:length])


def payload_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def response_rows(input_path: Path, payload_path: Path) -> list[dict[str, Any]]:
    """Extract bit responses and margins from aligned 256x256 RGB images."""
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
                "quality_template_corr": extractor.common_template_correlation(feature),
                **pose,
                # Preserve the published 512-bit extractor response for tools
                # that consume rows directly without a key-enrollment manifest.
                "bits": (candidate_margins[extractor.selected] > 0).astype(np.uint8),
            }
        )
    return rows


def select_stable_candidates(
    rows: Sequence[dict[str, Any]],
    response_bits: int,
    candidate_pool: Sequence[int] | None = None,
    require_unanimous: bool = True,
) -> np.ndarray:
    """Select per-device candidates stable across every enrolled condition.

    The minimum agreement across conditions is the primary score.  It prevents
    a projection that is stable at one temperature/current but flips under a
    permitted operating condition from entering the key-recovery code.
    """
    candidate_bits = np.stack([np.asarray(row["candidate_bits"], dtype=np.uint8) for row in rows], axis=0)
    candidate_margins = np.stack([np.asarray(row["candidate_margins"], dtype=np.float32) for row in rows], axis=0)
    reference = (candidate_bits.mean(axis=0) >= 0.5).astype(np.uint8)
    grouped: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault(str(row["condition"]), []).append(index)
    condition_agreement = np.stack(
        [(candidate_bits[indexes] == reference).mean(axis=0) for indexes in grouped.values()], axis=0
    )
    min_agreement = condition_agreement.min(axis=0)
    mean_agreement = (candidate_bits == reference).mean(axis=0)
    mean_margin = np.abs(candidate_margins).mean(axis=0)
    # np.lexsort uses the final key first: stability over every condition,
    # then global stability, then physical projection margin as a tie-breaker.
    allowed = np.arange(candidate_bits.shape[1], dtype=np.int64) if candidate_pool is None else np.asarray(candidate_pool, dtype=np.int64)
    if allowed.ndim != 1 or np.any(allowed < 0) or np.any(allowed >= candidate_bits.shape[1]):
        raise ValueError("Candidate pool contains out-of-range indices.")
    if np.unique(allowed).size != allowed.size:
        raise ValueError("Candidate pool contains duplicate indices.")
    if allowed.size < response_bits:
        raise ValueError("The global candidate pool is smaller than the required response size.")
    if not np.all(np.isfinite(candidate_margins[:, allowed])):
        raise ValueError("Enrollment candidate margins contain non-finite values.")
    unanimous_count = int(np.count_nonzero(min_agreement[allowed] == 1.0))
    if require_unanimous and unanimous_count < response_bits:
        raise ValueError(
            f"Failure to enroll: only {unanimous_count} candidates are unanimous across all "
            f"enrollment conditions; {response_bits} are required."
        )
    # A profile containing exactly ``response_bits`` candidates represents an
    # already-frozen shared support. Preserve its declared order so every
    # device evaluates the same physical projections in the same output-code
    # coordinates. Device-specific reliability and margin values are still
    # estimated below by the enrollment layer for soft LDPC decoding.
    if allowed.size == response_bits:
        return allowed.copy()
    order = allowed[np.lexsort((-mean_margin[allowed], -mean_agreement[allowed], -min_agreement[allowed]))]
    selected = order[:response_bits].astype(np.int64)
    if len(selected) != response_bits:
        raise ValueError("Not enough candidate projections for the requested response size.")
    return selected


def apply_candidate_selection(rows: Sequence[dict[str, Any]], indices: Sequence[int]) -> list[dict[str, Any]]:
    selected = np.asarray(indices, dtype=np.int64)
    return [{**row, "bits": np.asarray(row["candidate_bits"], dtype=np.uint8)[selected]} for row in rows]


def load_candidate_pool(path: Path) -> np.ndarray:
    data = np.load(path, allow_pickle=False)
    indices = np.asarray(data["eligible_indices"], dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("Candidate profile does not contain a non-empty eligible_indices array.")
    if np.any(indices < 0) or np.unique(indices).size != indices.size:
        raise ValueError("Candidate profile contains negative or duplicate indices.")
    return indices


def pairwise_hd(values: Sequence[np.ndarray]) -> list[float]:
    return [hamming(values[i], values[j]) for i in range(len(values)) for j in range(i + 1, len(values))]


def within_condition_threshold(rows: Sequence[dict[str, Any]], percentile: float = 95.0) -> float:
    grouped: dict[str, list[np.ndarray]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(np.asarray(row["bits"], dtype=np.uint8))
    distances: list[float] = []
    for values in grouped.values():
        distances.extend(pairwise_hd(values))
    if not distances:
        raise ValueError("At least two enrollment responses from one condition are required.")
    return float(np.percentile(np.asarray(distances, dtype=np.float32), percentile))


def response_quality(rows: Sequence[dict[str, Any]], pair_hd_limit: float) -> dict[str, Any]:
    """Find the largest mutually consistent frame subset in a short capture burst.

    A defective or partly occluded frame must not force a good capture burst
    to fail.  For at most nine frames, exhaustive subset selection is tiny
    (at most 512 subsets) and gives an unambiguous quality decision.
    """
    if len(rows) < 3:
        return {"passed": False, "max_pair_hd": float("nan"), "median_pair_hd": float("nan"), "selected_indices": [], "reason": "need_at_least_3_frames"}
    values = [np.asarray(row["bits"], dtype=np.uint8) for row in rows]
    distances = np.zeros((len(values), len(values)), dtype=np.float32)
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            distances[i, j] = distances[j, i] = hamming(values[i], values[j])
    best: tuple[int, ...] | None = None
    best_score: tuple[int, float, float] | None = None
    for size in range(3, len(values) + 1):
        for candidate in itertools.combinations(range(len(values)), size):
            pair_values = [float(distances[i, j]) for i, j in itertools.combinations(candidate, 2)]
            max_pair = max(pair_values)
            if max_pair > pair_hd_limit:
                continue
            score = (size, -max_pair, -float(np.median(pair_values)))
            if best_score is None or score > best_score:
                best, best_score = candidate, score
    if best is None:
        return {"passed": False, "max_pair_hd": float("nan"), "median_pair_hd": float("nan"), "selected_indices": [], "reason": "session_inconsistent"}
    selected_pairs = [float(distances[i, j]) for i, j in itertools.combinations(best, 2)]
    max_pair = float(max(selected_pairs))
    median_pair = float(np.median(selected_pairs))
    return {
        "passed": True,
        "max_pair_hd": max_pair,
        "median_pair_hd": median_pair,
        "selected_indices": list(best),
        "rejected_frames": len(rows) - len(best),
        "reason": "ok",
    }


@dataclass(frozen=True)
class EnrollmentManifest:
    protocol: str
    device_id: str
    response_bits: int
    key_bits: int
    repetition: int
    salt_b64: str
    helper_b64: str
    key_check_b64: str
    quality_pair_hd_limit: float
    quality_percentile: float
    payload_sha256: str
    identity_seed_id: str
    candidate_indices: list[int]
    candidate_pool_sha256: str

    @classmethod
    def load(cls, path: Path) -> "EnrollmentManifest":
        data = json.loads(path.read_text(encoding="utf-8"))
        manifest = cls(**data)
        if manifest.protocol != PROTOCOL:
            raise ValueError(f"Unsupported manifest protocol: {manifest.protocol}")
        if manifest.response_bits != manifest.key_bits * manifest.repetition:
            raise ValueError("response_bits must equal key_bits * repetition")
        return manifest

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @property
    def salt(self) -> bytes:
        return b64decode(self.salt_b64)

    @property
    def helper_bits(self) -> np.ndarray:
        return unpack_bits(b64decode(self.helper_b64), self.response_bits)

    @property
    def key_check(self) -> bytes:
        return b64decode(self.key_check_b64)


class InterleavedRepetitionCode:
    def __init__(self, response_bits: int, key_bits: int, repetition: int, salt: bytes) -> None:
        if response_bits != key_bits * repetition:
            raise ValueError("response_bits must equal key_bits * repetition")
        self.response_bits = response_bits
        self.key_bits = key_bits
        self.repetition = repetition
        seed = int.from_bytes(hashlib.sha256(b"microled-puf-interleaver-r1" + salt).digest()[:8], "little")
        self.positions = np.random.default_rng(seed).permutation(response_bits)

    def encode(self, secret: bytes) -> np.ndarray:
        source = unpack_bits(secret, self.key_bits)
        canonical = np.repeat(source, self.repetition)
        codeword = np.zeros(self.response_bits, dtype=np.uint8)
        codeword[self.positions] = canonical
        return codeword

    def decode(self, observed_codeword: np.ndarray) -> tuple[bytes, int]:
        return self.decode_observations([observed_codeword])

    def decode_observations(self, observed_codewords: Sequence[np.ndarray]) -> tuple[bytes, int]:
        canonical = np.stack([np.asarray(value, dtype=np.uint8)[self.positions] for value in observed_codewords], axis=0)
        groups = canonical.reshape(len(observed_codewords), self.key_bits, self.repetition)
        ones = groups.sum(axis=(0, 2))
        total = groups.shape[0] * self.repetition
        secret_bits = (ones * 2 >= total).astype(np.uint8)
        ambiguous_groups = int(np.count_nonzero(ones * 2 == total))
        return pack_bits(secret_bits), ambiguous_groups


def root_key(secret: bytes, manifest: EnrollmentManifest) -> bytes:
    info = (PROTOCOL + "|" + manifest.device_id).encode("utf-8")
    return hkdf_sha256(secret, manifest.salt, info, 32)


def identity_seed(secret: bytes, manifest: EnrollmentManifest) -> bytes:
    info = IDENTITY_CONTEXT + b"|" + manifest.device_id.encode("utf-8")
    return hkdf_sha256(secret, manifest.salt, info, 32)


def key_check(key: bytes, manifest: EnrollmentManifest) -> bytes:
    message = KEY_CHECK_CONTEXT + b"|" + manifest.device_id.encode("utf-8")
    return hmac.new(key, message, hashlib.sha256).digest()


def enroll(
    device_id: str,
    rows: Sequence[dict[str, Any]],
    payload_path: Path,
    key_bits: int = 128,
    repetition: int = 4,
    quality_percentile: float = 95.0,
    candidate_pool: Sequence[int] | None = None,
) -> EnrollmentManifest:
    if not rows:
        raise ValueError("No enrollment images were found.")
    response_bits = key_bits * repetition
    candidate_indices = select_stable_candidates(rows, response_bits, candidate_pool)
    selected_rows = apply_candidate_selection(rows, candidate_indices)
    if response_bits != key_bits * repetition:
        raise ValueError(f"This code needs {key_bits * repetition} response bits, got {response_bits}.")
    reference = majority_vote([np.asarray(row["bits"], dtype=np.uint8) for row in selected_rows])
    salt = secrets.token_bytes(16)
    code = InterleavedRepetitionCode(response_bits, key_bits, repetition, salt)
    secret = secrets.token_bytes((key_bits + 7) // 8)
    helper = reference ^ code.encode(secret)
    provisional = EnrollmentManifest(
        protocol=PROTOCOL,
        device_id=device_id,
        response_bits=response_bits,
        key_bits=key_bits,
        repetition=repetition,
        salt_b64=b64encode(salt),
        helper_b64=b64encode(pack_bits(helper)),
        key_check_b64="",
        quality_pair_hd_limit=within_condition_threshold(selected_rows, quality_percentile),
        quality_percentile=quality_percentile,
        payload_sha256=payload_digest(payload_path),
        identity_seed_id="",
        candidate_indices=[int(value) for value in candidate_indices],
        candidate_pool_sha256=(
            hashlib.sha256(np.asarray(candidate_pool, dtype=np.int64).tobytes()).hexdigest() if candidate_pool is not None else "unconstrained"
        ),
    )
    k_root = root_key(secret, provisional)
    seed_id = hashlib.sha256(identity_seed(secret, provisional)).hexdigest()[:24]
    return EnrollmentManifest(
        **{
            **asdict(provisional),
            "key_check_b64": b64encode(key_check(k_root, provisional)),
            "identity_seed_id": seed_id,
        }
    )


def reconstruct(manifest: EnrollmentManifest, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected_input = apply_candidate_selection(rows, manifest.candidate_indices)
    quality = response_quality(selected_input, manifest.quality_pair_hd_limit)
    if not bool(quality["passed"]):
        return {"accepted": False, "key_id": None, "ambiguous_groups": None, "quality": quality}
    selected_rows = [selected_input[index] for index in quality["selected_indices"]]
    code = InterleavedRepetitionCode(manifest.response_bits, manifest.key_bits, manifest.repetition, manifest.salt)
    observations = [np.asarray(row["bits"], dtype=np.uint8) ^ manifest.helper_bits for row in selected_rows]
    secret, ambiguous_groups = code.decode_observations(observations)
    recovered_key = root_key(secret, manifest)
    accepted = hmac.compare_digest(key_check(recovered_key, manifest), manifest.key_check)
    return {
        "accepted": accepted,
        "key_id": hashlib.sha256(recovered_key).hexdigest()[:24] if accepted else None,
        "identity_seed_id": hashlib.sha256(identity_seed(secret, manifest)).hexdigest()[:24] if accepted else None,
        "ambiguous_groups": ambiguous_groups,
        "selected_frames": len(selected_rows),
        "quality": quality,
        "root_key_b64": b64encode(recovered_key) if accepted else None,
    }


def reconstruct_adaptive(manifest: EnrollmentManifest, rows: Sequence[dict[str, Any]], vote_sizes: Sequence[int] = (3, 5, 9)) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for size in vote_sizes:
        if len(rows) < size:
            continue
        result = reconstruct(manifest, rows[:size])
        result["frames_acquired"] = size
        attempts.append(result)
        if bool(result["accepted"]):
            # Copy the attempt records before attaching them: placing result
            # itself inside result["attempts"] would create a circular JSON object.
            return {**result, "attempts": [dict(attempt) for attempt in attempts]}
    return {"accepted": False, "frames_acquired": None, "attempts": attempts, "key_id": None}


def default_payload_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "expanded_luma_support_payload.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    enroll_cmd = sub.add_parser("enroll", help="Create a public helper-data manifest from aligned enrollment images.")
    enroll_cmd.add_argument("--input", type=Path, required=True)
    enroll_cmd.add_argument("--device-id", required=True)
    enroll_cmd.add_argument("--manifest", type=Path, required=True)
    enroll_cmd.add_argument("--payload", type=Path, default=default_payload_path())
    enroll_cmd.add_argument("--candidate-profile", type=Path, default=None, help="Frozen candidate universe for device-local stability-only selection.")
    enroll_cmd.add_argument("--quality-percentile", type=float, default=95.0)

    reproduce_cmd = sub.add_parser("reproduce", help="Regenerate the local root key from new aligned images.")
    reproduce_cmd.add_argument("--input", type=Path, required=True)
    reproduce_cmd.add_argument("--manifest", type=Path, required=True)
    reproduce_cmd.add_argument("--payload", type=Path, default=default_payload_path())
    reproduce_cmd.add_argument("--vote-sizes", type=int, nargs="*", default=[3, 5, 9])
    reproduce_cmd.add_argument("--show-root-key", action="store_true", help="Print the reconstructed key; use only on a protected test host.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "enroll":
        rows = response_rows(args.input, args.payload)
        pool = load_candidate_pool(args.candidate_profile) if args.candidate_profile else None
        manifest = enroll(args.device_id, rows, args.payload, quality_percentile=args.quality_percentile, candidate_pool=pool)
        manifest.save(args.manifest)
        print(json.dumps({"manifest": str(args.manifest), "device_id": manifest.device_id, "quality_pair_hd_limit": manifest.quality_pair_hd_limit, "identity_seed_id": manifest.identity_seed_id}, indent=2))
        return

    manifest = EnrollmentManifest.load(args.manifest)
    if manifest.payload_sha256 != payload_digest(args.payload):
        raise SystemExit("Extractor payload digest does not match the enrollment manifest.")
    result = reconstruct_adaptive(manifest, response_rows(args.input, args.payload), args.vote_sizes)
    if not args.show_root_key:
        result.pop("root_key_b64", None)
        for attempt in result.get("attempts", []):
            attempt.pop("root_key_b64", None)
    print(json.dumps(result, indent=2, allow_nan=False))
    if not bool(result["accepted"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
