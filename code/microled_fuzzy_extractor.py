"""Single-image 2,048-bit response reproduction with one LDPC fuzzy extractor.

Enrollment reads aligned images from the declared operating conditions.
Reproduction reads one aligned image and the enrollment manifest.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import math
import re
import secrets
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from microled_puf import PUFExtractor, condition_name, device_name, iter_images, majority_vote, read_rgb01
PROTOCOL = "microled-puf-single-shot-ldpc-r1"
ROOT_CONTEXT = b"microled-puf-single-shot-root-r1"
IDENTITY_CONTEXT = b"microled-puf-single-shot-identity-r1"
CHECK_CONTEXT = b"microled-puf-single-shot-check-r1"
DEFAULT_RESPONSE_BITS = 2048
DEFAULT_CHECKS = 1536
DEFAULT_VARIABLE_DEGREE = 3
DEFAULT_CHECK_DEGREE = 4
DEFAULT_GRAPH_SEED = 20260710
DEFAULT_QUALITY_TEMPLATE_CORR_MIN = 0.40
EXPECTED_CURRENTS_MA = (10, 20, 30)
EXPECTED_TEMPERATURES_C = (20, 30, 40)
CONDITION_PATTERN = re.compile(r"^(M\d+)_(\d+)mA_(\d+)C_0$", re.IGNORECASE)
MANIFEST_AUTH_CONTEXT = b"microled-puf-manifest-auth-r1|"


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"), validate=True)


def pack_bits(bits: np.ndarray) -> bytes:
    return np.packbits(
        np.asarray(bits, dtype=np.uint8),
        bitorder="big",
    ).tobytes()


def hkdf_sha256(
    ikm: bytes,
    salt: bytes,
    info: bytes,
    length: int,
) -> bytes:
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


def response_rows(
    input_path: Path,
    payload_path: Path,
) -> list[dict[str, Any]]:
    extractor = PUFExtractor.from_payload(payload_path)
    root = input_path if input_path.is_dir() else input_path.parent
    rows: list[dict[str, Any]] = []
    for path in iter_images(input_path):
        rgb = read_rgb01(path)
        feature, pose = extractor.feature_and_pose_from_rgb(rgb)
        margins = extractor.candidate_margins_from_feature(feature)
        rows.append(
            {
                "path": str(path),
                "condition": condition_name(path, root),
                "candidate_bits": (margins > 0).astype(np.uint8),
                "candidate_margins": margins,
                "quality_template_corr": (
                    extractor.common_template_correlation(feature)
                ),
                **pose,
                "bits": (
                    margins[extractor.selected] > 0
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


def encode_float16(values: np.ndarray) -> str:
    return b64encode(np.asarray(values, dtype="<f2").tobytes())


def decode_float16(text: str, count: int) -> np.ndarray:
    values = np.frombuffer(b64decode(text), dtype="<f2").astype(np.float32)
    if values.size != count:
        raise ValueError(f"Expected {count} float16 values, got {values.size}.")
    return values


def unpack_helper_bits(text: str, count: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b64decode(text), dtype=np.uint8), bitorder="big")[:count].astype(np.uint8)


def read_auth_key(path: Path) -> bytes:
    key = path.read_bytes()
    if len(key) < 32:
        raise ValueError("Manifest tag key must contain at least 32 bytes.")
    return key


def manifest_auth_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".hmac")


class RegularLDPC:
    """Deterministic regular LDPC graph with normalized min-sum decoding."""

    def __init__(self, n: int, m: int, dv: int, dc: int, seed: int) -> None:
        if n * dv != m * dc:
            raise ValueError("Regular LDPC graph requires n*dv == m*dc.")
        self.n = n
        self.m = m
        self.dv = dv
        self.dc = dc
        self.seed = seed
        self.check_vars = self._build_graph()
        self.edge_vars = self.check_vars.ravel()

    def _build_graph(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        variable_sockets = np.repeat(np.arange(self.n, dtype=np.int32), self.dv)
        base_check_sockets = np.repeat(np.arange(self.m, dtype=np.int32), self.dc)
        for _ in range(512):
            check_sockets = base_check_sockets.copy()
            rng.shuffle(check_sockets)
            order = np.argsort(check_sockets, kind="stable")
            check_vars = variable_sockets[order].reshape(self.m, self.dc)
            sorted_rows = np.sort(check_vars, axis=1)
            if not np.any(np.diff(sorted_rows, axis=1) == 0):
                return check_vars
        raise RuntimeError("Could not generate a duplicate-free regular LDPC graph.")

    def syndrome(self, bits: np.ndarray) -> np.ndarray:
        values = np.asarray(bits, dtype=np.uint8)
        return (values[self.check_vars].sum(axis=1) & 1).astype(np.uint8)

    def rank(self) -> int:
        """Compute GF(2) row rank for helper-data leakage accounting."""
        basis: dict[int, int] = {}
        for variables in self.check_vars:
            value = 0
            for variable in variables:
                value ^= 1 << int(variable)
            while value:
                pivot = value.bit_length() - 1
                if pivot not in basis:
                    basis[pivot] = value
                    break
                value ^= basis[pivot]
        return len(basis)

    def decode_error(
        self,
        target_syndrome: np.ndarray,
        prior_llr: np.ndarray,
        max_iterations: int = 40,
        alpha: float = 0.80,
    ) -> dict[str, Any]:
        """Decode the most likely error vector for H*e = target_syndrome."""
        target = np.asarray(target_syndrome, dtype=np.uint8)
        prior = np.asarray(prior_llr, dtype=np.float32)
        if target.shape != (self.m,) or prior.shape != (self.n,):
            raise ValueError("LDPC syndrome or LLR shape mismatch.")
        zero = np.zeros(self.n, dtype=np.uint8)
        if not np.any(target):
            return {"converged": True, "error": zero, "iterations": 0, "unsatisfied_checks": 0}

        variable_to_check = prior[self.edge_vars].reshape(self.m, self.dc).copy()
        columns = np.arange(self.dc)[None, :]
        hard = zero
        unsatisfied = self.m
        for iteration in range(1, max_iterations + 1):
            signs = np.where(variable_to_check >= 0.0, 1.0, -1.0)
            magnitudes = np.abs(variable_to_check)
            min_index = np.argmin(magnitudes, axis=1)
            min_value = magnitudes[np.arange(self.m), min_index]
            second_value = np.partition(magnitudes, 1, axis=1)[:, 1]
            outgoing_magnitude = np.where(columns == min_index[:, None], second_value[:, None], min_value[:, None])
            parity_sign = np.where(target == 0, 1.0, -1.0)
            outgoing_sign = parity_sign[:, None] * np.prod(signs, axis=1)[:, None] * signs
            check_to_variable = alpha * outgoing_sign * outgoing_magnitude

            posterior = prior + np.bincount(
                self.edge_vars,
                weights=check_to_variable.ravel(),
                minlength=self.n,
            ).astype(np.float32)
            hard = (posterior < 0.0).astype(np.uint8)
            residual = self.syndrome(hard) ^ target
            unsatisfied = int(residual.sum())
            if unsatisfied == 0:
                return {"converged": True, "error": hard, "iterations": iteration, "unsatisfied_checks": 0}
            variable_to_check = np.clip(
                posterior[self.edge_vars].reshape(self.m, self.dc) - check_to_variable,
                -30.0,
                30.0,
            )
        return {
            "converged": False,
            "error": hard,
            "iterations": max_iterations,
            "unsatisfied_checks": unsatisfied,
        }


@lru_cache(maxsize=8)
def get_regular_ldpc(n: int, m: int, dv: int, dc: int, seed: int) -> RegularLDPC:
    """Build each immutable Tanner graph once per process."""
    return RegularLDPC(n, m, dv, dc, seed)


@dataclass(frozen=True)
class SingleShotManifest:
    protocol: str
    device_id: str
    response_bits: int
    check_bits: int
    variable_degree: int
    check_degree: int
    graph_seed: int
    graph_rank: int
    candidate_indices: list[int]
    syndrome_b64: str
    reliability_llr_f16_b64: str
    margin_scale_f16_b64: str
    salt_b64: str
    key_check_b64: str
    identity_seed_id: str
    payload_sha256: str
    candidate_profile_sha256: str
    yolo_model_sha256: str
    stn_model_sha256: str
    enrollment_images: int
    enrollment_conditions: list[str]
    quality_template_corr_min: float

    @classmethod
    def load(
        cls,
        path: Path,
        auth_key: bytes | None = None,
        require_auth: bool = True,
    ) -> "SingleShotManifest":
        raw = path.read_bytes()
        if auth_key is not None:
            tag_path = manifest_auth_path(path)
            if not tag_path.is_file():
                raise ValueError(f"Missing manifest authentication tag: {tag_path}")
            expected = hmac.new(auth_key, MANIFEST_AUTH_CONTEXT + raw, hashlib.sha256).hexdigest()
            supplied = tag_path.read_text(encoding="ascii").strip()
            if not hmac.compare_digest(expected, supplied):
                raise ValueError("Manifest authentication failed.")
        elif require_auth:
            raise ValueError("A manifest tag key is required.")
        manifest = cls(**json.loads(raw.decode("utf-8")))
        manifest.validate()
        return manifest

    def save(self, path: Path, auth_key: bytes | None = None) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (json.dumps(asdict(self), indent=2, sort_keys=True) + "\n").encode("utf-8")
        path.write_bytes(raw)
        if auth_key is not None:
            tag = hmac.new(auth_key, MANIFEST_AUTH_CONTEXT + raw, hashlib.sha256).hexdigest()
            manifest_auth_path(path).write_text(tag + "\n", encoding="ascii")

    def validate(self) -> None:
        if self.protocol != PROTOCOL:
            raise ValueError(f"Unsupported protocol: {self.protocol}")
        if self.response_bits <= 0 or self.check_bits <= 0:
            raise ValueError("Manifest response/check sizes must be positive.")
        if self.response_bits * self.variable_degree != self.check_bits * self.check_degree:
            raise ValueError("Manifest LDPC degrees do not define a regular graph.")
        indices = np.asarray(self.candidate_indices, dtype=np.int64)
        if indices.shape != (self.response_bits,) or np.any(indices < 0):
            raise ValueError("Manifest candidate index count or range is invalid.")
        if np.unique(indices).size != indices.size:
            raise ValueError("Manifest candidate indices must be unique.")
        if len(b64decode(self.syndrome_b64)) != (self.check_bits + 7) // 8:
            raise ValueError("Manifest syndrome byte length is invalid.")
        reliability = self.reliability_llr
        scale = self.margin_scale
        if not np.all(np.isfinite(reliability)) or np.any(reliability <= 0):
            raise ValueError("Manifest reliability LLR values must be finite and positive.")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("Manifest margin scales must be finite and positive.")
        if len(self.salt) < 16 or len(self.key_check) != hashlib.sha256().digest_size:
            raise ValueError("Manifest salt or key-check length is invalid.")
        if not math.isfinite(self.quality_template_corr_min) or not -1.0 <= self.quality_template_corr_min <= 1.0:
            raise ValueError("Manifest quality threshold must be finite and within [-1, 1].")
        if self.enrollment_images != 9 or len(self.enrollment_conditions) != 9:
            raise ValueError("Final scheme requires exactly nine enrollment images and conditions.")
        parsed = [CONDITION_PATTERN.fullmatch(value) for value in self.enrollment_conditions]
        if any(value is None for value in parsed):
            raise ValueError("Manifest contains an invalid enrollment condition name.")
        source_devices = {value.group(1).upper() for value in parsed if value is not None}
        expected = {
            f"{next(iter(source_devices))}_{current}mA_{temperature}C_0"
            for current in EXPECTED_CURRENTS_MA
            for temperature in EXPECTED_TEMPERATURES_C
        }
        if len(source_devices) != 1 or set(self.enrollment_conditions) != expected:
            raise ValueError("Manifest enrollment conditions are not the required 3x3 operating grid.")
        graph = get_regular_ldpc(
            self.response_bits,
            self.check_bits,
            self.variable_degree,
            self.check_degree,
            self.graph_seed,
        )
        if graph.rank() != self.graph_rank:
            raise ValueError("Manifest LDPC rank does not match its graph parameters.")
        for name, value in {
            "payload": self.payload_sha256,
            "candidate profile": self.candidate_profile_sha256,
            "YOLO model": self.yolo_model_sha256,
            "STN model": self.stn_model_sha256,
        }.items():
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"Manifest {name} SHA-256 is invalid.")

    @property
    def syndrome(self) -> np.ndarray:
        return unpack_helper_bits(self.syndrome_b64, self.check_bits)

    @property
    def reliability_llr(self) -> np.ndarray:
        return decode_float16(self.reliability_llr_f16_b64, self.response_bits)

    @property
    def margin_scale(self) -> np.ndarray:
        return decode_float16(self.margin_scale_f16_b64, self.response_bits)

    @property
    def salt(self) -> bytes:
        return b64decode(self.salt_b64)

    @property
    def key_check(self) -> bytes:
        return b64decode(self.key_check_b64)


def select_per_condition(
    rows: Sequence[dict[str, Any]],
    per_condition: int,
    quality_template_corr_min: float = DEFAULT_QUALITY_TEMPLATE_CORR_MIN,
) -> list[dict[str, Any]]:
    if per_condition <= 0:
        raise ValueError("per_condition must be positive.")
    if not math.isfinite(quality_template_corr_min) or not -1.0 <= quality_template_corr_min <= 1.0:
        raise ValueError("Quality threshold must be finite and within [-1, 1].")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), []).append(row)
    parsed = {condition: CONDITION_PATTERN.fullmatch(condition) for condition in grouped}
    if any(value is None for value in parsed.values()):
        invalid = sorted(condition for condition, value in parsed.items() if value is None)
        raise ValueError(f"Unexpected enrollment condition names: {invalid}")
    devices = {value.group(1).upper() for value in parsed.values() if value is not None}
    if len(devices) != 1:
        raise ValueError(f"Enrollment must contain exactly one source device, got {sorted(devices)}.")
    device = next(iter(devices))
    expected = [
        f"{device}_{current}mA_{temperature}C_0"
        for current in EXPECTED_CURRENTS_MA
        for temperature in EXPECTED_TEMPERATURES_C
    ]
    if set(grouped) != set(expected):
        missing = sorted(set(expected) - set(grouped))
        extra = sorted(set(grouped) - set(expected))
        raise ValueError(f"Enrollment requires the exact 3x3 condition grid; missing={missing}, extra={extra}.")
    selected: list[dict[str, Any]] = []
    for condition in expected:
        valid = [
            row
            for row in grouped[condition]
            if math.isfinite(float(row.get("quality_template_corr", float("nan"))))
            and float(row["quality_template_corr"]) >= quality_template_corr_min
        ]
        if len(valid) < per_condition:
            raise ValueError(
                f"Failure to enroll: {condition} has {len(valid)} quality-passed images; "
                f"{per_condition} are required."
            )
        selected.extend(valid[:per_condition])
    return selected


def derive_root_key(reference: np.ndarray, salt: bytes, device_id: str) -> bytes:
    info = ROOT_CONTEXT + b"|" + device_id.encode("utf-8")
    return hkdf_sha256(pack_bits(reference), salt, info, 32)


def derive_identity_seed(reference: np.ndarray, salt: bytes, device_id: str) -> bytes:
    info = IDENTITY_CONTEXT + b"|" + device_id.encode("utf-8")
    return hkdf_sha256(pack_bits(reference), salt, info, 32)


def make_key_check(key: bytes, device_id: str) -> bytes:
    return hmac.new(key, CHECK_CONTEXT + b"|" + device_id.encode("utf-8"), hashlib.sha256).digest()


def enroll_single_shot(
    device_id: str,
    rows: Sequence[dict[str, Any]],
    payload_path: Path,
    candidate_profile_path: Path,
    response_bits: int = DEFAULT_RESPONSE_BITS,
    quality_template_corr_min: float = DEFAULT_QUALITY_TEMPLATE_CORR_MIN,
    models_dir: Path | None = None,
    require_unanimous: bool | None = None,
) -> SingleShotManifest:
    if len(rows) != 9:
        raise ValueError("Final single-shot enrollment requires exactly nine images.")
    if not math.isfinite(quality_template_corr_min) or not -1.0 <= quality_template_corr_min <= 1.0:
        raise ValueError("Quality threshold must be finite and within [-1, 1].")
    for row in rows:
        quality = float(row.get("quality_template_corr", float("nan")))
        if not math.isfinite(quality) or quality < quality_template_corr_min:
            raise ValueError(f"Failure to enroll: image does not pass the quality gate: {row.get('path')}.")
    candidate_pool = load_candidate_pool(candidate_profile_path)
    profile = np.load(candidate_profile_path, allow_pickle=False)
    frozen_shared_support = (
        "support_order_frozen" in profile.files
        and bool(np.asarray(profile["support_order_frozen"]).reshape(-1)[0])
    )
    if require_unanimous is None:
        require_unanimous = not frozen_shared_support
    candidate_indices = select_stable_candidates(
        rows,
        response_bits,
        candidate_pool,
        require_unanimous=require_unanimous,
    )
    selected_rows = apply_candidate_selection(rows, candidate_indices)
    bit_stack = np.stack([np.asarray(row["bits"], dtype=np.uint8) for row in selected_rows])
    reference = majority_vote(list(bit_stack))
    flip_count = (bit_stack != reference).sum(axis=0).astype(np.float32)
    error_probability = np.clip((flip_count + 0.5) / (bit_stack.shape[0] + 1.0), 1e-3, 0.25)
    reliability_llr = np.log((1.0 - error_probability) / error_probability).astype(np.float32)
    margin_stack = np.stack(
        [np.abs(np.asarray(row["candidate_margins"], dtype=np.float32)[candidate_indices]) for row in rows]
    )
    margin_scale = np.maximum(np.median(margin_stack, axis=0), 1e-5).astype(np.float32)

    graph = get_regular_ldpc(
        response_bits,
        DEFAULT_CHECKS,
        DEFAULT_VARIABLE_DEGREE,
        DEFAULT_CHECK_DEGREE,
        DEFAULT_GRAPH_SEED,
    )
    salt = secrets.token_bytes(16)
    root_key = derive_root_key(reference, salt, device_id)
    identity = derive_identity_seed(reference, salt, device_id)
    model_root = models_dir if models_dir is not None else Path(__file__).resolve().parents[1] / "models"
    yolo_path = model_root / "yolo11n_microled_best.pt"
    stn_path = model_root / "luma_spatial_head_stn.pt"
    if not yolo_path.is_file() or not stn_path.is_file():
        raise FileNotFoundError("YOLO/STN model files are required for manifest version binding.")
    manifest = SingleShotManifest(
        protocol=PROTOCOL,
        device_id=device_id,
        response_bits=response_bits,
        check_bits=DEFAULT_CHECKS,
        variable_degree=DEFAULT_VARIABLE_DEGREE,
        check_degree=DEFAULT_CHECK_DEGREE,
        graph_seed=DEFAULT_GRAPH_SEED,
        graph_rank=graph.rank(),
        candidate_indices=[int(value) for value in candidate_indices],
        syndrome_b64=b64encode(pack_bits(graph.syndrome(reference))),
        reliability_llr_f16_b64=encode_float16(reliability_llr),
        margin_scale_f16_b64=encode_float16(margin_scale),
        salt_b64=b64encode(salt),
        key_check_b64=b64encode(make_key_check(root_key, device_id)),
        identity_seed_id=hashlib.sha256(identity).hexdigest()[:24],
        payload_sha256=payload_digest(payload_path),
        candidate_profile_sha256=payload_digest(candidate_profile_path),
        yolo_model_sha256=payload_digest(yolo_path),
        stn_model_sha256=payload_digest(stn_path),
        enrollment_images=len(rows),
        enrollment_conditions=sorted({str(row["condition"]) for row in rows}),
        quality_template_corr_min=quality_template_corr_min,
    )
    manifest.validate()
    return manifest


def reproduce_single_shot(
    manifest: SingleShotManifest,
    row: dict[str, Any],
    max_iterations: int = 40,
    decoder_alpha: float = 0.80,
) -> dict[str, Any]:
    if not bool(row.get("pose_gate_passed", True)):
        return {
            "accepted": False,
            "key_id": None,
            "identity_seed_id": None,
            "failure_stage": "pose_gate",
            "quality_template_corr": float(row.get("quality_template_corr", float("nan"))),
            "quality_template_corr_min": manifest.quality_template_corr_min,
            "decoder_attempted": False,
            "decoder_converged": False,
            "iterations": 0,
            "unsatisfied_checks": None,
            "estimated_error_weight": None,
            "median_margin_ratio": None,
            "root_key_b64": None,
        }
    quality_corr = float(row.get("quality_template_corr", float("nan")))
    if not np.isfinite(quality_corr) or quality_corr < manifest.quality_template_corr_min:
        return {
            "accepted": False,
            "key_id": None,
            "identity_seed_id": None,
            "failure_stage": "quality_gate",
            "quality_template_corr": quality_corr,
            "quality_template_corr_min": manifest.quality_template_corr_min,
            "decoder_attempted": False,
            "decoder_converged": False,
            "iterations": 0,
            "unsatisfied_checks": None,
            "estimated_error_weight": None,
            "median_margin_ratio": None,
            "root_key_b64": None,
        }
    if "candidate_margins" not in row and "selected_candidate_margins" not in row:
        raise ValueError("Quality-passed row does not contain candidate margins.")
    indices = np.asarray(manifest.candidate_indices, dtype=np.int64)
    if "selected_candidate_margins" in row:
        margins = np.asarray(row["selected_candidate_margins"], dtype=np.float32)
        if margins.size != manifest.response_bits:
            raise ValueError("Selected candidate margin count does not match the manifest.")
    else:
        candidate_margins = np.asarray(row["candidate_margins"], dtype=np.float32)
        if np.any(indices >= candidate_margins.size):
            raise ValueError("Manifest candidate index exceeds the available candidate bank.")
        margins = candidate_margins[indices]
    if not np.all(np.isfinite(margins)):
        raise ValueError("Selected candidate margins contain non-finite values.")
    observed = (margins > 0).astype(np.uint8)
    ratio = np.clip(np.abs(margins) / np.maximum(manifest.margin_scale, 1e-5), 0.15, 3.0)
    prior_llr = np.clip(manifest.reliability_llr * np.sqrt(ratio), 0.05, 12.0)
    graph = get_regular_ldpc(
        manifest.response_bits,
        manifest.check_bits,
        manifest.variable_degree,
        manifest.check_degree,
        manifest.graph_seed,
    )
    target = graph.syndrome(observed) ^ manifest.syndrome
    decoded = graph.decode_error(target, prior_llr, max_iterations=max_iterations, alpha=decoder_alpha)
    recovered = observed ^ np.asarray(decoded["error"], dtype=np.uint8)
    recovered_key = derive_root_key(recovered, manifest.salt, manifest.device_id)
    accepted = bool(decoded["converged"]) and hmac.compare_digest(
        make_key_check(recovered_key, manifest.device_id), manifest.key_check
    )
    return {
        "accepted": accepted,
        "key_id": hashlib.sha256(recovered_key).hexdigest()[:24] if accepted else None,
        "identity_seed_id": (
            hashlib.sha256(derive_identity_seed(recovered, manifest.salt, manifest.device_id)).hexdigest()[:24]
            if accepted
            else None
        ),
        "failure_stage": None if accepted else "decoder_or_key_check",
        "quality_template_corr": quality_corr,
        "quality_template_corr_min": manifest.quality_template_corr_min,
        "decoder_attempted": True,
        "decoder_converged": bool(decoded["converged"]),
        "iterations": int(decoded["iterations"]),
        "unsatisfied_checks": int(decoded["unsatisfied_checks"]),
        "estimated_error_weight": int(np.asarray(decoded["error"], dtype=np.uint8).sum()),
        "median_margin_ratio": float(np.median(ratio)),
        "root_key_b64": b64encode(recovered_key) if accepted else None,
    }


def default_payload_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "expanded_luma_support_payload.npz"


def default_profile_path() -> Path:
    return Path(__file__).resolve().parents[1] / "models" / "stability_only_candidate_profile.npz"


def default_models_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "models"


def enrollment_response_rows(input_path: Path, payload_path: Path, source_device: str | None) -> list[dict[str, Any]]:
    """Extract only the requested device when a population root is supplied."""
    if source_device and input_path.is_dir():
        condition_dirs = [
            path
            for path in input_path.iterdir()
            if path.is_dir() and device_name(path.name) == source_device.upper()
        ]
        if condition_dirs:
            rows: list[dict[str, Any]] = []
            for condition_dir in sorted(condition_dirs):
                rows.extend(response_rows(condition_dir, payload_path))
            return rows
    return response_rows(input_path, payload_path)


def quality_gated_single_image_row(
    input_path: Path,
    payload_path: Path,
    threshold: float,
    candidate_indices: Sequence[int],
) -> dict[str, Any]:
    """Run structural quality scoring before evaluating sparse-projection bits."""
    paths = iter_images(input_path)
    if len(paths) != 1:
        raise ValueError(f"Single-shot reproduction requires exactly one image, got {len(paths)}.")
    path = paths[0]
    extractor = PUFExtractor.from_payload(payload_path)
    feature, pose = extractor.feature_and_pose_from_rgb(read_rgb01(path))
    quality_corr = extractor.common_template_correlation(feature)
    row: dict[str, Any] = {
        "path": str(path),
        "condition": condition_name(path, input_path.parent if input_path.is_file() else input_path),
        "quality_template_corr": quality_corr,
        **pose,
    }
    if not bool(pose["pose_gate_passed"]):
        return row
    if not np.isfinite(quality_corr) or quality_corr < threshold:
        return row
    margins = extractor.selected_candidate_margins_from_feature(feature, candidate_indices)
    row["selected_candidate_margins"] = margins
    row["candidate_bits"] = (margins > 0).astype(np.uint8)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    enroll_cmd = sub.add_parser("enroll")
    enroll_cmd.add_argument("--input", type=Path, required=True)
    enroll_cmd.add_argument("--device-id", required=True)
    enroll_cmd.add_argument("--source-device", default=None, help="Condition-prefix device name, e.g. M1, when --input contains multiple devices.")
    enroll_cmd.add_argument("--manifest", type=Path, required=True)
    enroll_cmd.add_argument("--payload", type=Path, default=default_payload_path())
    enroll_cmd.add_argument("--candidate-profile", type=Path, default=default_profile_path())
    enroll_cmd.add_argument("--per-condition", type=int, default=1)
    enroll_cmd.add_argument("--quality-corr-min", type=float, default=DEFAULT_QUALITY_TEMPLATE_CORR_MIN)
    enroll_cmd.add_argument("--models-dir", type=Path, default=default_models_dir())
    enroll_cmd.add_argument("--manifest-auth-key", type=Path, default=None)
    enroll_cmd.add_argument("--allow-unsigned-manifest", action="store_true")

    reproduce_cmd = sub.add_parser("reproduce")
    reproduce_cmd.add_argument("--input", type=Path, required=True, help="Exactly one aligned RGB image.")
    reproduce_cmd.add_argument("--manifest", type=Path, required=True)
    reproduce_cmd.add_argument("--payload", type=Path, default=default_payload_path())
    reproduce_cmd.add_argument("--candidate-profile", type=Path, default=default_profile_path())
    reproduce_cmd.add_argument("--models-dir", type=Path, default=default_models_dir())
    reproduce_cmd.add_argument("--manifest-auth-key", type=Path, default=None)
    reproduce_cmd.add_argument("--allow-unsigned-manifest", action="store_true")
    reproduce_cmd.add_argument("--max-iterations", type=int, default=40)
    reproduce_cmd.add_argument("--decoder-alpha", type=float, default=0.80)
    reproduce_cmd.add_argument("--show-root-key", action="store_true")

    init_cmd = sub.add_parser(
        "init-auth-key",
        help="Create a 256-bit manifest tag key.",
    )
    init_cmd.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init-auth-key":
        if args.out.exists():
            raise SystemExit(f"Refusing to overwrite existing tag key: {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(secrets.token_bytes(32))
        print(json.dumps({"manifest_auth_key": str(args.out), "bytes": 32}, indent=2))
        return
    if args.command == "enroll":
        if args.per_condition != 1:
            raise SystemExit("The final scheme requires --per-condition 1.")
        if args.manifest_auth_key is None and not args.allow_unsigned_manifest:
            raise SystemExit(
                "Provide --manifest-auth-key, or use "
                "--allow-unsigned-manifest."
            )
        all_rows = enrollment_response_rows(args.input, args.payload, args.source_device)
        source_devices = sorted({device_name(str(row["condition"])) for row in all_rows})
        if args.source_device:
            all_rows = [row for row in all_rows if device_name(str(row["condition"])) == args.source_device.upper()]
            if not all_rows:
                raise SystemExit(f"No rows found for source device {args.source_device}.")
        elif len(source_devices) > 1:
            raise SystemExit(f"Enrollment input contains multiple devices {source_devices}; provide --source-device.")
        rows = select_per_condition(all_rows, args.per_condition, args.quality_corr_min)
        manifest = enroll_single_shot(
            args.device_id,
            rows,
            args.payload,
            args.candidate_profile,
            quality_template_corr_min=args.quality_corr_min,
            models_dir=args.models_dir,
        )
        auth_key = read_auth_key(args.manifest_auth_key) if args.manifest_auth_key is not None else None
        manifest.save(args.manifest, auth_key=auth_key)
        print(
            json.dumps(
                {
                    "manifest": str(args.manifest),
                    "device_id": manifest.device_id,
                    "enrollment_images": manifest.enrollment_images,
                    "response_bits": manifest.response_bits,
                    "syndrome_bits": manifest.check_bits,
                    "graph_rank": manifest.graph_rank,
                    "identity_seed_id": manifest.identity_seed_id,
                },
                indent=2,
            )
        )
        return

    if args.manifest_auth_key is None and not args.allow_unsigned_manifest:
        raise SystemExit(
            "Provide --manifest-auth-key, or use --allow-unsigned-manifest."
        )
    auth_key = read_auth_key(args.manifest_auth_key) if args.manifest_auth_key is not None else None
    manifest = SingleShotManifest.load(
        args.manifest,
        auth_key=auth_key,
        require_auth=not args.allow_unsigned_manifest,
    )
    if manifest.payload_sha256 != payload_digest(args.payload):
        raise SystemExit("Extractor payload digest does not match the enrollment manifest.")
    if manifest.candidate_profile_sha256 != payload_digest(args.candidate_profile):
        raise SystemExit("Candidate-profile digest does not match the enrollment manifest.")
    yolo_path = args.models_dir / "yolo11n_microled_best.pt"
    stn_path = args.models_dir / "luma_spatial_head_stn.pt"
    if manifest.yolo_model_sha256 != payload_digest(yolo_path):
        raise SystemExit("YOLO model digest does not match the enrollment manifest.")
    if manifest.stn_model_sha256 != payload_digest(stn_path):
        raise SystemExit("STN model digest does not match the enrollment manifest.")
    row = quality_gated_single_image_row(
        args.input,
        args.payload,
        manifest.quality_template_corr_min,
        manifest.candidate_indices,
    )
    result = reproduce_single_shot(
        manifest,
        row,
        max_iterations=args.max_iterations,
        decoder_alpha=args.decoder_alpha,
    )
    if not args.show_root_key:
        result.pop("root_key_b64", None)
    print(json.dumps(result, indent=2))
    if not result["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
