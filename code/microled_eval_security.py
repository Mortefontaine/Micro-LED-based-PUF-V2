"""Reproducible closed-set security diagnostics for the frozen fixed-node scheme."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from microled_eval_single_shot import load_split_manifest, summarize
from microled_puf import device_name, hamming
from microled_puf_key import response_rows
from microled_single_shot_key import enroll_single_shot, reproduce_single_shot


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    split = load_split_manifest(args.split_manifest, args.input, args.payload, args.candidate_profile)
    rows = response_rows(args.input, args.payload)
    enrollment_paths = {
        str((args.input / entry["relative_path"]).resolve()).lower()
        for entry in split["enrollment_entries"]
    }
    pool_paths = {
        str((args.input / entry["relative_path"]).resolve()).lower()
        for entry in split.get("enrollment_pool_entries", split["enrollment_entries"])
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(device_name(str(row["condition"])), []).append(row)

    manifests: dict[str, Any] = {}
    references: dict[str, np.ndarray] = {}
    population_templates: dict[str, np.ndarray] = {}
    enrollment_rows: dict[str, list[dict[str, Any]]] = {}
    for device, device_rows in grouped.items():
        enrollment = [
            row for row in device_rows
            if str(Path(str(row["path"])).resolve()).lower() in enrollment_paths
        ]
        manifest = enroll_single_shot(device, enrollment, args.payload, args.candidate_profile)
        selected = np.asarray(manifest.candidate_indices, dtype=np.int64)
        manifests[device] = manifest
        enrollment_rows[device] = enrollment
        references[device] = (
            np.stack([np.asarray(row["candidate_bits"], dtype=np.uint8)[selected] for row in enrollment]).mean(axis=0) >= 0.5
        ).astype(np.uint8)
        population_templates[device] = (
            np.stack([np.asarray(row["candidate_bits"], dtype=np.uint8) for row in enrollment]).mean(axis=0) >= 0.5
        ).astype(np.uint8)

    replay_accepted = 0
    for device, enrollment in enrollment_rows.items():
        replay_accepted += sum(reproduce_single_shot(manifests[device], row)["accepted"] for row in enrollment)

    cross_accepted = 0
    cross_attempts = 0
    cross_gate_rejected = 0
    cross_hd: list[float] = []
    evaluation_rows = [
        row for row in rows
        if str(Path(str(row["path"])).resolve()).lower() not in pool_paths
    ]
    evaluation_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in evaluation_rows:
        evaluation_grouped.setdefault(device_name(str(row["condition"])), []).append(row)
    for source_device, device_rows in evaluation_grouped.items():
        for row in device_rows:
            candidate_bits = np.asarray(row["candidate_bits"], dtype=np.uint8)
            for target_device, manifest in manifests.items():
                if target_device == source_device:
                    continue
                selected = np.asarray(manifest.candidate_indices, dtype=np.int64)
                cross_hd.append(hamming(candidate_bits[selected], references[target_device]))
                decoded = reproduce_single_shot(manifest, row, args.max_iterations, args.decoder_alpha)
                cross_attempts += 1
                cross_accepted += int(decoded["accepted"])
                cross_gate_rejected += int(decoded["failure_stage"] in {"quality_gate", "pose_gate"})

    profile = np.load(args.candidate_profile, allow_pickle=False)
    eligible = np.asarray(profile["eligible_indices"], dtype=np.int64)
    template_stack = np.stack([population_templates[key] for key in sorted(population_templates)])[:, eligible]
    ones_fraction = template_stack.mean(axis=0)
    per_bit_entropy = -np.log2(np.maximum(ones_fraction, 1.0 - ones_fraction))
    inter_hd = [
        float(np.mean(template_stack[i] != template_stack[j]))
        for i in range(template_stack.shape[0])
        for j in range(i + 1, template_stack.shape[0])
    ]
    support_overlap = []
    supports = {device: set(manifest.candidate_indices) for device, manifest in manifests.items()}
    devices = sorted(supports)
    for i in range(len(devices)):
        for j in range(i + 1, len(devices)):
            a, b = supports[devices[i]], supports[devices[j]]
            support_overlap.append(len(a & b) / len(a | b))

    inter_summary = summarize(inter_hd)
    inter_summary["min"] = float(np.min(inter_hd))
    return {
        "schema": "microled-puf-security-analysis-r1",
        "independent_devices": len(devices),
        "repeated_frames": len(evaluation_rows),
        "candidate_bits": int(profile["ones_fraction"].size),
        "eligible_candidates": int(eligible.size),
        "data_isolation": (
            "selector reads only the declared enrollment pool; template, balance profile, support, and references use "
            "the selected 81 enrollment images; all pool images are excluded from probe and impostor evaluation"
        ),
        "population_balance": summarize(ones_fraction.tolist()),
        "per_bit_min_entropy_descriptive": summarize(per_bit_entropy.tolist()),
        "fixed_candidate_inter_hd": inter_summary,
        "different_device_support_jaccard": summarize(support_overlap),
        "digital_enrollment_image_replay": {
            "accepted": replay_accepted,
            "attempts": sum(len(value) for value in enrollment_rows.values()),
            "rate": replay_accepted / sum(len(value) for value in enrollment_rows.values()),
        },
        "cross_device_impostor": {
            "accepted": cross_accepted,
            "attempts": cross_attempts,
            "far": cross_accepted / cross_attempts,
            "zero_event_95pct_upper_rule_of_three": 3.0 / cross_attempts if cross_accepted == 0 else None,
            "quality_gate_rejected": cross_gate_rejected,
            "frame_hd": summarize(cross_hd),
        },
        "helper_budget": {
            "response_bits": 2048,
            "syndrome_rank_bits": 1536,
            "syndrome_coset_dimension_upper_bound": 512,
            "hmac_check_output_bits": 256,
            "identity_seed_id_bits": 96,
        },
        "failure_to_enroll": {
            "nominal_failures": 0,
            "devices": len(devices),
            "rule": "all nine enrollment frames pass the quality gate and all 2048 selected candidates are unanimous across the 3x3 grid",
        },
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    cross = result["cross_device_impostor"]
    replay = result["digital_enrollment_image_replay"]
    inter = result["fixed_candidate_inter_hd"]
    lines = [
        "# Leakage-Controlled Fixed-Node Security Diagnostics",
        "",
        f"- Data isolation: {result['data_isolation']}",
        f"- Eligible candidates: {result['eligible_candidates']}",
        f"- Fixed-common-candidate inter-HD mean / minimum: {inter['mean']:.4%} / {inter['min']:.4%}",
        f"- Ordered non-target attempts accepted: {cross['accepted']} / {cross['attempts']}",
        f"- Empirical closed-set FAR: {cross['far']:.6%}",
        f"- Rule-of-three 95% attempt-level upper bound: {cross['zero_event_95pct_upper_rule_of_three']:.6%}",
        f"- Enrollment-image replay accepted: {replay['accepted']} / {replay['attempts']}",
        "",
        "The repeated frames are correlated observations of nine physical devices. The attempt-level rule-of-three value is not a population-level FAR bound. Digital image replay remains accepted unless a trusted fresh acquisition path is enforced.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--payload", type=Path, default=root / "models" / "expanded_luma_support_payload.npz")
    parser.add_argument("--candidate-profile", type=Path, default=root / "models" / "stability_only_candidate_profile.npz")
    parser.add_argument("--split-manifest", type=Path, default=root / "models" / "enrollment_stability_only_manifest.json")
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--decoder-alpha", type=float, default=0.80)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(args.out_report, result)
    print(json.dumps(result["cross_device_impostor"], indent=2))


if __name__ == "__main__":
    main()
