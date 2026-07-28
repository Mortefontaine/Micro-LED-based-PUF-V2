"""Held-out evaluation of single-image micro-LED PUF key regeneration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from microled_puf import device_name, hamming, natural_key
from microled_puf_key import payload_digest, response_rows
from microled_single_shot_key import (
    enroll_single_shot,
    reproduce_single_shot,
)


def summarize(values: list[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "n": int(data.size),
        "mean": float(data.mean()) if data.size else float("nan"),
        "std": float(data.std()) if data.size else float("nan"),
        "p95": float(np.percentile(data, 95)) if data.size else float("nan"),
        "p99": float(np.percentile(data, 99)) if data.size else float("nan"),
        "max": float(data.max()) if data.size else float("nan"),
    }


def load_split_manifest(path: Path, input_root: Path, payload: Path, profile: Path) -> dict[str, Any]:
    split = json.loads(path.read_text(encoding="utf-8"))
    if split.get("schema") != "microled-puf-enrollment-split-r1":
        raise ValueError("Unsupported or missing enrollment split manifest schema.")
    if split.get("payload_sha256") != payload_digest(payload):
        raise ValueError("Evaluation payload does not match the frozen enrollment split manifest.")
    if split.get("candidate_profile_sha256") != payload_digest(profile):
        raise ValueError("Evaluation candidate profile does not match the frozen enrollment split manifest.")
    entries = split.get("enrollment_entries")
    if not isinstance(entries, list) or len(entries) != 81:
        raise ValueError("Final fixed-nine evaluation requires exactly 81 declared enrollment entries.")
    pool_entries = split.get("enrollment_pool_entries", entries)
    if not isinstance(pool_entries, list) or len(pool_entries) < len(entries):
        raise ValueError("Enrollment pool must contain every selected enrollment entry.")
    root = input_root.resolve()
    def validate_entries(values: list[dict[str, Any]], label: str) -> set[str]:
        seen: set[str] = set()
        for entry in values:
            relative = Path(*PurePosixPath(str(entry["relative_path"])).parts)
            candidate = (input_root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"{label} path escapes input root: {relative}") from exc
            if not candidate.is_file():
                raise FileNotFoundError(f"Declared {label} image is missing: {candidate}")
            digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError(f"Declared {label} image digest mismatch: {candidate}")
            key = str(candidate).lower()
            if key in seen:
                raise ValueError(f"Duplicate {label} image in split manifest: {candidate}")
            seen.add(key)
        return seen

    selected_paths = validate_entries(entries, "enrollment")
    pool_paths = validate_entries(pool_entries, "enrollment-pool")
    if not selected_paths.issubset(pool_paths):
        raise ValueError("Every selected enrollment image must belong to the declared enrollment pool.")
    return split


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    split = load_split_manifest(args.split_manifest, args.input, args.payload, args.candidate_profile)
    enrollment_paths = {
        str((args.input / str(entry["relative_path"])).resolve()).lower()
        for entry in split["enrollment_entries"]
    }
    pool_entries = split.get("enrollment_pool_entries", split["enrollment_entries"])
    probe_exclusion_paths = {
        str((args.input / str(entry["relative_path"])).resolve()).lower()
        for entry in pool_entries
    }
    rows = response_rows(args.input, args.payload)
    orientation_lookup: dict[tuple[str, str], float] = {}
    if args.orientation_metrics is not None:
        for metric in csv.DictReader(args.orientation_metrics.open(encoding="utf-8-sig")):
            relative = PurePosixPath(str(metric["relative_path"]))
            orientation_lookup[(relative.parent.name, relative.name)] = float(metric["orientation_margin"])
    gate_stages = {"quality_gate", "pose_gate", "orientation_gate"}
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[device_name(str(row["condition"]))][str(row["condition"])].append(row)

    result: dict[str, Any] = {
        "settings": {key: str(value) for key, value in vars(args).items()},
        "data_isolation": {
            "enrollment_pool_images": len(pool_entries),
            "selected_enrollment_images": len(split["enrollment_entries"]),
            "probe_exclusion_rule": "all enrollment-pool images are excluded from probes",
        },
        "devices": {},
    }
    all_success: list[bool] = []
    all_hd: list[float] = []
    all_qualified_hd: list[float] = []
    all_iterations: list[float] = []
    failures: list[dict[str, Any]] = []
    probe_records: list[dict[str, Any]] = []
    population_templates: dict[str, np.ndarray] = {}
    gate_rejected = 0
    decoder_failed = 0
    outcomes_by_condition: dict[str, list[bool]] = defaultdict(list)
    hd_bins = [(0.00, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]
    bin_results: dict[str, list[bool]] = {f"{low:.2f}-{high:.2f}": [] for low, high in hd_bins}

    for device, conditions in sorted(grouped.items(), key=lambda item: natural_key(item[0])):
        device_rows = [row for values in conditions.values() for row in values]
        enrollment = [row for row in device_rows if str(Path(str(row["path"])).resolve()).lower() in enrollment_paths]
        probes = [
            row
            for row in device_rows
            if str(Path(str(row["path"])).resolve()).lower() not in probe_exclusion_paths
        ]
        if len(enrollment) != 9:
            raise ValueError(f"{device} has {len(enrollment)} declared enrollment images; expected 9.")
        manifest = enroll_single_shot(device, enrollment, args.payload, args.candidate_profile)
        selected = np.asarray(manifest.candidate_indices, dtype=np.int64)
        reference_stack = np.stack([np.asarray(row["candidate_bits"], dtype=np.uint8)[selected] for row in enrollment])
        reference = (reference_stack.mean(axis=0) >= 0.5).astype(np.uint8)
        population_templates[device] = (
            np.stack([np.asarray(row["candidate_bits"], dtype=np.uint8) for row in enrollment]).mean(axis=0) >= 0.5
        ).astype(np.uint8)
        successes: list[bool] = []
        hds: list[float] = []
        probe_uniformities: list[float] = []
        qualified_hds: list[float] = []
        iterations: list[float] = []
        device_gate_rejected = 0
        device_decoder_failed = 0
        for row in probes:
            observed = np.asarray(row["candidate_bits"], dtype=np.uint8)[selected]
            observed_uniformity = float(observed.mean())
            hd = hamming(observed, reference)
            decoded = reproduce_single_shot(
                manifest,
                row,
                max_iterations=args.max_iterations,
                decoder_alpha=args.decoder_alpha,
            )
            orientation_margin = orientation_lookup.get((str(row["condition"]), Path(str(row["path"])).name))
            orientation_rejected = (
                args.orientation_metrics is not None
                and (orientation_margin is None or orientation_margin < args.orientation_margin_min)
            )
            success = bool(decoded["accepted"]) and not orientation_rejected
            failure_stage = "orientation_gate" if orientation_rejected else decoded.get("failure_stage")
            if failure_stage in gate_stages:
                gate_rejected += 1
                device_gate_rejected += 1
            elif not success:
                decoder_failed += 1
                device_decoder_failed += 1
            if failure_stage not in gate_stages:
                qualified_hds.append(hd)
                all_qualified_hd.append(hd)
            successes.append(success)
            outcomes_by_condition[str(row["condition"])].append(success)
            hds.append(hd)
            probe_uniformities.append(observed_uniformity)
            iterations.append(float(decoded["iterations"]))
            all_success.append(success)
            all_hd.append(hd)
            all_iterations.append(float(decoded["iterations"]))
            probe_records.append(
                {
                    "device": device,
                    "condition": str(row["condition"]),
                    "frame": Path(str(row["path"])).name,
                    "quality_template_corr": float(decoded["quality_template_corr"]),
                    "quality_gate_passed": failure_stage not in gate_stages,
                    "orientation_margin": orientation_margin,
                    "orientation_gate_passed": not orientation_rejected,
                    "pose_initial_corr": row.get("pose_initial_corr"),
                    "pose_final_corr": row.get("pose_final_corr"),
                    "pose_refined": bool(row.get("pose_refined", False)),
                    "pose_attempted": bool(row.get("pose_attempted", False)),
                    "pose_gate_passed": bool(row.get("pose_gate_passed", True)),
                    "pose_angle_deg": float(row.get("pose_angle_deg", 0.0)),
                    "pose_scale": float(row.get("pose_scale", 1.0)),
                    "pose_tx32": float(row.get("pose_tx32", 0.0)),
                    "pose_ty32": float(row.get("pose_ty32", 0.0)),
                    "raw_mismatch_bits": int(round(hd * manifest.response_bits)),
                    "raw_hd": hd,
                    "observed_ones": int(observed.sum()),
                    "observed_uniformity_percent": 100.0 * observed_uniformity,
                    "decoder_attempted": bool(decoded["decoder_attempted"]),
                    "decoder_converged": bool(decoded["decoder_converged"]),
                    "decoder_iterations": int(decoded["iterations"]),
                    "unsatisfied_checks": decoded["unsatisfied_checks"],
                    "estimated_error_bits": decoded["estimated_error_weight"],
                    "exact_reference_recovered": success,
                    "accepted": success,
                    "outcome_stage": "accepted" if success else str(failure_stage),
                }
            )
            if not success:
                failures.append(
                    {
                        "device": device,
                        "condition": str(row["condition"]),
                        "path": str(row["path"]),
                        "raw_hd": hd,
                        "decoder_converged": bool(decoded["decoder_converged"]),
                        "iterations": int(decoded["iterations"]),
                        "unsatisfied_checks": (
                            int(decoded["unsatisfied_checks"]) if decoded["unsatisfied_checks"] is not None else None
                        ),
                        "median_margin_ratio": (
                            float(decoded["median_margin_ratio"]) if decoded["median_margin_ratio"] is not None else None
                        ),
                        "failure_stage": str(failure_stage),
                        "quality_template_corr": float(decoded["quality_template_corr"]),
                    }
                )
            for low, high in hd_bins:
                if low <= hd < high:
                    bin_results[f"{low:.2f}-{high:.2f}"].append(success)
                    break
        result["devices"][device] = {
            "attempts": len(successes),
            "accepted": int(sum(successes)),
            "success_rate": float(np.mean(successes)),
            "quality_gate_rejected": device_gate_rejected,
            "decoder_failed_after_gate": device_decoder_failed,
            "qualified_success_rate": (
                float(sum(successes) / (len(successes) - device_gate_rejected))
                if len(successes) > device_gate_rejected
                else float("nan")
            ),
            "reference_uniformity": float(reference.mean()),
            "probe_uniformity": summarize(probe_uniformities),
            "raw_hd": summarize(qualified_hds),
            "raw_hd_all_probes": summarize(hds),
            "decoder_iterations": summarize(iterations),
        }

    immediate_retry_eligible = 0
    immediate_retry_success = 0
    for outcomes in outcomes_by_condition.values():
        for index, success in enumerate(outcomes):
            if success or index + 1 >= len(outcomes):
                continue
            immediate_retry_eligible += 1
            immediate_retry_success += int(outcomes[index + 1])

    qualified_attempts = len(all_success) - gate_rejected
    result["overall"] = {
        "attempts": len(all_success),
        "accepted": int(sum(all_success)),
        "success_rate": float(np.mean(all_success)),
        "failed_closed": len(all_success) - int(sum(all_success)),
        "quality_gate_rejected": gate_rejected,
        "quality_gate_reject_rate": gate_rejected / len(all_success),
        "decoder_failed_after_gate": decoder_failed,
        "qualified_attempts": qualified_attempts,
        "qualified_success_rate": int(sum(all_success)) / qualified_attempts,
        "immediate_retry_eligible": immediate_retry_eligible,
        "immediate_retry_success": immediate_retry_success,
        "immediate_retry_success_rate": (
            immediate_retry_success / immediate_retry_eligible if immediate_retry_eligible else None
        ),
        "raw_hd": summarize(all_qualified_hd),
        "raw_hd_all_probes": summarize(all_hd),
        "decoder_iterations": summarize(all_iterations),
        "success_by_raw_hd": {
            key: {
                "attempts": len(values),
                "accepted": int(sum(values)),
                "success_rate": float(np.mean(values)) if values else None,
            }
            for key, values in bin_results.items()
        },
    }
    result["failures"] = failures
    if args.out_probes_csv is not None:
        args.out_probes_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_probes_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(probe_records[0]))
            writer.writeheader()
            writer.writerows(probe_records)
    if args.out_device_csv is not None:
        args.out_device_csv.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "device", "enrollment_images", "probe_images", "qualified_probe_images", "reference_uniformity_percent",
            "probe_uniformity_mean_percent", "probe_uniformity_std_percent",
            "intra_hd_mean_percent", "reliability_percent", "exact_key_success_percent",
        ]
        with args.out_device_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for device, values in result["devices"].items():
                writer.writerow(
                    {
                        "device": device,
                        "enrollment_images": 9,
                        "probe_images": values["attempts"],
                        "qualified_probe_images": values["raw_hd"]["n"],
                        "reference_uniformity_percent": 100.0 * values["reference_uniformity"],
                        "probe_uniformity_mean_percent": 100.0 * values["probe_uniformity"]["mean"],
                        "probe_uniformity_std_percent": 100.0 * values["probe_uniformity"]["std"],
                        "intra_hd_mean_percent": 100.0 * values["raw_hd"]["mean"],
                        "reliability_percent": 100.0 * (1.0 - values["raw_hd"]["mean"]),
                        "exact_key_success_percent": 100.0 * values["success_rate"],
                    }
                )
    if args.out_summary_csv is not None:
        profile = np.load(args.candidate_profile, allow_pickle=False)
        eligible = np.asarray(profile["eligible_indices"], dtype=np.int64)
        template_stack = np.stack([population_templates[key] for key in sorted(population_templates)])[:, eligible]
        inter_hd = [
            float(np.mean(template_stack[i] != template_stack[j]))
            for i in range(template_stack.shape[0])
            for j in range(i + 1, template_stack.shape[0])
        ]
        bit_alias = template_stack.mean(axis=0)
        uniformities = [value["reference_uniformity"] for value in result["devices"].values()]
        metrics = [
            ("dataset", "independent_devices", 9, "devices"),
            ("dataset", "enrollment_images", 81, "images"),
            ("dataset", "enrollment_pool_images", len(pool_entries), "images excluded from probes"),
            ("dataset", "probe_images", result["overall"]["attempts"], "images"),
            ("PUF", "uniformity_mean", 100.0 * float(np.mean(uniformities)), "% ones"),
            ("PUF", "uniformity_std", 100.0 * float(np.std(uniformities)), "%"),
            ("PUF", "uniqueness_fixed_common_candidates_mean", 100.0 * float(np.mean(inter_hd)), "% inter-HD"),
            ("PUF", "uniqueness_fixed_common_candidates_min", 100.0 * float(np.min(inter_hd)), "% inter-HD"),
            ("PUF", "reliability", 100.0 * (1.0 - result["overall"]["raw_hd"]["mean"]), "%"),
            ("PUF", "intra_hd_mean", 100.0 * result["overall"]["raw_hd"]["mean"], "%"),
            ("PUF", "intra_hd_p95", 100.0 * result["overall"]["raw_hd"]["p95"], "%"),
            ("PUF", "intra_hd_p99", 100.0 * result["overall"]["raw_hd"]["p99"], "%"),
            ("PUF", "bit_aliasing_mean", 100.0 * float(bit_alias.mean()), "% devices with bit 1"),
            ("fuzzy", "all_probe_exact_recovery", 100.0 * result["overall"]["success_rate"], "%"),
            ("fuzzy", "quality_gate_rejects", result["overall"]["quality_gate_rejected"], "images"),
            ("fuzzy", "qualified_exact_recovery", 100.0 * result["overall"]["qualified_success_rate"], "%"),
            ("security", "syndrome_rank_leakage", 1536, "bits"),
            ("security", "coset_dimension_upper_bound", 512, "bits"),
        ]
        args.out_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_summary_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["group", "metric", "value", "unit"])
            writer.writerows(metrics)
    return result


def write_report(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    overall = result["overall"]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Single-Image PUF Key-Regeneration Evaluation\n\n")
        isolation = result["data_isolation"]
        f.write(
            f"Enrollment predeclares {isolation['enrollment_pool_images']} pool images and selects "
            f"{isolation['selected_enrollment_images']} images (nine per device). Every pool image is excluded from "
            "testing; all remaining aligned images are one-shot key-regeneration attempts. The enrolled response and "
            "root key are not stored.\n\n"
        )
        f.write("## Overall\n\n")
        f.write(f"- Attempts: {overall['attempts']}\n")
        f.write(f"- Exact key reconstructions: {overall['accepted']} ({overall['success_rate']:.2%})\n")
        f.write(f"- Failed closed: {overall['failed_closed']}\n")
        f.write(f"- Pre-bit quality-gate rejections: {overall['quality_gate_rejected']} ({overall['quality_gate_reject_rate']:.2%})\n")
        f.write(f"- Decoder failures after quality gate: {overall['decoder_failed_after_gate']}\n")
        f.write(f"- Qualified-image exact key rate: {overall['qualified_success_rate']:.3%}\n")
        retry_rate = overall["immediate_retry_success_rate"]
        retry_text = f"{retry_rate:.2%}" if retry_rate is not None else "-"
        f.write(f"- Immediate next-frame recovery after a failed attempt: {overall['immediate_retry_success']} / {overall['immediate_retry_eligible']} ({retry_text})\n")
        f.write(f"- Qualified intra-HD samples (quality gate passed): {overall['raw_hd']['n']}\n")
        f.write(f"- Qualified intra-HD mean / p95 / p99 / max: {overall['raw_hd']['mean']:.4f} / {overall['raw_hd']['p95']:.4f} / {overall['raw_hd']['p99']:.4f} / {overall['raw_hd']['max']:.4f}\n")
        f.write(f"- All-probe raw-HD audit samples (includes quality rejects): {overall['raw_hd_all_probes']['n']}\n\n")
        f.write("## Per Device\n\n")
        f.write("| Device | Accepted | Attempts | First-shot success | Gate reject | Qualified success | Raw HD p99 |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for device, values in result["devices"].items():
            f.write(f"| {device} | {values['accepted']} | {values['attempts']} | {values['success_rate']:.2%} | {values['quality_gate_rejected']} | {values['qualified_success_rate']:.3%} | {values['raw_hd']['p99']:.4f} |\n")
        f.write("\n## Success by Raw HD\n\n")
        f.write("| Raw HD interval | Accepted | Attempts | Success |\n|---|---:|---:|---:|\n")
        for interval, values in overall["success_by_raw_hd"].items():
            rate = f"{values['success_rate']:.2%}" if values["success_rate"] is not None else "-"
            f.write(f"| {interval} | {values['accepted']} | {values['attempts']} | {rate} |\n")
        f.write("\n## Failed-Closed Frames\n\n")
        f.write("| Device | Condition | Frame | Stage | Template corr | Raw HD |\n|---|---|---|---|---:|---:|\n")
        for failure in result["failures"]:
            f.write(f"| {failure['device']} | {failure['condition']} | {Path(failure['path']).name} | {failure['failure_stage']} | {failure['quality_template_corr']:.4f} | {failure['raw_hd']:.4f} |\n")
        f.write("\n## Interpretation\n\n")
        f.write("An accepted attempt reproduced the enrolled key exactly and passed the HMAC check. A decoder failure or wrong candidate key is rejected; the implementation never releases a different key. This is a research decoder result, not a production cryptographic certification or a 128-bit min-entropy proof.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--payload", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "expanded_luma_support_payload.npz")
    parser.add_argument("--candidate-profile", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "stability_only_candidate_profile.npz")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models" / "enrollment_stability_only_manifest.json",
    )
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--decoder-alpha", type=float, default=0.80)
    parser.add_argument("--orientation-metrics", type=Path, default=None)
    parser.add_argument("--orientation-margin-min", type=float, default=0.0)
    parser.add_argument("--out-json", type=Path, default=Path("single_shot_key_eval.json"))
    parser.add_argument("--out-report", type=Path, default=Path("single_shot_key_eval.md"))
    parser.add_argument("--out-probes-csv", type=Path, default=None)
    parser.add_argument("--out-device-csv", type=Path, default=None)
    parser.add_argument("--out-summary-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_report(args.out_report, result)
    print(f"report={args.out_report} json={args.out_json}")


if __name__ == "__main__":
    main()
