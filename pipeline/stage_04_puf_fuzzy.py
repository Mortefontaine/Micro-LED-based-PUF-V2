"""Enroll M1-M6 and evaluate independent sample probes with the fuzzy extractor."""

from __future__ import annotations

import argparse
import csv
import json
import secrets
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    prepare_stage_dir,
    read_artifact,
    write_manifest,
)

sys.path.insert(0, str(REPO_ROOT / "code"))
from microled_fuzzy_extractor import (  # noqa: E402
    SingleShotManifest,
    enroll_single_shot,
    enrollment_response_rows,
    payload_digest,
    quality_gated_single_image_row,
    reproduce_single_shot,
    select_per_condition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--alignment-manifest", type=Path, default=None)
    parser.add_argument("--detector-manifest", type=Path, default=None)
    parser.add_argument("--stn-manifest", type=Path, default=None)
    parser.add_argument("--quality-corr-min", type=float, default=0.4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    alignment_manifest = args.alignment_manifest or args.output_root / "03_alignment" / "stage_manifest.json"
    detector_manifest = args.detector_manifest or args.output_root / "01_yolo" / "stage_manifest.json"
    stn_manifest = args.stn_manifest or args.output_root / "02_stn" / "stage_manifest.json"
    aligned_root = read_artifact(alignment_manifest, "aligned_images")
    detector = read_artifact(detector_manifest, "detector_model")
    stn = read_artifact(stn_manifest, "stn_model")
    payload = REPO_ROOT / "models" / "expanded_luma_support_payload.npz"
    profile = REPO_ROOT / "models" / "stability_only_candidate_profile.npz"
    stage_dir = args.output_root / "04_puf_fuzzy"
    prepare_stage_dir(stage_dir, args.output_root, args.force)
    models_dir = stage_dir / "bound_models"
    models_dir.mkdir()
    shutil.copy2(detector, models_dir / "yolo11n_microled_best.pt")
    shutil.copy2(stn, models_dir / "luma_spatial_head_stn.pt")
    auth_key = secrets.token_bytes(32)
    auth_key_path = stage_dir / "local_manifest_auth.key"
    auth_key_path.write_bytes(auth_key)
    manifests_dir = stage_dir / "device_manifests"
    manifests_dir.mkdir()

    probe_rows: list[dict[str, Any]] = []
    device_rows: list[dict[str, Any]] = []
    total_enrollment = 0
    for device in [f"M{index}" for index in range(1, 7)]:
        all_rows = enrollment_response_rows(aligned_root, payload, device)
        selected = select_per_condition(all_rows, 1, args.quality_corr_min)
        selected_paths = {str(Path(row["path"]).resolve()) for row in selected}
        manifest = enroll_single_shot(
            device,
            selected,
            payload,
            profile,
            quality_template_corr_min=args.quality_corr_min,
            models_dir=models_dir,
            require_unanimous=True,
        )
        manifest_path = manifests_dir / f"{device}.json"
        manifest.save(manifest_path, auth_key=auth_key)
        loaded = SingleShotManifest.load(manifest_path, auth_key=auth_key, require_auth=True)
        if loaded.payload_sha256 != payload_digest(payload):
            raise RuntimeError(f"{device} manifest payload digest mismatch")
        probes = [row for row in all_rows if str(Path(row["path"]).resolve()) not in selected_paths]
        first_pass_accepted = 0
        accepted_after_retry = 0
        retry_attempts = 0
        rescued = 0
        for row in probes:
            result = reproduce_single_shot(loaded, row)
            first_success = bool(result["accepted"])
            retry_result = None
            retry_row = None
            if not first_success:
                retry_attempts += 1
                retry_row = quality_gated_single_image_row(
                    Path(str(row["path"])),
                    payload,
                    loaded.quality_template_corr_min,
                    loaded.candidate_indices,
                    pose_mode="forced",
                )
                retry_result = reproduce_single_shot(loaded, retry_row)
            retry_success = bool(retry_result["accepted"]) if retry_result is not None else False
            final_success = first_success or retry_success
            final_result = retry_result if retry_result is not None else result
            first_pass_accepted += int(first_success)
            accepted_after_retry += int(final_success)
            rescued += int(retry_success)
            probe_rows.append(
                {
                    "device": device,
                    "condition": row["condition"],
                    "image": Path(row["path"]).relative_to(aligned_root).as_posix(),
                    "first_pass_accepted": int(first_success),
                    "pose_retry_attempted": int(retry_result is not None),
                    "pose_retry_refined": int(bool(retry_row.get("pose_refined", False))) if retry_row is not None else 0,
                    "pose_retry_accepted": int(retry_success),
                    "accepted_after_optional_retry": int(final_success),
                    "failure_stage": result["failure_stage"] or "",
                    "pose_retry_failure_stage": (
                        "" if retry_result is None or retry_result["failure_stage"] is None
                        else retry_result["failure_stage"]
                    ),
                    "decoder_converged": int(bool(result["decoder_converged"])),
                    "estimated_error_weight": (
                        "" if result["estimated_error_weight"] is None else result["estimated_error_weight"]
                    ),
                    "quality_template_corr": result["quality_template_corr"],
                    "key_id": final_result["key_id"] or "",
                }
            )
        total_enrollment += len(selected)
        device_rows.append(
            {
                "device": device,
                "enrollment_images": len(selected),
                "independent_probes": len(probes),
                "first_pass_accepted": first_pass_accepted,
                "first_pass_acceptance_rate_percent": 100.0 * first_pass_accepted / len(probes) if probes else 0.0,
                "pose_retry_attempts": retry_attempts,
                "rescued_by_pose_retry": rescued,
                "accepted_after_optional_retry": accepted_after_retry,
                "acceptance_rate_after_optional_retry_percent": 100.0 * accepted_after_retry / len(probes) if probes else 0.0,
                "identity_seed_id": manifest.identity_seed_id,
            }
        )

    write_csv(stage_dir / "per_probe_results.csv", probe_rows)
    write_csv(stage_dir / "per_device_results.csv", device_rows)
    first_pass_total = sum(int(row["first_pass_accepted"]) for row in probe_rows)
    retry_attempts_total = sum(int(row["pose_retry_attempted"]) for row in probe_rows)
    rescued_total = sum(int(row["pose_retry_accepted"]) for row in probe_rows)
    final_total = sum(int(row["accepted_after_optional_retry"]) for row in probe_rows)
    summary = {
        "devices": 6,
        "enrollment_images": total_enrollment,
        "independent_probes": len(probe_rows),
        "first_pass_accepted_probes": first_pass_total,
        "first_pass_acceptance_rate_percent": 100.0 * first_pass_total / len(probe_rows),
        "pose_retry_attempts": retry_attempts_total,
        "rescued_by_pose_retry": rescued_total,
        "accepted_after_optional_retry": final_total,
        "acceptance_rate_after_optional_retry_percent": 100.0 * final_total / len(probe_rows),
        "primary_metrics_include_pose_retry": False,
        "response_bits": 2048,
        "root_key_bits": 256,
        "identity_seed_bits": 256,
        "quality_corr_min": args.quality_corr_min,
        "selection_rule": "within_device_stability_only",
        "sample_selection_requires_unanimous_enrollment_bits": True,
        "inter_device_information_used_for_selection": False,
    }
    summary_path = stage_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    manifest = write_manifest(
        stage_dir,
        "04_puf_fuzzy",
        {
            "summary": summary_path,
            "per_probe_results": stage_dir / "per_probe_results.csv",
            "per_device_results": stage_dir / "per_device_results.csv",
            "device_manifests": manifests_dir,
            "manifest_auth_key": auth_key_path,
            "bound_models": models_dir,
        },
        {
            "alignment_manifest": alignment_manifest,
            "detector_manifest": detector_manifest,
            "stn_manifest": stn_manifest,
            "extractor_payload": payload,
            "candidate_profile": profile,
        },
        summary,
    )
    print(json.dumps(summary, indent=2))
    print(f"Stage manifest: {manifest}")


if __name__ == "__main__":
    main()
