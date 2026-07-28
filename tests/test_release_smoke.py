"""Fast integrity tests for the compact public release."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def test_stability_profile_has_fixed_8192_candidates() -> None:
    profile = np.load(ROOT / "models" / "stability_only_candidate_profile.npz")
    assert len(profile["eligible_indices"]) == 8192
    assert not bool(profile["cross_device_information_used"][0])


def test_frozen_manifest_covers_m1_to_m9() -> None:
    manifest = json.loads(
        (ROOT / "models" / "enrollment_stability_only_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    encoded = json.dumps(manifest)
    for device in range(1, 10):
        assert f"M{device}" in encoded


def test_compact_stn_pairs_exist() -> None:
    pair_csv = ROOT / "data" / "02_stn_pairs_M1_M6" / "alignment_pairs_M1_M6.csv"
    with pair_csv.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 108
    for row in rows:
        assert (pair_csv.parent / row["crop_relative"]).is_file()
        assert (pair_csv.parent / row["target_relative"]).is_file()


def test_headline_summary_declares_no_probe_overlap() -> None:
    path = ROOT / "results" / "single_shot" / "summary_metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = {(row["group"], row["metric"]): row["value"] for row in csv.DictReader(stream)}
    assert rows[("dataset", "independent_devices")] == "9"
    assert rows[("dataset", "enrollment_images")] == "81"
    assert rows[("dataset", "probe_images")] == "3645"
    assert float(rows[("fuzzy", "all_probe_exact_recovery")]) == 100.0


def test_closed_loop_entry_points_are_present() -> None:
    required = [
        "common.py",
        "stage_01_train_yolo.py",
        "stage_02_train_stn.py",
        "stage_03_align_raw.py",
        "stage_04_puf_fuzzy.py",
        "run_all_m1_m6.py",
        "README.md",
    ]
    for name in required:
        assert (ROOT / "pipeline" / name).is_file()


def test_executable_sources_do_not_reference_parent_project_paths() -> None:
    forbidden = ("c:\\users", "onedrive", "playground", "data/samples", "data\\samples")
    for folder in ("code", "training", "pipeline"):
        for path in (ROOT / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            assert not any(token in text for token in forbidden), path


def test_cleanroom_record_matches_expected_sample_scope() -> None:
    summary = json.loads(
        (
            ROOT
            / "validation"
            / "cleanroom_20260728"
            / "stage4_summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["devices"] == 6
    assert summary["enrollment_images"] == 54
    assert summary["independent_probes"] == 54
    assert summary["accepted_probes"] == 53
    assert summary["response_bits"] == 2048
    assert summary["inter_device_information_used_for_selection"] is False
