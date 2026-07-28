"""Export closed-set genuine/impostor scores, ROC/AUC and empirical EER.

The verification score is the raw Hamming distance evaluated on the claimed
device's frozen 2,048 positions.  Enrollment-pool images are excluded.  These
curves are descriptive image-attempt results for the nine evaluated devices;
repeated frames are not independent physical-device samples.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from microled_eval_single_shot import load_split_manifest
from microled_puf import device_name, hamming, natural_key
from microled_puf_key import response_rows
from microled_single_shot_key import enroll_single_shot


def summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def collect_scores(args: argparse.Namespace) -> list[dict[str, Any]]:
    split = load_split_manifest(
        args.split_manifest,
        args.input,
        args.payload,
        args.candidate_profile,
    )
    rows = response_rows(args.input, args.payload)
    enrollment_paths = {
        str((args.input / entry["relative_path"]).resolve()).lower()
        for entry in split["enrollment_entries"]
    }
    pool_paths = {
        str((args.input / entry["relative_path"]).resolve()).lower()
        for entry in split.get(
            "enrollment_pool_entries",
            split["enrollment_entries"],
        )
    }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(device_name(str(row["condition"])), []).append(row)

    manifests: dict[str, Any] = {}
    references: dict[str, np.ndarray] = {}
    for device, device_rows in grouped.items():
        enrollment = [
            row
            for row in device_rows
            if str(Path(str(row["path"])).resolve()).lower() in enrollment_paths
        ]
        manifest = enroll_single_shot(
            device,
            enrollment,
            args.payload,
            args.candidate_profile,
        )
        selected = np.asarray(manifest.candidate_indices, dtype=np.int64)
        references[device] = (
            np.stack(
                [
                    np.asarray(row["candidate_bits"], dtype=np.uint8)[selected]
                    for row in enrollment
                ]
            ).mean(axis=0)
            >= 0.5
        ).astype(np.uint8)
        manifests[device] = manifest

    output: list[dict[str, Any]] = []
    trial_id = 0
    devices = sorted(manifests, key=natural_key)
    for row in rows:
        path = Path(str(row["path"]))
        if str(path.resolve()).lower() in pool_paths:
            continue
        source = device_name(str(row["condition"]))
        candidate_bits = np.asarray(row["candidate_bits"], dtype=np.uint8)
        for target in devices:
            manifest = manifests[target]
            selected = np.asarray(manifest.candidate_indices, dtype=np.int64)
            hd = hamming(candidate_bits[selected], references[target])
            trial_id += 1
            output.append(
                {
                    "trial_id": trial_id,
                    "trial_type": "genuine" if source == target else "impostor",
                    "source_device": source,
                    "claimed_device": target,
                    "condition": str(row["condition"]),
                    "frame": path.name,
                    "mismatch_bits_of_2048": int(round(2048 * hd)),
                    "hd_fraction": float(hd),
                    "hd_percent": float(100.0 * hd),
                    "similarity_score": float(1.0 - hd),
                }
            )
    expected = 3645 * len(devices)
    if len(output) != expected:
        raise RuntimeError(f"Expected {expected} scored claims, obtained {len(output)}.")
    return output


def operating_curve(
    genuine: np.ndarray,
    impostor: np.ndarray,
) -> tuple[list[dict[str, float]], dict[str, Any]]:
    unique = np.unique(np.concatenate([genuine, impostor]))
    thresholds = np.concatenate(
        [
            np.asarray([-np.finfo(float).eps]),
            unique,
            np.asarray([1.0 + np.finfo(float).eps]),
        ]
    )
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        far = float(np.mean(impostor <= threshold))
        frr = float(np.mean(genuine > threshold))
        rows.append(
            {
                "threshold_hd_fraction": float(threshold),
                "threshold_hd_percent": float(100.0 * threshold),
                "far": far,
                "frr": frr,
                "tpr": 1.0 - frr,
                "tnr": 1.0 - far,
            }
        )

    fpr = np.asarray([row["far"] for row in rows], dtype=float)
    tpr = np.asarray([row["tpr"] for row in rows], dtype=float)
    order = np.argsort(fpr, kind="stable")
    auc = float(np.trapezoid(tpr[order], fpr[order]))

    max_genuine = float(genuine.max())
    min_impostor = float(impostor.min())
    separated = bool(max_genuine < min_impostor)
    if separated:
        eer = 0.0
        eer_threshold = 0.5 * (max_genuine + min_impostor)
    else:
        difference = np.asarray(
            [row["far"] - row["frr"] for row in rows],
            dtype=float,
        )
        crossing = np.flatnonzero(difference[:-1] * difference[1:] <= 0)
        if crossing.size:
            left = int(crossing[0])
            right = left + 1
            d0, d1 = difference[left], difference[right]
            weight = 0.0 if d0 == d1 else float(-d0 / (d1 - d0))
            far = rows[left]["far"] + weight * (
                rows[right]["far"] - rows[left]["far"]
            )
            frr = rows[left]["frr"] + weight * (
                rows[right]["frr"] - rows[left]["frr"]
            )
            eer = float(0.5 * (far + frr))
            eer_threshold = float(
                rows[left]["threshold_hd_fraction"]
                + weight
                * (
                    rows[right]["threshold_hd_fraction"]
                    - rows[left]["threshold_hd_fraction"]
                )
            )
        else:
            index = int(np.argmin(np.abs(difference)))
            eer = float(0.5 * (rows[index]["far"] + rows[index]["frr"]))
            eer_threshold = float(rows[index]["threshold_hd_fraction"])

    metrics = {
        "auc": auc,
        "eer": eer,
        "eer_threshold_hd_fraction": eer_threshold,
        "eer_threshold_hd_percent": 100.0 * eer_threshold,
        "perfect_empirical_separation": separated,
        "zero_error_threshold_interval_hd_fraction": (
            [max_genuine, min_impostor] if separated else None
        ),
        "zero_error_threshold_interval_hd_percent": (
            [100.0 * max_genuine, 100.0 * min_impostor]
            if separated
            else None
        ),
        "maximum_genuine_hd_fraction": max_genuine,
        "minimum_impostor_hd_fraction": min_impostor,
        "score_gap_hd_fraction": min_impostor - max_genuine,
    }
    return rows, metrics


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_distribution(
    out_dir: Path,
    genuine: np.ndarray,
    impostor: np.ndarray,
    metrics: dict[str, Any],
) -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig, ax = plt.subplots(figsize=(5.1, 3.6), dpi=200)
    bins = np.linspace(0.0, 0.85, 86)
    ax.hist(
        100.0 * impostor,
        bins=100.0 * bins,
        density=True,
        color="#9bbbd5",
        edgecolor="none",
        alpha=0.72,
        label=f"Impostor (n={impostor.size:,})",
    )
    ax.hist(
        100.0 * genuine,
        bins=100.0 * bins,
        density=True,
        color="#d95f59",
        edgecolor="none",
        alpha=0.82,
        label=f"Genuine (n={genuine.size:,})",
    )
    threshold = 100.0 * float(metrics["eer_threshold_hd_fraction"])
    ax.axvline(threshold, color="#303030", lw=1.1, ls=(0, (4, 3)))
    ax.text(
        threshold,
        ax.get_ylim()[1] * 0.96,
        f"threshold {threshold:.2f}%",
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.set_xlabel("Raw Hamming distance (%)")
    ax.set_ylabel("Probability density")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "verification_score_distribution.png", dpi=300)
    fig.savefig(out_dir / "verification_score_distribution.svg")
    plt.close(fig)


def plot_roc(
    out_dir: Path,
    curve: list[dict[str, float]],
    metrics: dict[str, Any],
) -> None:
    fpr = np.asarray([row["far"] for row in curve])
    tpr = np.asarray([row["tpr"] for row in curve])
    order = np.argsort(fpr, kind="stable")
    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig, ax = plt.subplots(figsize=(4.2, 4.0), dpi=200)
    ax.plot(fpr[order], tpr[order], color="#245b8a", lw=1.8)
    ax.plot([0, 1], [0, 1], color="#a0a0a0", lw=0.8, ls="--")
    ax.text(
        0.97,
        0.05,
        f"AUC = {metrics['auc']:.6f}\nEER = {100 * metrics['eer']:.4f}%",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )
    ax.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01))
    ax.set_xlabel("False-positive rate (FAR)")
    ax.set_ylabel("True-positive rate (1-FRR)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "roc_auc_curve.png", dpi=300)
    fig.savefig(out_dir / "roc_auc_curve.svg")
    plt.close(fig)


def plot_far_frr(
    out_dir: Path,
    curve: list[dict[str, float]],
    metrics: dict[str, Any],
    genuine_n: int,
    impostor_n: int,
) -> None:
    threshold = np.asarray([row["threshold_hd_percent"] for row in curve])
    far = np.asarray([row["far"] for row in curve])
    frr = np.asarray([row["frr"] for row in curve])
    floor = min(0.5 / genuine_n, 0.5 / impostor_n)
    plt.rcParams.update({"font.family": "Arial", "font.size": 9})
    fig, ax = plt.subplots(figsize=(5.1, 3.6), dpi=200)
    ax.semilogy(
        threshold,
        np.maximum(far, floor),
        color="#245b8a",
        lw=1.6,
        label="FAR",
    )
    ax.semilogy(
        threshold,
        np.maximum(frr, floor),
        color="#d95f59",
        lw=1.6,
        label="FRR",
    )
    eer_threshold = float(metrics["eer_threshold_hd_percent"])
    ax.axvline(eer_threshold, color="#303030", lw=1.0, ls=(0, (4, 3)))
    ax.set_xlim(0.0, 85.0)
    ax.set_xlabel("Acceptance threshold: raw HD (%)")
    ax.set_ylabel("Error rate")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0.98,
        0.96,
        "Zero empirical errors in the separation interval"
        if metrics["perfect_empirical_separation"]
        else f"EER = {100 * metrics['eer']:.4f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "far_frr_eer_curve.png", dpi=300)
    fig.savefig(out_dir / "far_frr_eer_curve.svg")
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    scores = collect_scores(args)
    genuine = np.asarray(
        [row["hd_fraction"] for row in scores if row["trial_type"] == "genuine"],
        dtype=float,
    )
    impostor = np.asarray(
        [row["hd_fraction"] for row in scores if row["trial_type"] == "impostor"],
        dtype=float,
    )
    curve, metrics = operating_curve(genuine, impostor)
    profile = np.load(args.candidate_profile, allow_pickle=False)
    shared_support = bool(
        "support_order_frozen" in profile.files
        and np.asarray(profile["support_order_frozen"]).reshape(-1)[0]
    )
    result = {
        "schema": "microled-puf-roc-eer-r1",
        "claim_scope": (
            "Descriptive closed-set image-attempt evaluation for M1-M9; "
            "repeated frames are correlated and do not constitute independent "
            "population samples."
        ),
        "score_definition": (
            (
                "Raw Hamming distance on the common frozen 2,048-projection "
                "support shared by all devices; accept when HD <= threshold."
            )
            if shared_support
            else (
                "Raw Hamming distance on the claimed device's frozen 2,048 "
                "device-specific positions; accept when HD <= threshold."
            )
        ),
        "shared_support": shared_support,
        "genuine": summarize(genuine),
        "impostor": summarize(impostor),
        **metrics,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "verification_scores.csv", scores)
    write_csv(args.out_dir / "roc_far_frr_curve.csv", curve)
    summary_rows = [
        {"trial_type": "genuine", **result["genuine"]},
        {"trial_type": "impostor", **result["impostor"]},
    ]
    write_csv(args.out_dir / "score_summary.csv", summary_rows)
    metric_rows = [
        {"metric": "AUC", "value": result["auc"], "unit": ""},
        {"metric": "EER", "value": result["eer"], "unit": "fraction"},
        {
            "metric": "EER_threshold",
            "value": result["eer_threshold_hd_percent"],
            "unit": "% raw HD",
        },
        {
            "metric": "maximum_genuine_HD",
            "value": 100.0 * result["maximum_genuine_hd_fraction"],
            "unit": "% raw HD",
        },
        {
            "metric": "minimum_impostor_HD",
            "value": 100.0 * result["minimum_impostor_hd_fraction"],
            "unit": "% raw HD",
        },
        {
            "metric": "empirical_score_gap",
            "value": 100.0 * result["score_gap_hd_fraction"],
            "unit": "percentage points",
        },
    ]
    write_csv(args.out_dir / "roc_eer_metrics.csv", metric_rows)
    (args.out_dir / "roc_eer_metrics.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    plot_distribution(args.out_dir, genuine, impostor, metrics)
    plot_roc(args.out_dir, curve, metrics)
    plot_far_frr(
        args.out_dir,
        curve,
        metrics,
        genuine.size,
        impostor.size,
    )
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--payload",
        type=Path,
        default=root / "models" / "expanded_luma_support_payload.npz",
    )
    parser.add_argument(
        "--candidate-profile",
        type=Path,
        default=root / "models" / "stability_only_candidate_profile.npz",
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=root / "models" / "enrollment_stability_only_manifest.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=root / "results" / "roc_eer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
