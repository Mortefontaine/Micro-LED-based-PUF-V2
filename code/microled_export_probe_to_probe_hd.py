"""Export old-definition probe-to-probe HD using the current PUF pipeline.

Intra-device distances compare two probes from the same device on that
device's enrolled ordered 2,048-bit support. Inter-device distances compare
the two final ordered 2,048-bit output codes, with each probe evaluated on its
own device-specific support.

The inter-device result is an output-code separation metric. It is not a
common-coordinate physical uniqueness measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from microled_eval_single_shot import load_split_manifest
from microled_puf import device_name, natural_key
from microled_puf_key import response_rows
from microled_single_shot_key import enroll_single_shot


def summarize(values: np.ndarray) -> dict[str, float | int]:
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "p01": float(np.percentile(values, 1)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


def sample_intra_pairs(
    size: int,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.triu_indices(size, k=1)
    if first.size < count:
        raise ValueError(
            f"Only {first.size} unique same-device pairs are available; "
            f"{count} were requested."
        )
    selected = rng.choice(first.size, size=count, replace=False)
    return first[selected], second[selected]


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "class",
        "device_pair",
        "left_probe",
        "right_probe",
        "mismatch_bits_of_2048",
        "hamming_distance",
        "hamming_distance_percent",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_density(
    path: Path,
    intra: np.ndarray,
    inter: np.ndarray,
    bins: np.ndarray,
) -> None:
    intra_counts, _ = np.histogram(intra, bins=bins)
    inter_counts, _ = np.histogram(inter, bins=bins)
    fields = [
        "bin_left",
        "bin_right",
        "bin_center",
        "bin_width",
        "intra_count",
        "intra_relative_frequency_percent",
        "intra_probability_density",
        "inter_count",
        "inter_relative_frequency_percent",
        "inter_probability_density",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, width in enumerate(np.diff(bins)):
            writer.writerow(
                {
                    "bin_left": float(bins[index]),
                    "bin_right": float(bins[index + 1]),
                    "bin_center": float(
                        (bins[index] + bins[index + 1]) / 2
                    ),
                    "bin_width": float(width),
                    "intra_count": int(intra_counts[index]),
                    "intra_relative_frequency_percent": float(
                        100.0 * intra_counts[index] / intra.size
                    ),
                    "intra_probability_density": float(
                        intra_counts[index] / (intra.size * width)
                    ),
                    "inter_count": int(inter_counts[index]),
                    "inter_relative_frequency_percent": float(
                        100.0 * inter_counts[index] / inter.size
                    ),
                    "inter_probability_density": float(
                        inter_counts[index] / (inter.size * width)
                    ),
                }
            )


def write_origin_density(
    path: Path,
    intra: np.ndarray,
    inter: np.ndarray,
    bins: np.ndarray,
) -> None:
    intra_counts, _ = np.histogram(intra, bins=bins)
    inter_counts, _ = np.histogram(inter, bins=bins)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Hamming distance",
                "Intra Hamming distance (%)",
                "Inter Hamming distance (%)",
            ]
        )
        for index in range(len(bins) - 1):
            writer.writerow(
                [
                    float((bins[index] + bins[index + 1]) / 2),
                    float(100.0 * intra_counts[index] / intra.size),
                    float(100.0 * inter_counts[index] / inter.size),
                ]
            )


def plot_distribution(
    out_dir: Path,
    intra: np.ndarray,
    inter: np.ndarray,
    bins: np.ndarray,
) -> None:
    colors = ("#2D6DA3", "#8EA5CF")
    figure, axis = plt.subplots(
        figsize=(7.2, 4.8),
        constrained_layout=True,
    )
    axis.hist(
        intra,
        bins=bins,
        weights=np.full(intra.size, 100.0 / intra.size),
        color=colors[0],
        edgecolor="white",
        linewidth=0.35,
        label="Intra-class",
    )
    axis.hist(
        inter,
        bins=bins,
        weights=np.full(inter.size, 100.0 / inter.size),
        color=colors[1],
        edgecolor="white",
        linewidth=0.35,
        label="Inter-class",
    )
    axis.axvline(intra.mean(), color=colors[0], lw=1.3, ls="--")
    axis.axvline(inter.mean(), color=colors[1], lw=1.3, ls="--")
    axis.text(
        0.25,
        0.63,
        f"Intra-class mean: {intra.mean():.4f}",
        transform=axis.transAxes,
        color=colors[0],
        fontsize=11,
    )
    axis.text(
        0.25,
        0.54,
        f"Inter-class mean: {inter.mean():.4f}",
        transform=axis.transAxes,
        color=colors[1],
        fontsize=11,
    )
    x_max = max(0.6, np.ceil(max(intra.max(), inter.max()) / 0.05) * 0.05)
    axis.set(
        xlim=(0, min(1.0, x_max)),
        ylim=(0, 100),
        xlabel="Hamming distance",
        ylabel="Relative frequency per 0.01 bin (%)",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper right")
    axis.tick_params(direction="out", length=3)
    figure.savefig(
        out_dir / "probe_to_probe_hd_current_pipeline.png",
        dpi=350,
        facecolor="white",
    )
    figure.savefig(
        out_dir / "probe_to_probe_hd_current_pipeline.svg",
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--intra-pairs-per-device", type=int, default=4000)
    parser.add_argument(
        "--inter-pairs-per-device-pair",
        type=int,
        default=1000,
    )
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

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
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[device_name(str(row["condition"]))].append(row)

    codes: dict[str, np.ndarray] = {}
    supports: dict[str, np.ndarray] = {}
    labels: dict[str, list[str]] = {}
    for device, device_rows in sorted(
        grouped.items(),
        key=lambda item: natural_key(item[0]),
    ):
        enrollment = [
            row
            for row in device_rows
            if str(Path(str(row["path"])).resolve()).lower()
            in enrollment_paths
        ]
        manifest = enroll_single_shot(
            device,
            enrollment,
            args.payload,
            args.candidate_profile,
        )
        selected = np.asarray(manifest.candidate_indices, dtype=np.int64)
        supports[device] = selected
        probes = [
            row
            for row in device_rows
            if str(Path(str(row["path"])).resolve()).lower() not in pool_paths
        ]
        codes[device] = np.stack(
            [
                np.asarray(row["candidate_bits"], dtype=np.uint8)[selected]
                for row in probes
            ]
        )
        labels[device] = [
            f"{row['condition']}/{Path(str(row['path'])).name}"
            for row in probes
        ]

    devices = sorted(codes, key=natural_key)
    shared_support = all(
        np.array_equal(supports[devices[0]], supports[device])
        for device in devices[1:]
    )
    expected_devices = [f"M{index}" for index in range(1, 10)]
    if devices != expected_devices:
        raise ValueError(f"Expected {expected_devices}; obtained {devices}.")
    if any(codes[device].shape != (405, 2048) for device in devices):
        shapes = {device: list(codes[device].shape) for device in devices}
        raise ValueError(f"Expected 405 x 2048 probe codes per device: {shapes}")

    rng = np.random.default_rng(args.seed)
    records: list[dict[str, Any]] = []
    intra_values: list[float] = []
    inter_values: list[float] = []

    for device in devices:
        left, right = sample_intra_pairs(
            codes[device].shape[0],
            args.intra_pairs_per_device,
            rng,
        )
        mismatch = np.count_nonzero(
            codes[device][left] != codes[device][right],
            axis=1,
        )
        values = mismatch / 2048.0
        intra_values.extend(values.tolist())
        for left_index, right_index, bits, value in zip(
            left,
            right,
            mismatch,
            values,
        ):
            records.append(
                {
                    "class": "Intra-class",
                    "device_pair": device,
                    "left_probe": labels[device][int(left_index)],
                    "right_probe": labels[device][int(right_index)],
                    "mismatch_bits_of_2048": int(bits),
                    "hamming_distance": float(value),
                    "hamming_distance_percent": float(100.0 * value),
                }
            )

    for left_device_index, left_device in enumerate(devices):
        for right_device in devices[left_device_index + 1 :]:
            right_count = codes[right_device].shape[0]
            total_pairs = codes[left_device].shape[0] * right_count
            selected_pairs = rng.choice(
                total_pairs,
                size=args.inter_pairs_per_device_pair,
                replace=False,
            )
            left = selected_pairs // right_count
            right = selected_pairs % right_count
            mismatch = np.count_nonzero(
                codes[left_device][left] != codes[right_device][right],
                axis=1,
            )
            values = mismatch / 2048.0
            inter_values.extend(values.tolist())
            for left_index, right_index, bits, value in zip(
                left,
                right,
                mismatch,
                values,
            ):
                records.append(
                    {
                        "class": "Inter-class",
                        "device_pair": f"{left_device}-{right_device}",
                        "left_probe": labels[left_device][int(left_index)],
                        "right_probe": labels[right_device][int(right_index)],
                        "mismatch_bits_of_2048": int(bits),
                        "hamming_distance": float(value),
                        "hamming_distance_percent": float(100.0 * value),
                    }
                )

    intra = np.asarray(intra_values, dtype=float)
    inter = np.asarray(inter_values, dtype=float)
    expected_class_count = 9 * args.intra_pairs_per_device
    if intra.size != expected_class_count:
        raise RuntimeError(
            f"Expected {expected_class_count} intra pairs; got {intra.size}."
        )
    expected_inter_count = 36 * args.inter_pairs_per_device_pair
    if inter.size != expected_inter_count:
        raise RuntimeError(
            f"Expected {expected_inter_count} inter pairs; got {inter.size}."
        )

    bins = np.arange(0.0, 1.001, 0.01)
    write_records(
        args.out_dir / "probe_to_probe_hd_current_pipeline_points.csv",
        records,
    )
    write_density(
        args.out_dir / "probe_to_probe_hd_current_pipeline_density.csv",
        intra,
        inter,
        bins,
    )
    write_origin_density(
        args.out_dir / "probe_to_probe_hd_current_pipeline_origin.csv",
        intra,
        inter,
        bins,
    )
    plot_distribution(args.out_dir, intra, inter, bins)

    summary = {
        "definition": {
            "intra": (
                "same-device probe pairs compared on that device's enrolled "
                "ordered 2048-bit support"
            ),
            "inter": (
                (
                    "cross-device probe pairs compared on the same common "
                    "ordered 2048-projection support"
                )
                if shared_support
                else (
                    "cross-device probe pairs comparing each device's final "
                    "ordered 2048-bit output code"
                )
            ),
            "interpretation": (
                (
                    "common-coordinate physical-response separation"
                )
                if shared_support
                else (
                    "output-code separation; not common-coordinate physical "
                    "uniqueness"
                )
            ),
        },
        "input": {
            "devices": devices,
            "probes_per_device": {
                device: int(codes[device].shape[0]) for device in devices
            },
            "excluded_enrollment_pool_images": len(pool_paths),
            "seed": args.seed,
            "shared_support": shared_support,
            "intra_pairs_per_device": args.intra_pairs_per_device,
            "inter_pairs_per_device_pair": (
                args.inter_pairs_per_device_pair
            ),
        },
        "intra": summarize(intra),
        "inter": summarize(inter),
        "strict_gap": float(inter.min() - intra.max()),
    }
    (args.out_dir / "probe_to_probe_hd_current_pipeline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
