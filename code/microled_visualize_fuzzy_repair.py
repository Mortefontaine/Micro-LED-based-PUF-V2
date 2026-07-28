"""Export one real 2048-bit response and visualize LDPC-repaired positions.

This is a private research/audit utility.  The exported bitstrings must not be
shipped with a production node or treated as public helper data.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image

from microled_eval_single_shot import load_split_manifest
from microled_puf_key import response_rows
from microled_single_shot_key import enroll_single_shot, get_regular_ldpc


def bitstring(bits: np.ndarray) -> str:
    return "".join(str(int(value)) for value in bits)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="M1")
    parser.add_argument("--condition", default="M1_30mA_30C_0")
    parser.add_argument("--frame", default="frame_0131.png")
    parser.add_argument("--input", type=Path, default=root / "data" / "03_aligned_puf_M1_M6")
    parser.add_argument("--payload", type=Path, default=root / "models" / "expanded_luma_support_payload.npz")
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
        "--output-dir",
        type=Path,
        default=root / "work" / "fuzzy_repair_M1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device.upper()
    split = load_split_manifest(args.split_manifest, args.input, args.payload, args.candidate_profile)
    enrollment_files = [
        (args.input / entry["relative_path"]).resolve()
        for entry in split["enrollment_entries"]
        if str(entry["relative_path"]).split("_", 1)[0].upper() == device
    ]
    enrollment_paths = {str(path).lower() for path in enrollment_files}

    enrollment_files = sorted(enrollment_files)
    probe_file = args.input / args.condition / args.frame
    if not probe_file.is_file():
        raise FileNotFoundError(probe_file)
    rows = [response_rows(path, args.payload)[0] for path in [*enrollment_files, probe_file]]
    enrollment = [
        row for row in rows if str(Path(str(row["path"])).resolve()).lower() in enrollment_paths
    ]
    if len(enrollment) != 9:
        raise RuntimeError(f"Expected 9 enrollment rows for {device}, found {len(enrollment)}")

    manifest = enroll_single_shot(device, enrollment, args.payload, args.candidate_profile)
    selected = np.asarray(manifest.candidate_indices, dtype=np.int64)
    enrollment_bits = np.stack(
        [np.asarray(row["candidate_bits"], dtype=np.uint8)[selected] for row in enrollment]
    )
    reference = (enrollment_bits.mean(axis=0) >= 0.5).astype(np.uint8)

    matches = [
        row
        for row in rows
        if str(row["condition"]) == args.condition and Path(str(row["path"])).name == args.frame
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one probe {args.condition}/{args.frame}, found {len(matches)}")
    row = matches[0]
    margins = np.asarray(row["candidate_margins"], dtype=np.float32)[selected]
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
    decoded = graph.decode_error(target, prior_llr, max_iterations=40, alpha=0.80)
    repair_mask = np.asarray(decoded["error"], dtype=np.uint8)
    recovered = observed ^ repair_mask

    if not bool(decoded["converged"]):
        raise RuntimeError("LDPC decoder did not converge for the selected probe")
    if not np.array_equal(recovered, reference):
        raise RuntimeError("Decoded response does not exactly match the enrolled M1 response")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "M1_raw_response_2048.txt").write_text(bitstring(observed) + "\n", encoding="ascii")
    (args.output_dir / "M1_recovered_response_2048.txt").write_text(bitstring(recovered) + "\n", encoding="ascii")
    (args.output_dir / "M1_repair_mask_2048.txt").write_text(bitstring(repair_mask) + "\n", encoding="ascii")

    with (args.output_dir / "M1_fuzzy_repair_bits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bit_index", "raw_bit", "recovered_bit", "repaired", "signed_margin"])
        for index in range(2048):
            writer.writerow([index, int(observed[index]), int(recovered[index]), int(repair_mask[index]), float(margins[index])])

    metadata = {
        "device": device,
        "probe": str(Path(str(row["path"])).resolve()),
        "response_bits": 2048,
        "repair_count": int(repair_mask.sum()),
        "raw_hamming_distance_to_reference": float(np.mean(observed != reference)),
        "decoder_converged": True,
        "decoder_iterations": int(decoded["iterations"]),
        "unsatisfied_checks": int(decoded["unsatisfied_checks"]),
        "exact_reference_recovered": True,
        "repair_indices_zero_based": np.flatnonzero(repair_mask).astype(int).tolist(),
    }
    (args.output_dir / "M1_fuzzy_repair_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    # Compact bitmaps matching the supplied 295x147 reference image.  Nearest-
    # neighbor resizing preserves the discrete 32x64 bit cells without blur.
    compact_size = (295, 147)
    raw_rgb = np.where(observed.reshape(32, 64)[:, :, None] == 1, 0, 255).astype(np.uint8)
    raw_rgb = np.repeat(raw_rgb, 3, axis=2)
    recovered_rgb = np.where(recovered.reshape(32, 64)[:, :, None] == 1, 0, 255).astype(np.uint8)
    recovered_rgb = np.repeat(recovered_rgb, 3, axis=2)
    flip_rgb = np.full((32, 64, 3), 255, dtype=np.uint8)
    flip_rgb[repair_mask.reshape(32, 64).astype(bool)] = np.array([23, 107, 135], dtype=np.uint8)
    for filename, pixels in [
        ("M1_raw_bits_295x147.png", raw_rgb),
        ("M1_recovered_bits_295x147.png", recovered_rgb),
        ("M1_flip_only_blue_295x147.png", flip_rgb),
    ]:
        Image.fromarray(pixels, mode="RGB").resize(compact_size, Image.Resampling.NEAREST).save(
            args.output_dir / filename, optimize=True
        )

    raw_grid = observed.reshape(32, 64)
    repaired_grid = recovered.reshape(32, 64)
    mask_grid = repair_mask.reshape(32, 64).astype(bool)
    display = repaired_grid.astype(np.uint8)
    display[mask_grid] = 2
    response_cmap = ListedColormap(["#F5F7F8", "#173F5F"])
    repaired_cmap = ListedColormap(["#F5F7F8", "#173F5F", "#D62828"])

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), constrained_layout=True)
    axes[0].imshow(raw_grid, cmap=response_cmap, vmin=0, vmax=1, interpolation="nearest", aspect="auto")
    axes[0].set_title("Raw M1 probe response (2048 bits)", fontsize=15, fontweight="bold")
    axes[1].imshow(display, cmap=repaired_cmap, vmin=0, vmax=2, interpolation="nearest", aspect="auto")
    axes[1].set_title(
        f"LDPC-recovered response — {int(repair_mask.sum())} repaired positions shown in red",
        fontsize=15,
        fontweight="bold",
    )
    for axis in axes:
        axis.set_xlabel("Bit position within each 64-bit row")
        axis.set_ylabel("Row (32 rows)")
        axis.set_xticks(np.arange(0, 64, 8))
        axis.set_yticks(np.arange(0, 32, 4))
    axes[1].legend(
        handles=[
            Patch(facecolor="#F5F7F8", edgecolor="#999999", label="Recovered bit 0"),
            Patch(facecolor="#173F5F", label="Recovered bit 1"),
            Patch(facecolor="#D62828", label="Position flipped by LDPC"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        f"Micro-LED PUF fuzzy extraction example: {device}, {args.condition}/{args.frame}\n"
        f"raw HD = {100 * np.mean(observed != reference):.2f}% · decoder iterations = {int(decoded['iterations'])} · exact recovery",
        fontsize=17,
        fontweight="bold",
    )
    figure_path = args.output_dir / "M1_2048bit_fuzzy_repair.png"
    fig.savefig(figure_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({**metadata, "figure": str(figure_path.resolve())}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
