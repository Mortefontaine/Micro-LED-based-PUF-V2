"""Export compact Origin-ready uniformity and raw-BER point tables."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(values_by_device: dict[str, list[float]], value_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for device in sorted(values_by_device, key=lambda value: int(value[1:])):
        values = np.asarray(values_by_device[device], dtype=float)
        rows.append(
            {
                "device": device,
                "n": int(values.size),
                f"mean_{value_name}": float(values.mean()),
                f"std_{value_name}": float(values.std()),
                f"p05_{value_name}": float(np.percentile(values, 5)),
                f"p95_{value_name}": float(np.percentile(values, 95)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.probe_csv.open(encoding="utf-8-sig", newline="") as handle:
        probes = list(csv.DictReader(handle))

    uniformity_rows: list[dict[str, object]] = []
    ber_rows: list[dict[str, object]] = []
    uniformity_by_device: dict[str, list[float]] = defaultdict(list)
    ber_by_device: dict[str, list[float]] = defaultdict(list)
    for row in probes:
        device = row["device"]
        device_index = int(device[1:])
        uniformity = float(row["observed_uniformity_percent"])
        raw_ber = 100.0 * float(row["raw_hd"])
        common = {
            "device": device,
            "device_index": device_index,
            "condition": row["condition"],
            "frame": row["frame"],
        }
        uniformity_rows.append({**common, "uniformity_percent": uniformity})
        ber_rows.append(
            {
                **common,
                "raw_ber_percent": raw_ber,
                "gate_passed": row["quality_gate_passed"],
                "accepted": row["accepted"],
            }
        )
        uniformity_by_device[device].append(uniformity)
        ber_by_device[device].append(raw_ber)

    write_rows(args.out_dir / "uniformity_points_origin_long.csv", uniformity_rows)
    write_rows(args.out_dir / "raw_ber_points_origin_long.csv", ber_rows)
    write_rows(
        args.out_dir / "uniformity_summary.csv",
        summary(uniformity_by_device, "uniformity_percent"),
    )
    write_rows(
        args.out_dir / "raw_ber_summary.csv",
        summary(ber_by_device, "raw_ber_percent"),
    )
    print(f"uniformity_points={len(uniformity_rows)} raw_ber_points={len(ber_rows)}")


if __name__ == "__main__":
    main()
