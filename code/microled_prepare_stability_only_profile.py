"""Create a candidate profile that uses no cross-device enrollment statistics.

All fixed sparse projections are eligible. Each device later chooses its own
2,048 positions using only within-device enrollment stability and margin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out-profile", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=8192)
    args = parser.parse_args()

    if args.candidate_count < 2048:
        raise ValueError("At least 2,048 fixed candidates are required.")

    args.out_profile.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_profile,
        eligible_indices=np.arange(args.candidate_count, dtype=np.int64),
        # Deliberately unavailable: no cross-device bit statistics are read
        # when creating this enrollment profile.
        ones_fraction=np.full(args.candidate_count, np.nan, dtype=np.float32),
        device_ids=np.asarray([], dtype="<U1"),
        low_fraction=np.asarray([0.0], dtype=np.float32),
        high_fraction=np.asarray([1.0], dtype=np.float32),
        selection_mode=np.asarray(["within_device_stability_only"]),
        cross_device_information_used=np.asarray([False]),
    )

    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    manifest["candidate_profile_sha256"] = sha256(args.out_profile)
    manifest["candidate_count"] = args.candidate_count
    manifest["eligible_count"] = args.candidate_count
    manifest["candidate_profile_selection_rule"] = (
        "all fixed candidates are eligible; each device selects 2048 positions "
        "using only its own enrollment stability and projection margin"
    )
    manifest["cross_device_enrollment_information_used"] = False
    manifest["experiment_note"] = (
        "Stability-only enrollment control. No population balance, inter-HD, "
        "or non-target-device response is used to select a device's support."
    )
    args.out_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_count": args.candidate_count,
                "eligible_count": args.candidate_count,
                "candidate_profile_sha256": manifest[
                    "candidate_profile_sha256"
                ],
                "cross_device_enrollment_information_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
