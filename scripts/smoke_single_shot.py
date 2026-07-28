"""End-to-end M1 enrollment/regeneration smoke test on the compact sample."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "code" / "microled_single_shot_key.py"
ALIGNED = ROOT / "data" / "03_aligned_puf_M1_M6"
PAYLOAD = ROOT / "models" / "expanded_luma_support_payload.npz"
PROFILE = ROOT / "models" / "stability_only_candidate_profile.npz"
MODELS = ROOT / "models"


def run(*arguments: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(CLI), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    candidates = sorted((ALIGNED / "M1_10mA_20C_0").glob("*.png"))
    if len(candidates) < 2:
        raise RuntimeError("The compact M1 condition needs two aligned images")
    probe = candidates[1]
    with tempfile.TemporaryDirectory(prefix="microled_puf_smoke_") as folder:
        state = Path(folder)
        auth_key = state / "auth.key"
        manifest = state / "M1_manifest.json"
        run("init-auth-key", "--out", str(auth_key))
        enrollment = run(
            "enroll",
            "--input",
            str(ALIGNED),
            "--source-device",
            "M1",
            "--device-id",
            "demo-M1",
            "--manifest",
            str(manifest),
            "--payload",
            str(PAYLOAD),
            "--candidate-profile",
            str(PROFILE),
            "--per-condition",
            "1",
            "--models-dir",
            str(MODELS),
            "--manifest-auth-key",
            str(auth_key),
        )
        regeneration = run(
            "reproduce",
            "--input",
            str(probe),
            "--manifest",
            str(manifest),
            "--payload",
            str(PAYLOAD),
            "--candidate-profile",
            str(PROFILE),
            "--models-dir",
            str(MODELS),
            "--manifest-auth-key",
            str(auth_key),
        )
    if not regeneration.get("accepted"):
        raise RuntimeError(f"Key regeneration failed: {regeneration}")
    print(
        json.dumps(
            {
                "enrollment_images": enrollment["enrollment_images"],
                "response_bits": enrollment["response_bits"],
                "accepted": regeneration["accepted"],
                "decoder_converged": regeneration["decoder_converged"],
                "probe": probe.relative_to(ROOT).as_posix(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
