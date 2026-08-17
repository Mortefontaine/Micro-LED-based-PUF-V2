# Micro-LED PUF executable code

This repository contains one executable path:

```text
YOLO training
    -> STN training
    -> raw-image localization and alignment
    -> 2,048-bit PUF response extraction
    -> first-pass LDPC reconstruction (no pose refinement)
       -> if unsuccessful: one pose-refinement retry
    -> 2,048-bit response reconstruction
    -> HKDF-SHA256 derivation
       -> 256-bit root key
       -> 256-bit device-bound identity seed
```

A compact M1-M6 dataset and the required model files are included. The STN
subset contains three paired images per device/operating condition. Enrollment
uses one image from each of the nine current-temperature conditions and
requires at least 2,048 projection positions to be unanimous across those
nine images.

The source code and included repository materials are released under the
[MIT License](LICENSE).

## Derived-output convention

The fuzzy-extractor code reconstructs the 2,048-bit enrolled response and then
uses two domain-separated HKDF-SHA256 invocations to derive 32-byte outputs:

- `derive_root_key(...)` returns 256 bits;
- `derive_identity_seed(...)` returns a separate 256-bit, device-ID-bound
  value.

No truncation is applied to either output. Stage 4 records both output lengths
as `root_key_bits = 256` and `identity_seed_bits = 256`.

## Failure-triggered pose retry

Enrollment and every first-pass probe use the STN-normalized image directly;
pose refinement is disabled on this path. If first-pass reconstruction is
unsuccessful, the same image may undergo one common-template pose search and
one additional reconstruction attempt. Stage 4 records first-pass acceptance,
retry attempts and retry recovery in separate fields. Its primary acceptance
metric contains first-pass results only.

## Scope of the compact demo

The included M1-M6 data and one-epoch defaults are provided to execute every
software interface and verify the hand-off between stages. This compact demo
is not a reproduction of the publication-scale M1-M9 training, evaluation or
reported numerical results. Publication reproduction requires the separately
maintained full dataset, the original training schedule and the complete
evaluation split.

The compact STN fine-tuning stage uses a learning rate of `1e-4`, matching the
STN training rate used by the full local workflow. The alignment stage applies
the same default minimum valid-source fraction of `0.98`.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux or macOS
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The pinned package versions are the versions used for the clean-environment
validation of this compact release with Python 3.11.

## Run the complete example

```bash
python pipeline/run_all_m1_m6.py --force
```

All generated files are written under `work/m1_m6_closed_loop/`.

## Code inventory

These are all Python files retained in the repository.

| File | Purpose |
|---|---|
| `training/train_yolo.py` | Trains or fine-tunes the YOLO detector and writes a detector checkpoint. |
| `training/train_stn.py` | Trains the single STN model from paired crops and writes an STN checkpoint. |
| `code/microled_stn.py` | Defines the STN architecture and the input-feature transformation shared by training and inference. |
| `code/microled_align.py` | Loads YOLO and STN checkpoints, localizes each micro-LED image and writes aligned 256 x 256 RGB images. |
| `code/microled_puf.py` | Converts aligned images into features and evaluates the fixed projection bank. |
| `code/microled_prepare_stability_only_profile.py` | Creates the ordered candidate-position profile used during enrollment. |
| `code/microled_fuzzy_extractor.py` | Selects 2,048 device-local positions, performs LDPC decoding and derives separate 256-bit root-key and device-bound identity-seed outputs. This is the only fuzzy-extractor implementation. |
| `pipeline/common.py` | Reads and writes stage manifests and passes file paths between stages. |
| `pipeline/stage_01_train_yolo.py` | Runs YOLO training and records the detector checkpoint. |
| `pipeline/stage_02_train_stn.py` | Runs STN training and records the STN checkpoint. |
| `pipeline/stage_03_align_raw.py` | Passes both checkpoints to the alignment program and records the aligned-image folder. |
| `pipeline/stage_04_puf_fuzzy.py` | Uses aligned images for enrollment and probe processing. |
| `pipeline/run_all_m1_m6.py` | Runs stages 1-4 in order. |
| `scripts/test_fuzzy_extractor.py` | Runs one M1 enrollment and one separate M1 probe through the fuzzy extractor. |

## Run individual stages

```bash
python pipeline/stage_01_train_yolo.py --force
python pipeline/stage_02_train_stn.py --force
python pipeline/stage_03_align_raw.py --force
python pipeline/stage_04_puf_fuzzy.py --force
```

Each stage writes `stage_manifest.json`. The next stage reads the preceding
manifest rather than relying on an external project path.

## Run the fuzzy extractor directly

```bash
python code/microled_fuzzy_extractor.py --help
python scripts/test_fuzzy_extractor.py
```
