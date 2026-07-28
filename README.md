# Micro-LED PUF executable code

This repository contains one executable path:

```text
YOLO training
    -> STN training
    -> raw-image localization and alignment
    -> 2,048-bit PUF response extraction
    -> LDPC fuzzy extraction
    -> 128-bit output
```

A compact M1-M6 dataset and the required model files are included.

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux or macOS
source .venv/bin/activate
python -m pip install -r requirements.txt
```

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
| `code/microled_fuzzy_extractor.py` | Selects 2,048 device-local positions, performs LDPC decoding and produces the 128-bit output. This is the only fuzzy-extractor implementation. |
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
