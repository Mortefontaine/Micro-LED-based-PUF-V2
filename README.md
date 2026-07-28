# Micro-LED PUF extraction pipeline

This repository contains the executable YOLO, STN, PUF-response and fuzzy-
extractor pipeline for nine micro-LED devices. A compact M1-M6 image subset is
included for installation checks and end-to-end execution.

## Method

The same detector, alignment model, 8,192-candidate projection bank and
selection rule are used for every device. During enrollment, each device uses
its own images to select 2,048 projection positions by within-device stability
and projection margin.

The repository contains one fuzzy-extractor implementation:

`code/microled_fuzzy_extractor.py`

It uses a regular LDPC decoder for response reproduction and HKDF-SHA256 for
the 128-bit output.

## Repository layout

```text
code/       detector, alignment, PUF response and fuzzy extractor
training/   YOLO and STN dataset preparation and training
models/     packaged YOLO, STN, transform payload and candidate profile
data/       compact M1-M6 samples
pipeline/   four-stage executable pipeline
scripts/    indexing and validation commands
tests/      lightweight tests
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux or macOS
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the complete M1-M6 example

```bash
python pipeline/run_all_m1_m6.py --force
```

The four stages are:

1. fine-tune the packaged YOLO checkpoint;
2. fine-tune the packaged STN checkpoint;
3. align the compact raw-image set;
4. enroll each device and reproduce outputs from separate probe images.

Each stage writes `stage_manifest.json`. The following stage reads its input
paths and digests from that manifest. Generated files are written under
`work/`.

Run a stage individually:

```bash
python pipeline/stage_01_train_yolo.py --force
python pipeline/stage_02_train_stn.py --force
python pipeline/stage_03_align_raw.py --force
python pipeline/stage_04_puf_fuzzy.py --force
```

## Direct fuzzy-extractor commands

Show the command-line interface:

```bash
python code/microled_fuzzy_extractor.py --help
```

Run the compact M1 enrollment and reproduction check:

```bash
python scripts/smoke_single_shot.py
```

## Validation

```bash
python scripts/validate_release.py
python scripts/smoke_single_shot.py
```

The complete experimental image archive and plotting data are distributed
separately.
