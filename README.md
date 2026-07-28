# Micro-LED PUF with device-local stability-only enrollment

This lightweight code release contains the YOLO detector, STN alignment, PUF
extraction and fuzzy-extractor implementation used for a nine-device micro-LED
PUF study. A small M1–M6 image subset is included only for executable examples
and smoke tests. The complete experimental dataset and publication figure data
are archived separately and are intentionally not mirrored in this repository.

## What is different in this release

The enrollment rule does **not** use inter-device Hamming distance, device
labels from other micro-LEDs, or a population-separation objective.

Each device:

1. is detected by the same YOLO front end;
2. is aligned by the same STN;
3. is mapped into the same fixed universe of 8,192 candidate projections;
4. selects 2,048 positions using only nine enrollment images from that device,
   ranked by within-device stability and projection margin;
5. regenerates a stable 128-bit key from the noisy 2,048-bit response and
   public helper data.

Different devices may select different subsets, but the algorithm,
hyperparameters and candidate universe are shared.

## Repository map

```text
code/       PUF, enrollment and fuzzy-extractor runtime code
training/   YOLO/STN dataset preparation and training implementations
models/     frozen YOLO, STN, transform payload and stability-only profile
data/       small M1–M6 samples for the four main stages
docs/       pipeline, training, reproducibility and security notes
scripts/    release indexing and validation helpers
tests/      lightweight smoke tests
validation/ clean-room M1–M6 execution record
```

## Data and result availability

This repository is the executable code component of the release. It does not
duplicate the publication-scale raw data, processed figure tables, Origin
projects or rendered publication figures. The separate data archive should be
cited for those artifacts. `validation/cleanroom_20260728/` contains only the
small run record needed to demonstrate that the included M1–M6 example closes
the complete software loop.

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Inspect the included data

- `data/00_raw_rgb_M1_M6/`: representative raw RGB frames.
- `data/01_yolo_detector_sample/`: a compact YOLO dataset.
- `data/02_stn_pairs_M1_M6/`: paired detector crops and targets.
- `data/03_aligned_puf_M1_M6/`: two aligned images per condition and device,
  sufficient for code-path smoke tests but not for reproducing the reported
  M1–M9 statistics.

See `data/README.md` for the sampling policy and file index.

## Train the compact front ends

YOLO smoke training:

```bash
python training/microled_train_ultralight_yolo.py \
  --data data/01_yolo_detector_sample/microled.yaml \
  --project work/yolo --epochs 1 --batch 4
```

STN smoke training:

```bash
python training/microled_train_luma_spatial_head_stn.py \
  --pair-csv data/02_stn_pairs_M1_M6/alignment_pairs_M1_M6.csv \
  --out-dir work/stn --epochs 1 --batch-size 8 --device cpu
```

The release models in `models/` were trained on the full M1–M9 working
dataset, not on this compact public sample.

## Run the self-contained M1–M6 loop

The executable public-data path fine-tunes the packaged detector and STN,
passes their checkpoint manifests to the raw-image alignment stage, and then
passes the aligned images into device-local PUF enrollment and fuzzy
regeneration:

```bash
python pipeline/run_all_m1_m6.py --force
```

Every stage writes `stage_manifest.json`, including artifact paths and SHA-256
digests. The next stage resolves its inputs from those manifests. See
`pipeline/README.md` for the individual commands and interpretation boundary.

This compact loop is an executability/reproducibility example. It does not
replace the complete M1–M9 dataset or claim that one-epoch M1–M6 fine-tuning
reproduces the publication statistics.

## PUF and fuzzy-extractor entry points

- `code/microled_prepare_stability_only_profile.py`: freezes the common
  candidate universe without using inter-device separation.
- `code/microled_single_shot_key.py`: enrolls a device and regenerates its
  key from a one-shot probe.

Run `python <script> --help` for the exact command-line interface. The compact
data are intended for smoke tests; publication-scale evaluation belongs with
the separately archived full dataset.

Structural and end-to-end checks:

```bash
python scripts/validate_release.py
python scripts/smoke_single_shot.py
```

## Claim boundary

This is research code. The fuzzy extractor demonstrates stable, verified,
fail-closed key regeneration. Neither this code release nor its compact sample
constitutes a standalone proof of 128-bit PUF min-entropy, a production
cryptographic certification, or resistance to invasive physical attacks.
Public helper-data leakage and the entropy assumption must be treated
separately; see `docs/SECURITY_BOUNDARY.md`.
