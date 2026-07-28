# Micro-LED PUF with device-local stability-only enrollment

This lightweight research release contains the YOLO detector, STN alignment,
PUF extraction and fuzzy-extractor implementation used for a nine-device
micro-LED PUF study. A small M1–M6 image subset is included for inspection and
smoke tests; the frozen model files and reported metrics were obtained from the
full M1–M9 experiment.

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
code/       PUF, enrollment, fuzzy extractor and evaluation code
training/   YOLO/STN dataset preparation and training implementations
models/     frozen YOLO, STN, transform payload and stability-only profile
data/       small M1–M6 samples for the four main stages
results/    compact M1–M9 reported results and Origin-ready CSV files
docs/       pipeline, training, reproducibility and security notes
scripts/    release indexing and validation helpers
tests/      lightweight smoke tests
```

## Reported M1–M9 results

| Metric | Result |
|---|---:|
| Reliability | 99.7829% |
| Mean probe/reference intra-HD | 0.2171% |
| Reference uniformity | 50.1139% |
| Device-balanced probe uniformity | 50.1199% |
| Bit aliasing | 50.1506% |
| Common-candidate uniqueness | 53.8615% |
| Exact fuzzy-extractor recovery | 100% (3,645/3,645 probes) |
| ROC AUC / empirical EER | 1.000 / 0 |

`results/probe_to_probe_hd/` also contains the legacy figure definition:
output-code intra-HD `0.4410 ± 1.2980%` and inter-HD
`49.9468 ± 0.8997%`. That inter-HD describes separation of the final output
codes. Since devices may use different selected positions, it must not be
presented as common-coordinate physical uniqueness.

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
- `code/microled_eval_single_shot.py`: evaluates the frozen M1–M9 split when
  the full aligned dataset is supplied.
- `code/microled_visualize_fuzzy_repair.py`: exports raw, repaired and flip
  maps.

Run `python <script> --help` for the exact command-line interface. The compact
data are intended for smoke tests; reproducing the published numerical table
requires the full corpus named in the frozen split manifest.

Structural and end-to-end checks:

```bash
python scripts/validate_release.py
python scripts/smoke_single_shot.py
```

## Claim boundary

This is research code. The fuzzy extractor demonstrates stable, verified,
fail-closed key regeneration. The reported results do not constitute a
standalone proof of 128-bit PUF min-entropy, a production cryptographic
certification, or resistance to invasive physical attacks. Public helper-data
leakage and the entropy assumption must be treated separately; see
`docs/SECURITY_BOUNDARY.md`.
