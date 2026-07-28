# Compact M1–M6 sample data

This directory is intentionally small enough for a source-code repository.
It is a stratified sample of the experimental pipeline, not the complete
M1–M9 corpus used to obtain the reported metrics.

| Stage | Contents | Intended use |
|---|---|---|
| `00_raw_rgb_M1_M6` | 108 raw frames | visual inspection and detector input |
| `01_yolo_detector_sample` | compact images/labels split | YOLO smoke training |
| `02_stn_pairs_M1_M6` | 108 crop/target pairs | STN smoke training |
| `03_aligned_puf_M1_M6` | 108 aligned images | PUF/FE code-path examples |

For each of M1–M6, the aligned sample contains two images from each of the
nine current/temperature conditions. This supports a minimal one-image
enrollment plus an independent example probe per condition, but it is too
small to reproduce the statistical confidence of the 3,645-probe evaluation.

The experimental training pipeline deliberately introduced larger acquisition
variations such as displacement, rotation and blur, with one transformed
sample replacing one nominal sample rather than retaining both. This increases
the diversity presented to the front end, but it must be described as
robustness-oriented data collection/augmentation, not as additional
independent physical devices.

`DATASET_INDEX.csv` summarizes each stage. `SAMPLE_FILE_SHA256.csv` records
file-level checksums.
