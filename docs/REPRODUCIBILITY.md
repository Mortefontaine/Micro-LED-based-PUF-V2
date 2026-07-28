# Reproducibility levels

## Level 1: no GPU, no retraining

- Inspect all result CSV/JSON/Markdown files.
- Verify hashes with `python scripts/validate_release.py`.
- Parse the frozen enrollment split and device mapping.
- Regenerate table- and result-driven figures where supported.

## Level 2: frozen inference

Install PyTorch, OpenCV and Ultralytics. Run the frozen detector/STN and the
PUF enrollment/reproduction interface on the released sample.

## Level 3: model-training smoke test

The representative detector and STN training subsets exercise the public data
loaders and training programs. They are too small to reproduce the final
model quality.

## Level 4: full paper statistics

The image archive used for 3,645 normal-temperature probes and 1,800
high-temperature probes is not stored in this lightweight GitHub release.
Complete per-probe and aggregate outputs are included. If a journal requires
image-level re-estimation, publish the full archive separately under a DOI and
record its URL and checksum here.
