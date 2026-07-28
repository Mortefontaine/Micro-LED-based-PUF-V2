# Execution levels

## Source and package checks

```bash
python scripts/validate_release.py
```

## Fuzzy-extractor check

```bash
python scripts/smoke_single_shot.py
```

## Compact model-training and pipeline check

```bash
python pipeline/run_all_m1_m6.py --force
```

This command trains for one epoch on the compact M1-M6 subset, aligns the
included raw images, creates six enrollment manifests and processes the
separate compact probe images.

The complete experimental image archive is maintained separately from this
code repository.
