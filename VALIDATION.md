# Validation commands

Run:

```bash
python scripts/validate_release.py
python scripts/smoke_single_shot.py
python pipeline/run_all_m1_m6.py --force
```

The release check compiles and imports the code, checks the packaged candidate
profile, resolves the compact STN pairs and verifies the expected entry
points.

The single-device check enrolls a 2,048-bit M1 response and processes a
separate M1 image.

The four-stage command trains the compact YOLO and STN examples, aligns the
included M1-M6 raw images, enrolls six devices and processes the compact probe
set.
