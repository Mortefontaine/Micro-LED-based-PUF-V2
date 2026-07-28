# Executable M1–M6 closed loop

The four stages operate only on files shipped in this repository. Each stage
writes a `stage_manifest.json`; the next stage resolves and verifies artifacts
from that manifest.

```text
M1–M6 YOLO sample ──> stage 1 detector checkpoint ─┐
M1–M6 STN pairs  ──> stage 2 STN checkpoint ──────┤
                                                   v
M1–M6 raw images ──> stage 3 aligned RGB images ──> stage 4 PUF/FE results
```

Run everything:

```bash
python pipeline/run_all_m1_m6.py --force
```

Or run stages separately:

```bash
python pipeline/stage_01_train_yolo.py --force
python pipeline/stage_02_train_stn.py --force
python pipeline/stage_03_align_raw.py --force
python pipeline/stage_04_puf_fuzzy.py --force
```

Defaults fine-tune the packaged checkpoints for one epoch on the compact public
sample. This makes the example deterministic enough for executable
reproduction without claiming that the small subset retrains the publication
models from scratch. Stage 3 processes all 108 raw M1–M6 images. Stage 4 uses
one image from every device/condition for enrollment and the other as an
independent regeneration probe.

Because the compact set provides only one enrollment image per condition, it
does not always contain 2,048 candidates that are unanimous across all nine
conditions. The sample stage therefore fills the 2,048 positions by the same
device-local flip-count and projection-margin ranking without an unanimity
constraint. It still uses no inter-device information. This relaxation is
recorded in the final stage manifest and applies only to the executable sample;
the reported full-data pipeline retains its frozen main definition.

Generated authentication keys, helper manifests and trained outputs stay
under `work/`, which is excluded by `.gitignore`.
