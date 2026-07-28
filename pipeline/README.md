# Executable M1-M6 pipeline

Each stage writes a `stage_manifest.json`, and the next stage reads its input
artifacts from that manifest.

```text
YOLO sample -> detector checkpoint
STN pairs   -> STN checkpoint
raw images  -> aligned images -> PUF enrollment and reproduction
```

Run all stages:

```bash
python pipeline/run_all_m1_m6.py --force
```

Run stages separately:

```bash
python pipeline/stage_01_train_yolo.py --force
python pipeline/stage_02_train_stn.py --force
python pipeline/stage_03_align_raw.py --force
python pipeline/stage_04_puf_fuzzy.py --force
```

The compact set contains one enrollment image and one probe image for each
device/condition pair. Stage 4 ranks candidate positions by device-local
agreement and projection margin without requiring unanimous candidates.

Generated checkpoints, aligned images and manifests are written under `work/`.
