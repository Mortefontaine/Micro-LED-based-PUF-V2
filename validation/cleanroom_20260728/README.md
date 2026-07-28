# Clean-room M1–M6 validation

This folder records an end-to-end run performed on 2026-07-28 from a fresh
copy of this release and a newly created Python virtual environment. Before
installation, neither `torch` nor `cv2` was importable from that environment.

## Environment

- Windows, CPU-only execution
- Python 3.11.14
- exact top-level package versions: `requirements-tested.txt` at repository root
- no parent-project source directory was placed on `PYTHONPATH`
- no files outside the clean repository copy were used as model, data or code inputs

## Command

```bash
python pipeline/run_all_m1_m6.py --force --device cpu \
  --yolo-epochs 1 --stn-epochs 1 --yolo-batch 4 --stn-batch 8
```

The pipeline completed all four stages:

1. YOLO fine-tuning on the included compact detector sample;
2. STN fine-tuning on the 108 included M1–M6 alignment pairs;
3. alignment of all 108 included raw images;
4. per-device stability-only enrollment and independent fuzzy-extractor
   regeneration.

Each stage consumed the preceding stage's manifest/checkpoint/image artifacts.
The run therefore tests the file interfaces between scripts, rather than
executing four disconnected examples.

## Observed results

- YOLO validation: precision 1.000, recall 1.000, mAP50 0.995 and
  mAP50–95 0.995 on the six-image compact validation split.
- STN validation correlation: 0.6324 before and 0.8482 after alignment.
- Alignment: 108/108 input images processed.
- PUF enrollment: six devices and 54 enrollment images.
- Independent regeneration: 53/54 probes accepted (98.148%).
- M1, M2, M3, M4 and M6: 9/9 accepted; M5: 8/9 accepted.
- The failed probe was `M5_20mA_40C_0/frame_0002.png`; its quality correlation
  passed the 0.4 gate, but the decoder/key check did not converge.
- Response length: 2,048 bits; derived key: 256 bits; the protocol uses
  128 identity-bound bits.
- Bit selection used within-device enrollment stability only. No inter-device
  Hamming-distance information was used.

The machine-readable outputs in this directory are:

- `stage1_yolo_results.csv`
- `stage2_stn_training_history.csv`
- `stage3_alignment_metrics.csv`
- `stage4_per_device_results.csv`
- `stage4_per_probe_results.csv`
- `stage4_summary.json`

Private authentication material, helper-data manifests and generated
checkpoints are intentionally not committed with this validation record. They
are regenerated under `work/m1_m6_closed_loop/` when the pipeline runs.

## Interpretation boundary

This is a compact executable-data validation, not a reproduction of the
publication-scale M1–M9 statistics. The one-epoch detector metrics are based on
only six validation images and must not be presented as an independent
benchmark. Because the compact enrollment subset provides fewer unanimously
stable candidates than the full dataset, Stage 4 ranks candidates by
device-local flip count and margin without requiring unanimity. The
publication-scale enrollment path retains its stricter defaults.
