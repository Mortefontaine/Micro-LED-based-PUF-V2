# Training and preprocessing code

These scripts document the model-building path before the frozen runtime
artifacts in `models/`:

- `microled_prepare_light_detector_dataset.py`
- `microled_train_ultralight_yolo.py`
- `microled_prepare_affine_alignment_pairs.py`
- `microled_train_luma_spatial_head_stn.py`

The remaining modules provide imported model architectures, alternative
alignment controls and loss functions. All defaults now resolve inside this
repository. The primary YOLO and luma-STN trainers use the compact M1-M6
sample and write only to `work/`; the complete chained example is under
`pipeline/`.

The frozen release models were rebuilt with the public M1-M9 device set:

- YOLO11n: 1,969 training images and 490 validation images, initialized from
  generic `yolo11n.pt`.
- STN stage 1: 4,050 paired images, 26 epochs, blue-derived geometric loss.
- STN stage 2: initialized from stage 1, 12 epochs at `1e-4`, Rec.709-luma
  fine-tuning while preserving RGB output.

The reported evaluation is a nine-device closed-set experiment. No unseen-
device generalization claim is made.

The public detector subset is intentionally small. It verifies data loading
and command execution but is not expected to reproduce the accuracy of the
frozen detector. The same limitation applies to STN retraining.
