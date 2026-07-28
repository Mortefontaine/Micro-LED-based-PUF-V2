# Training and preprocessing

Primary scripts:

- `microled_prepare_light_detector_dataset.py`
- `microled_train_ultralight_yolo.py`
- `microled_prepare_affine_alignment_pairs.py`
- `microled_train_luma_spatial_head_stn.py`

All default paths resolve inside this repository. Training outputs are written
under `work/`.

The packaged models in `models/` use the full nine-device training set. The
compact M1-M6 data exercise the training entry points and checkpoint formats.
