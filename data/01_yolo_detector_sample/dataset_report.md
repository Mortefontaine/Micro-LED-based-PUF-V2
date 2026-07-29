# Micro-LED Lightweight Detector Dataset

- Initial model: `models/yolo11n_microled_best.pt`
- Included images: 18
- Training images and labels: 12 image-label pairs
- Validation images and labels: 6 image-label pairs
- Devices represented: M1-M6
- Included Ultralytics YAML: `data/01_yolo_detector_sample/microled.yaml`

This compact dataset is included to execute and inspect the detector-training
interface. It is not the full publication-scale detector dataset. Stage 1
resolves the dataset root and writes a runtime YAML before invoking
`training/train_yolo.py`.
