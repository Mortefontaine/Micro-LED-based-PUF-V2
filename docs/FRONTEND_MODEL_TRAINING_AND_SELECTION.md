# Front-end model training

## YOLO

- architecture: YOLO11n;
- initialization: `yolo11n.pt`;
- full training split: 1,969 images;
- full validation split: 490 images.

## STN

- stage 1: 4,050 paired crops, 26 epochs, blue-derived geometric loss;
- stage 2: stage-1 initialization, 12 epochs, learning rate `1e-4`;
- stage-2 input: Rec.709 luma with RGB-preserving warping.

The packaged weights are stored in `models/`. The compact M1-M6 subset
exercises the same dataset readers, model interfaces and checkpoint hand-off
with one training epoch by default.
