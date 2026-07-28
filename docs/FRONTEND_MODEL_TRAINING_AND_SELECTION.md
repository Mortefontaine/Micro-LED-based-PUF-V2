# Front-end model training and selection

The frozen YOLO11n and STN use all nine reported devices, M1-M9. This is a
closed-set nine-device evaluation; no unseen-device generalization claim is
made.

## Frozen training scope

- YOLO11n: 1,969 training and 490 validation images, initialized from generic
  `yolo11n.pt`.
- STN stage 1: 4,050 paired crops, 26 epochs, blue-derived geometric loss.
- STN stage 2: initialized from stage 1, 12 epochs at a `1e-4` learning rate,
  Rec.709-luma fine-tuning with RGB-preserving warping.

## Selection evidence

On the M1-M9 validation set, the new YOLO reached 100% recall and 99.5%
mAP50. On the same 4,050 YOLO crops, the new STN improved mean luma
correlation from 0.9494 to 0.9566 and the fifth percentile from 0.8764 to
0.8915 relative to the previous STN.

After rebuilding enrollment artifacts, all 3,645 non-enrollment probes
reproduced the enrolled key exactly. Mean intra-device HD was 0.3509%, p95 was
1.8945%, and p99 was 5.5449%. No cross-device attempt was accepted in 29,160
ordered impostor trials.

An independent post-freeze check on 1,800 supplied 65/80 C tight crops yielded
1,795 exact key recoveries (99.7222%). These temperature images were not used
for training, enrollment construction or parameter selection. The temperature
test exercises the STN-compatible PUF/fuzzy-extraction backend and does not
claim to evaluate raw-camera YOLO localization at elevated temperature.

The complete model-selection record is retained with the local experiment
artifacts. The public repository contains the selected frozen models and the
complete per-probe result tables.
