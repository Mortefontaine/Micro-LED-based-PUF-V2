# Stability-only enrollment protocol

## Frozen before device enrollment

- YOLO detector and STN weights;
- transform payload and the ordered 8,192 candidate projections;
- number of enrollment images (nine);
- number of selected bits (2,048);
- stability, margin, quality-gate and decoder parameters.

## Device-local enrollment

For a device \(d\), only its nine declared enrollment images are processed.
For candidate \(j\), the implementation computes the repeated binary outcome,
within-device flip/error count and projection margin. Candidates are ranked
deterministically by the frozen stability/margin rule, and the top 2,048 are
selected.

No response from another device, no probe image, no class label separation and
no inter-device Hamming distance enters the ranking.

The enrollment output contains public helper data, the selected position
indices, frozen configuration/model digests and a key-verification tag. It does
not store the enrolled 2,048-bit response or regenerated root key.

## Regeneration

A one-shot probe passes through the same detector, alignment and projection
pipeline. Its selected 2,048 noisy bits and the public helper data enter the
decoder. A candidate 128-bit key is released only if decoding and the
verification tag succeed; otherwise the operation fails closed.

## Evaluation

The full result uses 9 devices, 81 selected enrollment images and 3,645
held-out one-shot probes. The complete enrollment pool is excluded from the
probe set. Position selection is run before evaluating reliability,
uniformity, uniqueness, bit aliasing, fuzzy recovery and ROC/EER.
