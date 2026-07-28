# Stability-only enrollment

The following inputs are fixed before per-device enrollment:

- YOLO and STN weights;
- transform payload and ordered 8,192 candidate projections;
- number of enrollment images;
- 2,048-bit response length;
- stability, margin, image-quality and decoder parameters.

For each device, the code evaluates only that device's declared enrollment
images. Each candidate receives a within-device agreement value and an average
absolute projection margin. Candidates are ordered deterministically by these
values, and the first 2,048 positions are stored in the device manifest.

Responses from other devices and inter-device Hamming distances are not used
by the position-selection function.

A probe image follows the same alignment and projection steps. Its 2,048
values and the stored helper data are passed to the LDPC decoder, followed by
HKDF-SHA256 output derivation and tag comparison.
