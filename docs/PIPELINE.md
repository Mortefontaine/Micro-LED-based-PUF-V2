# End-to-end PUF pipeline

## Factory enrollment

1. Capture RGB near-field images under nine current/temperature conditions.
2. Localize the emitting area with the frozen YOLO11n detector.
3. Apply a 1.70x expanded-background similarity STN and save 256x256 RGB.
4. Convert RGB to Rec.709 luma and remove the radial intensity trend.
5. Aggregate the residual into a 32x32 patch feature.
6. Evaluate the frozen deterministic 8,192-projection candidate bank.
7. For each physical device, use only its own nine selected enrollment images
   to rank stable candidates and freeze 2,048 public support indices.
8. Form the 2,048-bit enrolled reference and fuzzy-extractor helper data.
9. Derive local key material with HKDF-SHA256 and retain only authenticated
   helper metadata, not the raw response or root key.

The extraction architecture and candidate bank are shared by all devices.
Only enrollment-derived public support indices and reference-dependent helper
data are device-specific.

## Authentication/reproduction

1. Capture and align one fresh image.
2. Apply the pre-bit image-quality/pose gate.
3. Evaluate the enrolled device's 2,048 support positions.
4. Reconstruct the enrolled response with the regular LDPC secure sketch.
5. Derive the candidate key with HKDF-SHA256.
6. Accept only if decoder bounds and HMAC verification both pass.
7. On a quality-gate rejection, request a new frame; no wrong key is released.

## Dataset isolation

The 405 predeclared enrollment-pool images are excluded from probe metrics.
Only 81 selected images contribute to the frozen template/profile and
device-specific enrollment. The remaining 3,645 images are probes.

High-temperature images were evaluated only after normal-temperature
artifacts and parameters were frozen.
