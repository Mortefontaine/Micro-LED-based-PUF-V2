# End-to-end PUF pipeline

## Enrollment

1. Read RGB near-field images from the declared operating conditions.
2. Localize the emitting area with the packaged YOLO detector.
3. Apply the expanded-background STN and save 256 x 256 RGB images.
4. Convert RGB to Rec.709 luma and remove the radial intensity trend.
5. Aggregate the residual into a 32 x 32 feature map.
6. Evaluate the ordered 8,192-projection candidate bank.
7. Select 2,048 positions using within-device stability and projection margin.
8. Form the 2,048-bit reference and the LDPC helper data.
9. Write the enrollment manifest.

## Reproduction

1. Read and align one probe image.
2. Apply the image-quality and pose thresholds.
3. Evaluate the enrolled device's 2,048 projection positions.
4. Run the regular LDPC decoder with the stored helper data.
5. Derive the 128-bit output with HKDF-SHA256.
6. Record the decoder and tag-comparison result.

The extraction architecture, candidate bank and ranking rule are shared across
all devices. The selected positions and helper manifest are created separately
for each enrolled device.
