# Final code path

The authoritative runtime path is:

1. `microled_expanded_align.py` — YOLO localization and expanded-background
   STN alignment.
2. `microled_puf.py` — Rec.709 conversion, radial residual, 32 × 32 feature
   map and deterministic projections.
3. `microled_prepare_stability_only_profile.py` — freezes the common
   8,192-candidate universe without population balance or inter-device
   separation.
4. `microled_single_shot_key.py` — device-specific enrollment, 2,048-bit
   response, LDPC reconstruction, HKDF and HMAC verification.

`microled_align.py` provides the common alignment utilities, and
`microled_puf_key.py` contains the reusable fuzzy-extractor/key primitives.
The main deployment interface is `microled_single_shot_key.py`. Publication
plotting and figure-table exports are maintained with the separate data
archive, not in this runtime repository.
