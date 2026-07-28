# Runtime code

The runtime path is:

1. `microled_expanded_align.py` - YOLO localization and STN alignment.
2. `microled_puf.py` - Rec.709 conversion, radial residual, 32 x 32 feature
   map and deterministic projections.
3. `microled_prepare_stability_only_profile.py` - common 8,192-candidate
   projection profile.
4. `microled_response.py` - response extraction and serialization helpers.
5. `microled_fuzzy_extractor.py` - the single 2,048-bit LDPC fuzzy extractor.

`microled_align.py` contains shared alignment utilities.

Run any entry point with `--help` to view its arguments.
