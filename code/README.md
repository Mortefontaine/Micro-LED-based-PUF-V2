# Final code path

The authoritative runtime path is:

1. `microled_expanded_align.py` - YOLO localization and expanded-background STN alignment.
2. `microled_puf.py` - Rec.709 conversion, radial residual, 32x32 feature map and deterministic projections.
3. `microled_prepare_stability_only_profile.py` - freezes the common
   8,192-candidate universe without population balance or inter-device
   separation.
4. `microled_single_shot_key.py` - device-specific enrollment, 2,048-bit response, LDPC reconstruction, HKDF and HMAC.
5. `microled_eval_single_shot.py` - normal-temperature one-image evaluation.
6. `microled_eval_security.py` - ordered cross-device and replay diagnostics.
7. `microled_export_uniformity_ber_plot_data.py` and
   `microled_export_probe_to_probe_hd.py` - Origin-ready figure data.

The main deployment interface is `microled_single_shot_key.py`.
Visualization/export scripts do not define the PUF and are downstream of the
frozen evaluation outputs.
