# Result definitions

The authoritative no-inter-device enrollment results are:

- `single_shot/summary_metrics.csv`: headline PUF/FE values;
- `single_shot/per_device_metrics.csv`: per-device reliability and tail BER;
- `single_shot/per_probe_metrics.csv`: one row per held-out image;
- `plot_data/`: Origin-ready uniformity and BER point tables;
- `roc_eer/`: authentication ROC/AUC/EER;
- `probe_to_probe_hd/`: output-code HD distributions used by the legacy
  combined intra/inter plot;
- `security/`: helper-data and claim-boundary audit.

Two inter-device quantities appear in the package:

1. `uniqueness_fixed_common_candidates_mean` evaluates all devices on the
   same frozen candidate coordinates and is the conventional
   common-coordinate physical comparison.
2. The `Inter` row in `probe_to_probe_hd/hamming_distance_summary.csv`
   compares final device output codes after device-local 2,048-position
   selection. It is useful for authentication-score separation but is not a
   common-coordinate physical uniqueness measurement.

Enrollment and probes are disjoint according to
`../models/enrollment_stability_only_manifest.json`. No probe or inter-device
HD is used to choose a device's 2,048 positions.
