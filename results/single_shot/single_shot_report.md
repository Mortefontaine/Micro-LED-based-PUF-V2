# Single-Image PUF Key-Regeneration Evaluation

Enrollment predeclares 405 pool images and selects 81 images (nine per device). Every pool image is excluded from testing; all remaining aligned images are one-shot key-regeneration attempts. The enrolled response and root key are not stored.

## Overall

- Attempts: 3645
- Exact key reconstructions: 3645 (100.00%)
- Failed closed: 0
- Pre-bit quality-gate rejections: 0 (0.00%)
- Decoder failures after quality gate: 0
- Qualified-image exact key rate: 100.000%
- Immediate next-frame recovery after a failed attempt: 0 / 0 (-)
- Qualified intra-HD samples (quality gate passed): 3645
- Qualified intra-HD mean / p95 / p99 / max: 0.0022 / 0.0107 / 0.0415 / 0.1655
- All-probe raw-HD audit samples (includes quality rejects): 3645

## Per Device

| Device | Accepted | Attempts | First-shot success | Gate reject | Qualified success | Raw HD p99 |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 405 | 405 | 100.00% | 0 | 100.000% | 0.0238 |
| M2 | 405 | 405 | 100.00% | 0 | 100.000% | 0.0280 |
| M3 | 405 | 405 | 100.00% | 0 | 100.000% | 0.0137 |
| M4 | 405 | 405 | 100.00% | 0 | 100.000% | 0.0390 |
| M5 | 405 | 405 | 100.00% | 0 | 100.000% | 0.0282 |
| M6 | 405 | 405 | 100.00% | 0 | 100.000% | 0.0107 |
| M7 | 405 | 405 | 100.00% | 0 | 100.000% | 0.0327 |
| M8 | 405 | 405 | 100.00% | 0 | 100.000% | 0.0161 |
| M9 | 405 | 405 | 100.00% | 0 | 100.000% | 0.0892 |

## Success by Raw HD

| Raw HD interval | Accepted | Attempts | Success |
|---|---:|---:|---:|
| 0.00-0.02 | 3549 | 3549 | 100.00% |
| 0.02-0.05 | 68 | 68 | 100.00% |
| 0.05-0.10 | 24 | 24 | 100.00% |
| 0.10-0.20 | 4 | 4 | 100.00% |
| 0.20-1.01 | 0 | 0 | - |

## Failed-Closed Frames

| Device | Condition | Frame | Stage | Template corr | Raw HD |
|---|---|---|---|---:|---:|

## Interpretation

An accepted attempt reproduced the enrolled key exactly and passed the HMAC check. A decoder failure or wrong candidate key is rejected; the implementation never releases a different key. This is a research decoder result, not a production cryptographic certification or a 128-bit min-entropy proof.
