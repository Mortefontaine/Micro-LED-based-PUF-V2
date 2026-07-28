# Enrollment without inter-device Hamming-distance optimization

## Experimental question

Can enrollment select the 2,048 response bits for each micro-LED using only
that device's own enrollment observations, without using inter-device Hamming
distance or response balance across M1-M9?

## Controlled enrollment rule

- The optical preprocessing and the fixed bank of 8,192 sparse projections are
  shared by all devices and frozen before evaluation.
- For each device, nine enrollment images are used to estimate only:
  1. within-device bit stability; and
  2. projection margin.
- The 2,048 most stable, high-margin positions are frozen for that device.
- No response value, balance statistic, or Hamming distance from another
  device is available to the selector.
- The selected support is treated as public enrollment metadata rather than a
  secret.
- All 405 images in the enrollment pool are excluded from the 3,645 probe
  images.

This is a reliability-driven device-specific enrollment procedure. It is not
an inter-device discriminative optimization.

## Quantitative comparison

| Metric | Population-balanced baseline | Stability-only, no inter-HD | Direction |
|---|---:|---:|---|
| Mean probe-to-reference intra-HD | 0.3509% | 0.2171% | improved |
| Reliability, 100% - mean intra-HD | 99.6491% | 99.7829% | improved |
| 95th-percentile intra-HD | 1.8945% | 1.0742% | improved |
| 99th-percentile intra-HD | 5.5449% | 4.1504% | improved |
| Mean uniformity | 50.0163% | 50.1139% | both close to 50% |
| Mean bit aliasing | 49.9715% | 50.1506% | both close to 50% |
| FE exact recovery | 100% | 100% | preserved |
| Quality-gate rejects | 0 / 3,645 | 0 / 3,645 | preserved |
| Probe-pair intra-HD, old definition | 0.7014% | 0.4410% | improved |
| Probe-pair inter-HD, old definition | 49.7931% | 49.9468% | closer to 50% |
| Strict probe-pair HD gap | 25.6348 percentage points | 30.2246 percentage points | improved |
| Closed-set AUC | 1.000 | 1.000 | preserved |
| Empirical EER | 0 | 0 | preserved |
| Maximum genuine / minimum impostor HD | 19.0430% / 29.2480% | 16.5527% / 27.1484% | separation preserved |
| Zero-error score gap | 10.2051 percentage points | 10.5957 percentage points | slightly improved |

The old-definition inter-HD compares the final ordered 2,048-bit output codes
of different devices. Because the selected physical supports differ, it is an
output-code separation metric and must not be described as common-coordinate
physical uniqueness.

As an additional support-independent control, the enrollment templates were
also compared on the common fixed bank of all 8,192 projections. The observed
mean inter-device HD is 53.8615% (range 41.0645%-62.8296%). Unlike the previous
4-of-9/5-of-9 population-balance filter, this value was not imposed by the
selector.

## Security and selection-bias audit

- Candidate-bank population balance across the nine devices: mean 50.1506%.
- Descriptive per-bit min-entropy across nine devices: mean 0.7745 bit.
  This is a small-sample descriptive statistic, not a population entropy bound.
- Mean Jaccard overlap between the selected supports of two devices: 0.1607.
- Cross-device impostor trials: 0 accepted out of 29,160 attempts.
- Rule-of-three 95% upper bound from the observed zero events: 1.03e-4 per
  tested image attempt.
- Enrollment failures: 0 out of 9 devices.

The digital enrollment-image replay check accepts all 81 enrollment images, as
expected. Therefore, the result does not by itself address compromise or replay
of stored raw enrollment imagery; only helper data and frozen support metadata
should be retained in the intended deployment.

## Recommended interpretation

This stability-only enrollment is the more natural primary formulation:

> A global measurement and projection bank is fixed for all devices. During
> enrollment, each device independently freezes a reliability map containing
> its 2,048 most repeatable high-margin response positions. No inter-device
> Hamming-distance information is used in this selection. Subsequent
> measurements regenerate a noisy response on the frozen support, and the
> fuzzy extractor reconstructs the same 128-bit value using public helper
> data.

Inter-device HD, ROC/AUC, EER, uniqueness, and bit aliasing should be presented
strictly as held-out evaluation results, not as enrollment objectives. This
separates reliability engineering from identity evaluation and avoids the
appearance that M1-M9 were optimized against one another.

## Scope of claims

- The zero EER and AUC of 1 are descriptive closed-set results for nine physical
  devices and correlated repeated images; they are not population-level bounds.
- The device-specific selected support is part of public enrollment metadata.
  It should not be counted as secret entropy or used to inflate the 128-bit key
  claim.
- The LDPC syndrome reveals up to its rank (1,536 bits). A formal extractor
  security claim still requires a conservative lower bound on conditional
  min-entropy after all public helper data, not only uniformity or inter-HD.
