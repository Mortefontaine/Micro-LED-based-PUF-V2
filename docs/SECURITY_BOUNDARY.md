# Security, entropy and helper-data boundary

This implementation is a single-response optical weak PUF. Current and
temperature are operating conditions, not PUF challenges.

## Public information

The detector/STN weights, common template, deterministic projection bank,
stability-only candidate-universe profile, per-device support indices, LDPC construction and
helper manifest are public implementation material. Security must not rely on
their secrecy.

## Secret or local information

The enrolled 2,048-bit response, the reconstructed root key, manifest
authentication key and local protocol private keys are not part of this
release. `.gitignore` prevents accidental publication of `local_state/`,
`*.key` and generated private manifests.

## Fuzzy extractor statement

The code uses a regular LDPC secure sketch with 1,536 independent checks,
followed by HKDF-SHA256 and HMAC verification. The syndrome rank implies up to
1,536 bits of linear helper-data leakage and leaves a 512-dimensional coset
upper bound before accounting for source bias, correlation and public
selection metadata.

Consequently, this release supports reproducible robustness and helper-data
accounting, but it does not prove 128 bits of physical conditional
min-entropy. The derived 128-bit value should be described as stable key
material or an identity-bound seed, not as 128 experimentally certified bits
of PUF entropy.

## Empirical authentication scope

The closed-set experiment observed zero accepts among 29,160 ordered
non-target attempts. The attempt-level rule-of-three 95% upper bound is
0.010288%. Repeated frames are correlated observations of nine fabricated
devices, so this is not a population-level FAR bound.

All 81 enrollment-image replay attempts are accepted. A trusted sensor path,
liveness mechanism, protected acquisition interface or protocol-level
freshness binding is therefore required to resist digital image replay.

## Common-support control

The complete internal package retains a control that applies one fixed
2,048-position support to all devices. It is not duplicated into this compact
GitHub package. The main scheme uses the same extractor architecture and
candidate bank but device-local enrollment support selected without
inter-device information.
