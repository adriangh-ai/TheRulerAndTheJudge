# Changelog

## Unreleased

### Fixed

- Corrected discrete-grid pairwise comparison by allocating half of same-grid
  posterior mass to each direction (`P(A > B) + 0.5 P(A = B)`).
- Corrected False Separation Rate (FSR) accounting: the denominator is the number
  of human-indistinguishable pairs, and a false separation is counted only when
  the judge exceeds its separation threshold on one of those pairs.

### Refactored

- Split evaluation orchestration into focused modules while retaining the active
  production model and experiment defaults.
- Externalized paper settings into a validated JSON configuration and persisted
  the fully resolved configuration for every run.
- Propagated sampler, draw-selection, grid, threshold, and split seeds explicitly.
- Added deterministic explicit judge-file ordering, input schema validation, and
  file-content manifests without changing the legacy fallback order.
- Clarified the intended Monte Carlo and repeated-score likelihood semantics.
- Removed shadowed definitions from the production import path and retained
  superseded alternatives under `legacy/` for traceability.
- Centralized backend setup and delayed heavy probabilistic imports until after
  CLI parsing and input validation.
- Added result-comparison, numerical-contract, source-equivalence, configuration,
  and input-validation checks.
