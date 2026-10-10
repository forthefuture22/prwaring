# PR-Warn++ implementation status

## Code-complete research boundary

The repository now contains executable code for the planned in-repository
pipeline, not achieved paper results:

- leakage-safe audit, rigid-grid preprocessing, masks/delta-time, Train-only
  density/power-curve fitting and chronological Train/Val/Calib/Test windows;
- persistence and physical references; GRU, TCN, Graph-WaveNet-style,
  AGCRN-style and dynamic multi-graph point models; direct-power/residual,
  graph, density, correlation and missing-input ablations;
- Gaussian/Student-t/quantile marginal heads, per-horizon pooled CQR and a
  controlled deep ensemble;
- Gaussian residual, bootstrap, Gaussian copula, CVAE, DDIM, Direct CFM and
  VAE-CFM joint scenarios, including genuine minibatch OT assignment for A5;
- static conformal, predict-then-update ACI, context/OOD fallback, CRPS/ES/VS,
  dependence/ramp/reliability/tail metrics and dependent-sample statistics;
- deterministic and probabilistic A9 stresses with frozen selections/common
  random numbers; causal mean/forward/linear/GRU-D-style imputation carriers;
- A10 Calib-fitted/Test-frozen risk composition and A11 oracle/issue-time
  weather attachment with strict issue-time leakage gates;
- full inference latency/throughput/parameter/peak-VRAM benchmark code;
- a 230-job five-seed matrix covering A0--A11, G1--G6 and B0--B9, plus immutable
  config/command materialization.

Internal Graph WaveNet and AGCRN entries are explicitly **style
implementations** under a shared information/training contract. They are not
presented as exact reproductions of external author repositories. An original
IMFGCN comparison remains an external-method integration only if its exact code,
version and license are supplied.

## Locally verified

- Python compilation completed for `src` and `tests`.
- 52 dependency-light unit tests pass, including no-leakage, issue-time join,
  frozen calibration, scenario-baseline state and experiment-resolution tests.
- The NumPy end-to-end smoke pipeline passes.
- The generated matrices contain 230 unique, configuration-derived planned
  jobs and no achieved-result claim.

The local lightweight Python runtime does not contain PyTorch, HDF5, PyYAML or
the full project dependency set, so neural/HDF5 execution is intentionally not
reported as locally verified. Matrix materialization was also not executed in
that runtime because PyYAML is absent; its resolver is covered by unit tests.

## Requires server or frozen-data verification

1. Install the declared dependencies and run the CUDA tensor smoke.
2. Confirm frozen SDWPF schema, coordinates, units, rated power and hashes; run
   full audit/preprocessing and inspect physical-curve diagnostics.
3. Materialize the 230 jobs, then train/evaluate the selected five-seed
   point/marginal/joint methods and A0--A10 variants.
4. Run the real issue-time weather path only after a forecast archive with
   explicit issue/valid timestamps exists; run oracle ERA5 under its
   non-deployable label. This is the external-data blocker for A11.
5. Verify all neural forward/backward, HDF5 cache/checkpoint roundtrips,
   scenario sampling, calibration and stress regeneration on frozen artifacts.
6. Run paired HAC/DM and event/day-cluster bootstrap reports, five-seed
   aggregation, and declared-GPU P50/P95/P99/throughput/peak-VRAM benchmarks.

These are pending measurements and data checks, not pending algorithm code.
