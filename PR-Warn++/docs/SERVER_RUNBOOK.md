# PR-Warn++ v3.2 server runbook

This runbook turns the current code scaffold into auditable paper experiments.
It follows the eight Gates in the v3.2 plan.  Do not tune later stages on Test.

## 1. Freeze the environment

Record the GPU driver and install a CUDA-compatible PyTorch build explicitly.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the PyTorch wheel selected for the server CUDA first.
python -m pip install -e '.[data,dev]'
python -m prwarn.cli.smoke_torch --device cuda
```

Expected neural tensor contracts:

```text
X_fill/M/delta_t  [B,N,L,D]
Y_det             [B,N,H]
H_t               [B,N,d]
E_scen/Y_scen     [M,B,N,H]
P_farm_scen       [M,B,H]
```

Generate the frozen run contracts before launching jobs:

```bash
python -m prwarn.cli.make_experiment_matrix \
  --output outputs/plans/v3_2_matrix.json \
  --seeds 2025 2026 2027 2028 2029

python -m prwarn.cli.materialize_experiments \
  --matrix outputs/plans/v3_2_matrix.json \
  --base-config configs/sdwpf_v3_2.yaml \
  --output-dir outputs/plans/materialized_v3_2
```

The JSON marks every entry `planned_not_run`; only metrics read from completed
run artifacts may be reported as experimental evidence.

## 2. Gate 1 — data

Freeze exact dataset identifiers and SHA-256 hashes before preprocessing.

- SDWPF Figshare Version 2 is the main 134-turbine benchmark.
- Kelmarsh is external real-farm evidence; do not compare its absolute score to
  SDWPF as if node count and task were identical.
- CARE is a fault/data-quality stress dataset, not a synchronous 36-node graph.
- WIND Toolkit/WTK-LED is only needed for issue-time forecast/ensemble weather.

Required checks before training:

- duplicate `(TurbID, timestamp)` rows: zero;
- complete 10-minute rigid grid after reindexing;
- official `missing/unknown/abnormal` reason-code counts recorded;
- every feature has `availability_time` and `is_future_allowed` metadata;
- all Train/Val/Calib/Test timestamps and event IDs are disjoint;
- scaler, curve, correlation graph and OOD state are fitted on Train only.

Run the audit before preprocessing:

```bash
python -m prwarn.cli.audit_sdwpf \
  --input data/raw/sdwpf/sdwpf_full.parquet \
  --output outputs/audit/sdwpf_raw.json
```

Then build split-local windows. Replace the rated-power value and location
column names only after checking the frozen data dictionary:

```bash
python -m prwarn.cli.preprocess_sdwpf \
  --input data/raw/sdwpf/sdwpf_full.parquet \
  --locations data/raw/sdwpf/sdwpf_turb_location_elevation.csv \
  --output-dir data/processed/sdwpf_v3_2_history_only \
  --rated-power <RATED_POWER_IN_PATV_UNITS> \
  --pressure-unit pa --temperature-unit kelvin \
  --wind-direction-mode relative_plus_nacelle \
  --features Wspd Wdir Ndir Pab1 Pab2 Pab3 Prtv Patv Etmp Itmp Sp T2m
```

The archive stores standardized historical `x`, but leaves `y/current_y/P_pc`
in physical power units. Global origin-time wind direction is stored separately
before standardization, so the directional graph never consumes z-scores as
angles.

The main protocol is `history_only`.  Future ERA5/reanalysis belongs only to an
`oracle_weather` upper-bound experiment.  A deployable future-weather protocol
requires a forecast archive whose issue time is no later than forecast origin.

Audit and attach a true forecast archive before A11 training:

```bash
python -m prwarn.cli.audit_weather_archive \
  --input data/raw/weather/aligned_forecast.csv \
  --protocol issue_time_forecast \
  --output outputs/audit/issue_time_weather.json

python -m prwarn.cli.attach_future_weather \
  --input-dir data/processed/sdwpf_v3_2_history_only \
  --weather data/raw/weather/forecast_archive.parquet \
  --output-dir data/processed/sdwpf_v3_2_issue_time \
  --protocol issue_time_forecast \
  --features wind_speed temperature pressure
```

The attachment command fits weather scaling on Train only and selects the
latest issue no later than each origin. Oracle ERA5 uses the same tensor path
but remains marked non-deployable.

## 3. Gate 2 — weak physical centre

Use pressure in Pa and temperature in K:

```text
rho = pressure / (287.05 * temperature)
v_eq = wind_speed * (rho / 1.225) ** (1/3)
```

Fit `EmpiricalPowerCurve` only on Train observations with official valid/normal
status.  Save `curve.to_dict()` with the run.  Report standalone `P_pc` errors by
wind-speed bin, season and extreme/ramp regime before training a neural model.

## 4. Gate 3 — point forecasting

Start with persistence, GRU/TCN, Graph WaveNet, AGCRN and the existing IMFGCN
when available.  Then train `DynamicMultiGraphResidualForecaster` in stages:

```text
A_geo + A_adp
  -> + A_corr (raw and difference correlation are separate ablations)
  -> + A_dir
```

The target is `R = Y - P_pc`.  Use masked Huber/MAE only.  Save out-of-sample
`Y_det`, `H_t`, mixed adjacency and graph weights for every split.  Gate 3 passes
only if the centre is competitive overall and the claimed extreme subset gain
is repeatable.  If `A_dir` has no repeated extreme benefit, remove or demote it.

```bash
python -m prwarn.cli.train_deterministic \
  --data-dir data/processed/sdwpf_v3_2_history_only \
  --config configs/sdwpf_v3_2.yaml --device cuda
```

This writes the best checkpoint plus split-specific deterministic HDF5 caches.
The caches retain `Y_det`, `H_t`, graph gates and origin metadata without
materializing a full `[sample,134,134]` adjacency archive.

The repository provides controlled `gru`, `tcn`, `graphwavenet` and `agcrn`
architecture switches. The latter two are style implementations, so use an
external original repository only under a separately versioned label. Evaluate
persistence and `P_pc` from the identical frozen cache with
`prwarn.cli.evaluate_point_references`.

## 5. Gate 4 — Direct CFM

Construct the final error `E_true = Y - Y_det`, fit its normalization on Train,
and train `DirectCFMVelocity` with `conditional_flow_matching_loss`. The main
setting is standard independent-pair CFM. A5 separately enables the implemented
minibatch optimal-assignment coupling and must report its computational cost.

First run:

- Gaussian source;
- Euler, 8 reverse steps;
- `M=100` evaluation scenarios;
- the deterministic `H_t` and runtime-available context only.

Then run `steps={4,8,16,32}` quality/latency and `M={20,50,100,200}` Monte Carlo
convergence experiments.  Compare with at least Gaussian/Student-t, quantile or
CQR, Gaussian copula, CVAE, DDIM/diffusion and Direct CFM under equal information
and similar conditioner width/tuning budget.  Report CRPS, Energy Score,
Variogram Score, correlation diagnostics, Ramp-KS, parameter count, peak VRAM
and P50/P95/P99 latency.

```bash
python -m prwarn.cli.train_flow \
  --deterministic-run outputs/<DETERMINISTIC_RUN> \
  --config configs/sdwpf_v3_2.yaml --device cuda
```

Fit the three implemented Train-residual scenario baselines against the exact
same deterministic cache and export the same Calib/Test HDF5 contract:

```bash
python -m prwarn.cli.fit_probabilistic_baselines \
  --deterministic-run outputs/<DETERMINISTIC_RUN> \
  --methods gaussian_residual residual_bootstrap gaussian_copula \
  --scenarios 100
```

If `--fit-max-origins` is needed for covariance cost, use the same frozen cap
for all relevant methods and report it; the sampled Train indices are seeded
and the count is recorded in each run config.

Train the two neural joint-scenario competitors with the same cached `H_t`,
mixed graph, Train-only error normalization and scenario count:

```bash
python -m prwarn.cli.train_neural_baseline --method cvae \
  --deterministic-run outputs/<DETERMINISTIC_RUN> --device cuda

python -m prwarn.cli.train_neural_baseline --method ddim \
  --deterministic-run outputs/<DETERMINISTIC_RUN> --device cuda

python -m prwarn.cli.train_neural_baseline --method vae_cfm \
  --deterministic-run outputs/<DETERMINISTIC_RUN> --device cuda
```

G5 is a discrete VP epsilon-prediction diffusion model with a strided DDIM
sampler. Record both training diffusion steps and DDIM sampling steps; do not
compare latency without reporting them.

Train marginal Gaussian, Student-t and quantile/CQR references through the
shared deterministic cache:

```bash
python -m prwarn.cli.train_marginal_baseline --method quantile \
  --deterministic-run outputs/<DETERMINISTIC_RUN> --device cuda
python -m prwarn.cli.evaluate_marginal --run outputs/<MARGINAL_RUN>
```

For A5, `--coupling optimal_transport` invokes actual minibatch Hungarian
assignment through SciPy. Record its training-time overhead. For A6 and final
efficiency tables run `prwarn.cli.benchmark_inference` on the declared device.

Farm-level scenarios are saved by default. Add `--save-full-scenarios` only when
joint turbine diagnostics need them and storage has been budgeted explicitly.

## 6. Gate 5 — calibration

Use the dedicated Calib split.  Extract scenario quantiles and fit
`PerHorizonSplitConformal` on farm-level `[B,H]` intervals first.  Report PICP,
PINAW and Winkler by horizon and by normal/extreme subgroup.  Add ACI only after
the static anchor works; context kernels and worst-subgroup fallback are
empirical enhancements, not universal conditional-coverage guarantees.

```bash
python -m prwarn.cli.evaluate --flow-run outputs/<FLOW_RUN>

python -m prwarn.cli.evaluate --flow-run outputs/<FLOW_RUN> \
  --calibration-method aci --output-dir outputs/<FLOW_RUN>/evaluation_aci

python -m prwarn.cli.evaluate --flow-run outputs/<FLOW_RUN> \
  --calibration-method context_fallback \
  --output-dir outputs/<FLOW_RUN>/evaluation_context
```

This command fits corrections on `scenarios_calib.h5` only and evaluates the
frozen `scenarios_test.h5`; it also reports multi-threshold/multi-duration Ramp
Brier/AUPRC/reliability and CVaR summaries.

ACI follows strict chronological predict-then-update ordering on Test. The
context/fallback method only uses current power/availability and the
deterministic forecast path at issue time. It is an empirical A8 mechanism, not
a distribution-free conditional-coverage claim.

## 7. Gate 6 — risk proxies

Aggregate scenarios over turbines, then compute multiple ramp magnitudes,
durations and rates plus `CVaR_{0.90,0.95,0.99}`.  Future event labels are
evaluation targets only.  Report Brier/BSS, AUPRC, reliability, Event-F1, FNR,
lead time and event-level bootstrap confidence intervals.

`MahalanobisOOD` is only useful if increasing score predicts larger errors or
coverage degradation.  Data-quality risk is the controlled delta between clean
and deliberately degraded histories.  These are sensitivity/diagnostic results,
not causal effects.  Do not train Level 1–4 labels without independent operator
ground truth.

Build and evaluate A10 without fitting anything on Test:

```bash
python -m prwarn.cli.build_risk_ablation_inputs \
  --scenario-run outputs/<SCENARIO_RUN> \
  --deterministic-run outputs/<DETERMINISTIC_RUN> \
  --processed-data data/processed/sdwpf_v3_2_history_only \
  --output outputs/a10/risk_inputs.npz

python -m prwarn.cli.evaluate_risk_ablation \
  --input outputs/a10/risk_inputs.npz \
  --output outputs/a10/metrics.json
```

For A9, train the three structural variants separately, then reuse exactly the
same frozen missingness selections across runs:

```bash
python -m prwarn.cli.train_deterministic \
  --data-dir data/processed/sdwpf_v3_2_history_only \
  --experiment-id a9_x_only --missing-inputs --device cuda

python -m prwarn.cli.train_deterministic \
  --data-dir data/processed/sdwpf_v3_2_history_only \
  --experiment-id a9_x_mask --missing-inputs mask --device cuda

python -m prwarn.cli.train_deterministic \
  --data-dir data/processed/sdwpf_v3_2_history_only \
  --experiment-id a9_x_mask_delta --missing-inputs mask delta_t --device cuda

python -m prwarn.cli.evaluate_missing_stress \
  --data-dir data/processed/sdwpf_v3_2_history_only \
  --run x_only=outputs/a9_x_only \
  --run x_mask=outputs/a9_x_mask \
  --run x_mask_delta=outputs/a9_x_mask_delta \
  --output outputs/comparisons/a9_missing_stress.json --device cuda
```

The stress runner forward-fills only from still-visible history and rebuilds
`P_pc`, current power and directional-graph wind from the stressed carrier. It
never accepts forecast labels when rebuilding these issue-time quantities.

After the scenario checkpoints are frozen, regenerate rather than reuse clean
scenarios for probabilistic A9:

```bash
python -m prwarn.cli.evaluate_probabilistic_stress \
  --data-dir data/processed/sdwpf_v3_2_history_only \
  --run direct_cfm=outputs/<CFM_RUN> \
  --run cvae=outputs/<CVAE_RUN> \
  --run ddim=outputs/<DDIM_RUN> \
  --run gaussian_copula=outputs/<COPULA_RUN> \
  --output outputs/comparisons/a9_probabilistic_stress.json \
  --scenarios 100 --device cuda
```

The command reruns the deterministic centre/conditioner and the selected
scenario generator for every stress. Clean and stressed passes use common
random numbers. It reports point RMSE, turbine/farm CRPS, farm ES/VS and paired
farm-CRPS degradation with HAC/DM and day-cluster bootstrap. Residual baselines
must be regenerated once with the current code so `baseline_state.npz` exists.

Create paired frozen-Test reports and five-seed summaries with:

```bash
python -m prwarn.cli.compare_scenario_runs \
  --run-a outputs/<RUN_A> --run-b outputs/<RUN_B> \
  --label-a <METHOD_A> --label-b <METHOD_B> \
  --output outputs/comparisons/<A>_vs_<B>.json

python -m prwarn.cli.aggregate_seed_runs \
  --metrics outputs/seed*/metrics.json \
  --output outputs/summaries/five_seed_metrics.json
```

The paired report defines loss difference as A minus B: a positive value favors
B. CRPS/ES/VS use day-cluster bootstrap plus HAC/DM; ramp reports additionally
resample complete contiguous positive-event episodes.

## 8. Final evidence freeze

Run five fixed seeds only after configs and thresholds are frozen.  Every final
run directory must contain:

```text
config.json and config hash
git commit (or explicit unavailable marker)
dataset version and SHA-256
split boundaries and event manifest
weather protocol
best checkpoint
predictions/scenarios
metrics and runtime JSON
stdout/stderr log
```

Use event-level bootstrap for rare ramps and a dependence-aware test/HAC setup
for multi-step forecast comparisons.  Never treat turbine × horizon cells as
independent replicates.  If Direct CFM does not stably improve at least one
joint/ramp metric over a strong probabilistic baseline, keep the simpler model
and narrow the paper claim instead of adding modules.

## Current server-only verification boundary

The source compiles and its dependency-light contracts are locally tested. The
following evidence still has to come from the server or the frozen full data:

- CUDA PyTorch forward/backward smoke; point/marginal training; Direct-CFM and
  VAE-CFM reverse-ODE sampling; CVAE/diffusion-DDIM training and sampling;
- HDF5 deterministic caches and Calib/Test scenario export for all three
  residual baselines, CVAE, DDIM and Direct CFM;
- full A9 deterministic and probabilistic stress passes, including stressed
  scenario regeneration and paired degradation statistics;
- full SDWPF audit, exact column/coordinate definitions, pressure/temperature
  units and rated-power unit confirmation;
- five-seed result aggregation, event bootstrap/DM outputs on frozen Test, and
  A0--A11/G1--G6/B0--B9 execution from the materialized 230-job plan;
- A11 issue-time runs after a real archive with explicit issue/valid timestamps
  is frozen; until then only the attachment and leakage gates are verified;
- peak VRAM, throughput and P50/P95/P99 latency on the declared GPU.

These are pending measurements, not achieved results.
