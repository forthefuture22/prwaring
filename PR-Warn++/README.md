# PR-Warn++ v3.2

完整中文操作手册（从 SDWPF 下载、环境安装、预处理、训练、校准、
压力测试到每个源码/CLI 文件的职责）见
[`PR-Warn++_v3.2_代码使用说明.md`](docs/PR-Warn++_v3.2_代码使用说明.md)。

PR-Warn++ is a reproducible research implementation for **10–60 minute joint
probabilistic wind-power forecasting under extreme conditions and degraded
data**.  The implementation follows the local v3.2 research plan rather than
the removed legacy branches (VAE-first generation, pseudo OT-CFM, a strong
Jensen wake centre, or learned Level 1–4 warning labels).

The main chain is:

```text
raw SCADA
  -> rigid time grid + X_fill/M/delta_t/reason_code
  -> density-corrected empirical power curve P_pc
  -> dynamic multi-graph residual forecaster R_hat
  -> deterministic centre Y_det
  -> Direct Conditional Flow Matching on E = Y - Y_det
  -> joint turbine/horizon scenarios
  -> per-horizon split conformal intervals
  -> ramp/deviation/CVaR/OOD/data-quality proxies
```

## What is implemented now

- Leakage-aware chronological split and rigid-grid preprocessing.
- SDWPF official-style missing/unknown/abnormal reason codes.
- `X_fill`, observation mask `M`, and time-since-observation `delta_t`.
- Train-only density correction and monotone empirical power curve.
- Geographic, train-only correlation, directional and adaptive graph builders.
- A PyTorch dynamic multi-graph residual backbone.
- Controlled GRU, TCN, Graph-WaveNet-style and AGCRN-style deterministic
  baselines with the same cache contract.
- A PyTorch Direct CFM velocity model, masked CFM loss, reverse ODE sampler and
  an explicit minibatch optimal-transport coupling for A5.
- Graph-conditioned CVAE and VP-diffusion/DDIM joint residual baselines with a
  common deterministic-cache training and scenario-export CLI.
- A VAE-CFM implementation retained for the direct-CFM versus VAE-CFM A4
  ablation, with unified checkpoint loading and scenario regeneration.
- Graph-conditioned Gaussian, Student-t and quantile marginal models, pooled
  per-horizon CQR, plus controlled deep-ensemble export.
- Per-horizon split conformal calibration, chronological ACI, and empirical
  context-kernel calibration with conservative OOD fallback.
- Point, probabilistic, joint-distribution, ramp-event and risk-proxy metrics.
- Train-only Gaussian residual, residual-bootstrap and Gaussian-copula scenario
  baselines using the same Calib/Test artifact contract as Direct CFM.
- Reproducible MCAR, temporal-block, spatial-outage and extreme-conditioned
  missing-history stress mechanisms.
- Event-cluster bootstrap confidence intervals, Newey-West/HAC Diebold-Mariano
  comparison and multi-seed metric aggregation.
- Machine-readable A0--A11 ablations, G1--G6 joint probability benchmarks and
  B0--B9 point/marginal baselines with stable configuration-derived IDs (230
  five-seed jobs in the current matrix).
- Stress-safe A9 deterministic comparison runner and paired scenario reports
  with per-origin CRPS/ES/VS, HAC/DM and cluster/event bootstrap intervals.
- A9 probabilistic stress runner that regenerates scenarios from stressed
  conditions for G1--G6 and compares farm-CRPS degradation with common random
  numbers, HAC/DM and day-cluster bootstrap.
- CPU NumPy smoke tests plus a server-side PyTorch end-to-end smoke command.
- Raw SDWPF audit, real-data preprocessing, deterministic/CFM training, HDF5
  scenario export, Calib-only conformal fitting and frozen-Test evaluation CLIs.
- A11 oracle/issue-time weather attachment, Train-only standardization,
  latest-issue-at-or-before-origin selection and leakage audit; future-weather
  tensors are consumed by both the deterministic centre and cached scenario
  conditioner.
- A10 Calib-fitted/Test-frozen composition of event probability, OOD and
  online data-quality diagnostics with AUPRC, Brier and stylized-cost reports.

The large datasets and trained checkpoints are intentionally not committed.

## Environment

Server recommendation: Python 3.11, a PyTorch build matched to the server CUDA,
and the dependencies declared in `pyproject.toml`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

For a CUDA server, install the appropriate PyTorch wheel first, then install
this project.  Never silently change the CUDA wheel in a paper run.

## Local verification without a GPU

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m prwarn.cli.smoke_numpy
```

## Server smoke test

```powershell
$env:PYTHONPATH = "src"
python -m prwarn.cli.smoke_torch --device cuda
```

This uses synthetic data only and verifies tensor contracts:
`[B,N,L,D] -> [B,N,H] -> [M,B,N,H]`.

## Executable experiment pipeline

```bash
python -m prwarn.cli.audit_sdwpf \
  --input data/raw/sdwpf/sdwpf_full.parquet \
  --output outputs/audit/sdwpf_raw.json

python -m prwarn.cli.preprocess_sdwpf \
  --input data/raw/sdwpf/sdwpf_full.parquet \
  --locations data/raw/sdwpf/sdwpf_turb_location_elevation.csv \
  --output-dir data/processed/sdwpf_v3_2_history_only \
  --rated-power <RATED_POWER_IN_PATV_UNITS> \
  --pressure-unit pa --temperature-unit kelvin \
  --features Wspd Wdir Ndir Pab1 Pab2 Pab3 Prtv Patv Etmp Itmp Sp T2m

python -m prwarn.cli.train_deterministic \
  --data-dir data/processed/sdwpf_v3_2_history_only --device cuda

python -m prwarn.cli.train_deterministic \
  --data-dir data/processed/sdwpf_v3_2_history_only \
  --architecture gru --device cuda

python -m prwarn.cli.train_marginal_baseline --method quantile \
  --deterministic-run outputs/<DETERMINISTIC_RUN> --device cuda
python -m prwarn.cli.evaluate_marginal --run outputs/<MARGINAL_RUN>

python -m prwarn.cli.fit_probabilistic_baselines \
  --deterministic-run outputs/<DETERMINISTIC_RUN>

python -m prwarn.cli.train_neural_baseline --method cvae \
  --deterministic-run outputs/<DETERMINISTIC_RUN> --device cuda
python -m prwarn.cli.train_neural_baseline --method ddim \
  --deterministic-run outputs/<DETERMINISTIC_RUN> --device cuda
python -m prwarn.cli.train_neural_baseline --method vae_cfm \
  --deterministic-run outputs/<DETERMINISTIC_RUN> --device cuda

python -m prwarn.cli.train_flow \
  --deterministic-run outputs/<DETERMINISTIC_RUN> --device cuda

python -m prwarn.cli.evaluate --flow-run outputs/<FLOW_RUN>

# A7/A8 alternatives; each writes a separate evaluation directory.
python -m prwarn.cli.evaluate --flow-run outputs/<FLOW_RUN> \
  --calibration-method aci --output-dir outputs/<FLOW_RUN>/evaluation_aci
python -m prwarn.cli.evaluate --flow-run outputs/<FLOW_RUN> \
  --calibration-method context_fallback \
  --output-dir outputs/<FLOW_RUN>/evaluation_context

python -m prwarn.cli.make_experiment_matrix \
  --output outputs/plans/v3_2_matrix.json

python -m prwarn.cli.materialize_experiments \
  --matrix outputs/plans/v3_2_matrix.json \
  --output-dir outputs/plans/materialized_v3_2

python -m prwarn.cli.compare_scenario_runs \
  --run-a outputs/<CFM_RUN> --run-b outputs/<COPULA_RUN> \
  --label-a direct_cfm --label-b gaussian_copula \
  --output outputs/comparisons/cfm_vs_copula.json

python -m prwarn.cli.evaluate_probabilistic_stress \
  --data-dir data/processed/sdwpf_v3_2_history_only \
  --run direct_cfm=outputs/<CFM_RUN> \
  --run gaussian_copula=outputs/<COPULA_RUN> \
  --output outputs/comparisons/a9_probabilistic_stress.json --device cuda

python -m prwarn.cli.attach_future_weather \
  --input-dir data/processed/sdwpf_v3_2_history_only \
  --weather data/raw/weather/forecast_archive.parquet \
  --output-dir data/processed/sdwpf_v3_2_issue_time \
  --protocol issue_time_forecast --features wind_speed temperature pressure

python -m prwarn.cli.benchmark_inference \
  --data-dir data/processed/sdwpf_v3_2_history_only \
  --scenario-run outputs/<SCENARIO_RUN> \
  --output outputs/<SCENARIO_RUN>/benchmark.json --device cuda
```

Confirm the frozen dataset's coordinate column names, coordinate units, pressure
and temperature units, and rated power before preprocessing.  The code never
guesses these quantities from magnitude.

## Reproduction order

Do not jump directly to CFM training.

1. Freeze raw dataset versions and hashes.
2. Pass raw audit, official-mask, rigid-grid and no-leakage tests.
3. Fit and validate `P_pc` on Train-only normal observations.
4. Train deterministic baselines and the residual backbone.
5. Freeze/cache out-of-sample `Y_det` and `H_t`, then construct `E_true`.
6. Train Direct CFM and compare it fairly with strong probabilistic baselines.
7. Fit conformal calibration on the dedicated Calib split.
8. Evaluate continuous wind-side risk proxies and external evidence.

See `configs/sdwpf_v3_2.yaml` for the first executable protocol.  All future
weather fields must be assigned an availability protocol; future ERA5 is an
oracle-weather experiment, not online NWP.

## Project layout

```text
configs/                  experiment contracts
src/prwarn/data/          grid, masks, splits, windows, missing-data stresses
src/prwarn/baselines/     strong residual scenario baselines
src/prwarn/physics/       density and empirical power curve
src/prwarn/graphs/        explicit/adaptive graph construction
src/prwarn/models/        deterministic backbone and Direct CFM
src/prwarn/calibration/   conformal calibration
src/prwarn/risk/          continuous wind-side risk proxies
src/prwarn/eval/          metrics and dependent-sample statistical tests
src/prwarn/experiments/   canonical ablation/benchmark contracts
src/prwarn/cli/           smoke and future training entry points
tests/                    CPU contract and leakage tests
```

## Claim boundary

Outputs are wind-side operational/ramp/tail-risk **proxies**.  Without a grid
topology, load, dispatch and dynamic power-system model, they are not claims
about line overload, voltage/frequency stability, true reserve sufficiency or
actual dispatch cost.
