# Data contract

Raw datasets are not versioned in this repository.  Store immutable downloads
under `data/raw/<dataset>/` and record their DOI/version, byte size and SHA-256.

Processed `*.npz` split archives produced by `prwarn.cli.preprocess_sdwpf`
contain:

- `x`, `mask`, `delta_t`: `[sample,node,history,feature]`;
- `y`, `y_mask`, `p_pc`: `[sample,node,horizon]`;
- `current_y`, `current_y_mask`: `[sample,node]` at forecast origin;
- `wind_from`: `[sample,node]` origin-time global meteorological direction in degrees;
- `origin_time`: forecast origins;
- `turbines`: node order.

Archives produced by `prwarn.cli.attach_future_weather` additionally contain:

- `future_weather`, `future_weather_mask`: `[sample,node,horizon,weather_feature]`;
- Train-only weather mean/std, archive provenance, match coverage and leakage
  audit in `metadata.json`.

The preprocessor persists origin-time `P_pc` across the forecast horizon in the
`history_only` protocol.  It never reads realized future wind or ERA5.  Future
reanalysis may only be introduced by a separately named `oracle_weather`
experiment, while deployable future covariates require issue-time forecast data.
For the deployable protocol, the join selects the latest archive issue no later
than each forecast origin; a later issue is never accepted as a numerical fill.

`x` is standardized with per-feature Train-only mean/std saved in
`metadata.json`; `y`, `current_y`, and `p_pc` remain in physical power units.
