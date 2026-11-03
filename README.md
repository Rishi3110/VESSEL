# VESSEL: dry-bulk freight rate forecasting

XGBoost pipeline that forecasts a freight rate *h* periods ahead and turns the
forecast into a fix-now-or-wait chartering decision. It includes features with
no look-ahead, purged walk-forward validation, honest benchmarks with
significance tests, P10/P50/P90 quantile forecasts, SHAP explanations, a dollar
backtest of the decision rule, and a self-contained HTML report.

```bash
pip install -e ".[dev]"
python -m vessel run -c configs/default.yaml      # ~1 min, writes reports/report.html
pytest -q                                         # 21 tests, includes point-in-time leakage tests
```

No real data ships with the repo: Baltic Exchange and Clarksons data are
licensed. The default config runs on a **built-in market simulator**
(`vessel/synthetic.py`). It models the China steel and iron-ore demand chain, a
fleet that responds to rates through orders and scrapping, a convex supply
curve, port congestion and canal restrictions. The report carries a banner
whenever it runs on synthetic data. To use real data, see
[`data/raw/README.md`](data/raw/README.md). You only change the config, not the
code.

## What a run produces (`reports/`)

| File | What it is |
|---|---|
| `report.html` | Single-file report covering the live signal, verdict, hold-out table, CV stability, decision backtest, SHAP and diagnostics |
| `latest_signal.json` | Forecast for the latest date: P10/P50/P90, expected move, FIX_NOW/WAIT, alert flag, top SHAP drivers. Feed this to Slack, email or a dashboard |
| `metrics_test.csv` | All models on the hold-out: RMSE/MAE/MAPE in $, directional accuracy, skill vs naive, Diebold-Mariano p-value |
| `metrics_cv.csv` | Per-fold walk-forward results, which show whether the edge is stable or comes from one lucky period |
| `decision_backtest.csv`, `band_breaches.csv` | Every decision with its $ outcome, and every week where the market broke out of the forecast band |
| `feature_importance.csv`, `plots/*.png` | SHAP and gain importance, plus charts |
| `model_point.json`, `model_quantile.json`, `best_params.json` | Production models refit on all data, and the tuned hyperparameters |

## Pipeline, mapped to the methodology

| Step | Where | Implementation |
|---|---|---|
| 1. Problem framing | `configs/default.yaml` | Target is the weekly (`W-FRI`) rate, forecast `horizon: 4` weeks ahead as a regression on the **4-week log return**. Direction is scored from the same output |
| 2. Data | `vessel/data.py` | One CSV per source, each with a per-source **publication lag** |
| 3. Preprocessing | `vessel/data.py`, `models.py` | Resample to the last value released in each bin, bounded forward-fill, ADF report, target winsorization fitted on the training fold only |
| 4. Features | `vessel/features.py` | 93 features: multi-scale returns, MA gaps, volatility, range position, z-scores, spike flag, exogenous changes and z-scores, utilisation proxy, demand-minus-fleet growth, orderbook/fleet, bunker cost share, sin/cos calendar, days to Chinese New Year, interactions |
| 5. Split | `vessel/split.py` | Chronological hold-out (last 104 weeks) and expanding walk-forward CV, **purged by `horizon`** |
| 6. Model | `vessel/models.py` | Untuned `XGBRegressor()` baseline, then Optuna TPE (40 trials) scored on walk-forward RMSE, early stopping on the tail of each training fold, refit, then a multi-quantile model |
| 7. Evaluation | `vessel/evaluate.py` | RMSE, MAE and MAPE in $; directional accuracy vs the majority-direction rate; skill vs naive; **Diebold-Mariano** (HAC + HLN); quantile coverage and pinball loss |
| 8. Interpretability | `vessel/explain.py` | SHAP TreeExplainer: global importance plus per-forecast drivers in the live signal |
| 9. Decision layer | `vessel/decision.py` | Charterer or owner view, FIX_NOW/WAIT on P50 vs a threshold (optionally confirmed by the band), backtest $ vs always fixing now, capture ratio vs perfect foresight, band-breach alerts |

Benchmarks run on identical rows: last value (naive), moving average,
momentum, seasonal naive, untuned XGBoost, and **ridge regression on the same
features**.

## Where the original methodology is wrong or incomplete

Each of these was either built in or caught during this build.

1. **"Consider differencing" is not optional for XGBoost.** Trees cannot
   predict outside the target range they were trained on. If you train on
   2012–2020 levels, the model cannot forecast a 2021 spike. The model must
   predict returns; the level is rebuilt as `rate_t * exp(ŷ)`.
2. **"rate_t-1, t-7, t-30" lags only work for one-step forecasts.** At a 4-week
   horizon the target must be `rate_{t+h}` with every feature at or before `t`.
   Here, features are anchored at `t` and the target is shifted forward.
3. **Interpolation is leakage.** Interpolating a gap uses the *next*
   observation. Use forward-fill only, with a limit (`max_ffill_periods`).
4. **Publication lag is missing, and it is the largest silent leak.** China
   steel output for March is not known on 31 March; it is released around 15
   April. Stamping monthly data at period-end makes backtests look prophetic.
   Every source here carries `publication_lag_days`.
5. **`TimeSeriesSplit` with default settings still leaks.** With overlapping
   h-step targets, the last *h* training labels contain test-period prices. A
   purge gap of `horizon` is applied everywhere.
6. **"Early stopping with a validation set" must not use the test set.** The
   early-stopping slice is taken from the end of each training fold. The
   hold-out is touched once.
7. **Winsorizing features does nothing for trees,** because splits are
   invariant to monotone transforms. Outlier handling matters only for the
   target's loss. Clip the training target, never the evaluation target, and
   never delete spikes: they are the market.
8. **Day-of-week is noise at weekly frequency** (it is constant), so it is off
   by default.
9. **Directional accuracy needs a bar.** 60% means nothing if the market rose
   in 60% of weeks. The pipeline reports the majority-direction rate next to
   it. The naive forecast has no direction, so its accuracy is undefined, not
   50%.
10. **"Beats the naive baseline" needs a significance test.** On 104 weekly
    points, a 3% RMSE gap is noise. Diebold-Mariano with HAC variance (the
    overlapping errors are autocorrelated) separates real skill from luck.
11. **No linear benchmark.** If ridge on the same features matches XGBoost, use
    ridge, since it is simpler, more stable, and easier for the desk to trust.
    A related trap surfaced during this build: sklearn's `RidgeCV` defaults to
    leave-one-out CV, which leaks through autocorrelated targets. It picked
    alpha=0.1 and gave 27% MAPE. With purged time-series CV it picks alpha≈2200
    and matches naive.
12. **Point forecasts are the wrong input for the decision layer.** "Lock in
    now vs wait" depends on the downside, so the model emits P10/P50/P90. The
    rule is judged in dollars (P&L vs always fixing now, capture ratio, worst
    loss), not RMSE.
13. **Fleet and orderbook data are slow-moving.** They define the cycle over
    years and carry little information for a 4-week move. Expect momentum,
    volatility, congestion and seasonality to dominate at short horizons, and
    supply-demand balance to matter more at a 3–12-month horizon.

## Results on the synthetic data (default config)

Hold-out: Oct 2024 to Oct 2026, 104 weekly forecasts, 4 weeks ahead.

| Model | RMSE $/day | MAE $/day | MAPE | Dir. acc. | DM p vs naive |
|---|---|---|---|---|---|
| naive last value | 956 | 777 | 9.2% | – | – |
| ridge | 957 | 746 | 8.9% | 59.6% | 0.98 |
| XGBoost P50 (quantile) | 958 | 727 | 8.7% | 63.5% | 0.95 |
| XGBoost tuned | 1004 | **714** | **8.7%** | **66.3%** | 0.61 |
| XGBoost default | 1112 | 790 | 9.5% | 65.4% | 0.31 |

The majority-direction rate is 51.9%. In plain terms: XGBoost does **not** beat
naive on RMSE, but it has a real directional edge (66% vs 52%) and the lowest
MAE. Tuning was worth it (RMSE 1112 → 1004). The decision rule waited 3 times,
all correctly, and saved $110k vs always fixing now. That is only 8% of what
perfect foresight would have captured, because the rule is conservative by
design. These are simulator numbers; on real data, rerun the pipeline and read
its verdict section, which is generated from the metrics and never hand-edited.

## Layout

```
configs/default.yaml   every knob: data sources and lags, features, splits, tuning, decision rule
vessel/
  config.py            typed config; unknown keys fail loudly
  synthetic.py         market simulator (demo and tests)
  data.py              load, publication lag, resample, bounded ffill, ADF
  features.py          leakage-safe feature builder and target
  split.py             purged hold-out and walk-forward splits
  models.py            XGBoost point and quantile, Optuna tuning, ridge, naive baselines
  evaluate.py          metrics, Diebold-Mariano, quantile calibration
  decision.py          FIX_NOW/WAIT signals, $ backtest, band breaches, live signal
  explain.py           SHAP
  plots.py, report.py  charts and HTML report
  pipeline.py, cli.py  orchestration
tests/                 leakage (raw data truncated at past dates), splits, metrics, decisions, end-to-end
```

## Next steps worth doing with real data

- Put the forecast horizon on the decision axis: run `--horizon 1/4/13` and compare each horizon's skill against naive.
- Add forward freight agreement (FFA) curves as features. The FFA curve is the market's own forecast, and the real question is whether the model beats it.
- Schedule `python -m vessel run` weekly (cron or CI) and post `latest_signal.json` to the chartering channel.
