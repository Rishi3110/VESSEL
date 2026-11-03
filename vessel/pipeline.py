"""End-to-end run: data -> features -> tune -> CV -> hold-out test -> SHAP ->
decision backtest -> production refit -> live signal -> report."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vessel import plots
from vessel.config import Config
from vessel.data import load_panel, stationarity_report
from vessel.decision import backtest, band_breaches, live_signal
from vessel.evaluate import compare_models, majority_direction_rate, point_metrics, quantile_metrics
from vessel.explain import explain_row, importance_table, shap_values
from vessel.features import build_features
from vessel.models import (
    TargetWinsorizer, baseline_predictions, fit_ridge, fit_xgb, fit_xgb_default,
    predict_quantiles, tune, _xgb,
)
from vessel.report import write_report
from vessel.split import holdout_split, inner_val_split, walk_forward

log = logging.getLogger(__name__)


def cross_validate(X: pd.DataFrame, y: np.ndarray, rate: pd.Series, base: pd.DataFrame, params: dict, cfg: Config) -> pd.DataFrame:
    """Score tuned XGB, ridge and baselines on the same walk-forward folds of the dev set."""
    s, m = cfg.split, cfg.model
    rows = []
    for k, (tr, te) in enumerate(walk_forward(len(X), s.cv_splits, s.cv_test_periods, cfg.horizon), start=1):
        fit_idx, val_idx = inner_val_split(tr, s.val_periods, cfg.horizon)
        w = TargetWinsorizer(m.winsorize_target).fit(y[fit_idx])
        xg = _xgb(params, m.max_estimators, m.seed, None, m.early_stopping_rounds)
        xg.fit(X.iloc[fit_idx], w.transform(y[fit_idx]), eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=False)
        w_tr = TargetWinsorizer(m.winsorize_target).fit(y[tr])
        ridge = fit_ridge(X.iloc[tr], w_tr.transform(y[tr]), cfg.horizon)
        preds = {"xgb_tuned": xg.predict(X.iloc[te]), "ridge": ridge.predict(X.iloc[te])}
        for c in base.columns:
            preds[c] = base[c].iloc[te].fillna(0.0).to_numpy()
        rt = rate.iloc[te].to_numpy()
        for name, p in preds.items():
            rows.append({"fold": k, "model": name, "start": X.index[te[0]].date(), "end": X.index[te[-1]].date(),
                         **point_metrics(y[te], p, rt)})
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame, cv: pd.DataFrame, majority_da: float) -> list[str]:
    """Plain-language conclusions, generated from the numbers so they cannot be spun."""
    out = []
    xg = table.loc["xgb_tuned"]
    if xg["skill_vs_naive"] > 0 and xg["dm_pvalue"] < 0.05:
        out.append(f"Tuned XGBoost beats the naive last-value forecast by {100 * xg['skill_vs_naive']:.1f}% RMSE "
                   f"and the gap is statistically significant (DM p={xg['dm_pvalue']:.3f}).")
    elif xg["skill_vs_naive"] > 0:
        out.append(f"Tuned XGBoost is {100 * xg['skill_vs_naive']:.1f}% better than naive on RMSE, but the difference is NOT "
                   f"statistically significant (DM p={xg['dm_pvalue']:.3f}). Treat the edge as unproven.")
    else:
        out.append(f"Tuned XGBoost does NOT beat the naive last-value forecast on the hold-out "
                   f"(skill {100 * xg['skill_vs_naive']:.1f}%). Do not deploy it for point forecasts.")
    best = table.index[0]
    if best != "xgb_tuned":
        out.append(f"Best model on hold-out RMSE is '{best}', not XGBoost.")
    naive_mae = table.loc["naive_last_value", "mae_usd"]
    if xg["mae_usd"] < naive_mae:
        out.append(f"On MAE XGBoost is better than naive (${xg['mae_usd']:,.0f} vs ${naive_mae:,.0f}/day): it is right more often "
                   "but its misses are larger - RMSE punishes those. Check whether your desk's loss is closer to MAE or RMSE.")
    if table.loc["ridge", "rmse_usd"] <= xg["rmse_usd"]:
        out.append("A linear ridge model on the same features matches or beats XGBoost on RMSE - the non-linearity is not paying for itself there.")
    if xg["directional_acc"] <= majority_da:
        out.append(f"Directional accuracy {100 * xg['directional_acc']:.1f}% does not beat always calling the majority "
                   f"direction ({100 * majority_da:.1f}%).")
    else:
        out.append(f"Directional accuracy {100 * xg['directional_acc']:.1f}% vs {100 * majority_da:.1f}% for always calling the majority direction.")
    piv = cv.pivot_table(index="fold", columns="model", values="rmse_usd")
    wins = int((piv["xgb_tuned"] < piv["naive_last_value"]).sum())
    out.append(f"XGBoost beat naive in {wins} of {len(piv)} walk-forward folds (stability check).")
    return out


def run(cfg: Config) -> dict:
    out = Path(cfg.output_dir)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    h, tcol, freq = cfg.horizon, cfg.data.target_col, cfg.data.freq

    # 1-3. data, alignment, diagnostics
    panel = load_panel(cfg)
    stat = stationarity_report(panel[tcol])
    X, y, rate = build_features(panel, cfg)
    labelled = y.notna()
    XL, yL, rL = X[labelled], y[labelled].to_numpy(), rate[labelled]
    log.info("Panel %s rows, %d features, %d labelled rows", len(panel), X.shape[1], len(XL))

    # 5. chronological hold-out with purge gap
    dev, test = holdout_split(len(XL), cfg.split.test_periods, h)
    Xd, yd, rd = XL.iloc[dev], yL[dev], rL.iloc[dev]
    Xt, yt, rt = XL.iloc[test], yL[test], rL.iloc[test].to_numpy()
    base_all = baseline_predictions(rate, h, freq, cfg.features.rolling_windows[0])

    # 6. tuning on dev only, then CV comparison on dev folds
    tuned = tune(Xd, yd, cfg)
    cv = cross_validate(Xd, yd, rd, base_all.loc[Xd.index], tuned.params, cfg)

    # final fits on full dev
    xgb_point, n_point = fit_xgb(Xd, yd, tuned.params, cfg)
    xgb_q, n_q = fit_xgb(Xd, yd, tuned.params, cfg, cfg.model.quantiles)
    xgb_def = fit_xgb_default(Xd, yd, cfg.model.seed)
    ridge = fit_ridge(Xd, TargetWinsorizer(cfg.model.winsorize_target).fit(yd).transform(yd), h)

    # 7. hold-out evaluation (touched once)
    q_test = predict_quantiles(xgb_q, Xt)
    preds = {"xgb_tuned": xgb_point.predict(Xt), "xgb_default": xgb_def.predict(Xt),
             "xgb_quantile_p50": q_test[:, 1], "ridge": ridge.predict(Xt)}
    for c in base_all.columns:
        preds[c] = base_all.loc[Xt.index, c].fillna(0.0).to_numpy()
    table = compare_models(yt, preds, rt, h)
    qm = quantile_metrics(yt, q_test, cfg.model.quantiles)
    maj = majority_direction_rate(yt)

    # 8. interpretability
    sv, base_val = shap_values(xgb_point, Xt)
    imp = importance_table(xgb_point, Xt, sv)

    # 9. decision layer
    bt, bt_sum = backtest(Xt.index, rt, rt * np.exp(yt), q_test, cfg.decision)
    breaches = band_breaches(bt)

    # production refit on every labelled row -> live forecast for the latest date
    prod_point, _ = fit_xgb(XL, yL, tuned.params, cfg)
    prod_q, _ = fit_xgb(XL, yL, tuned.params, cfg, cfg.model.quantiles)
    x_last = X.iloc[[-1]]
    q_live = predict_quantiles(prod_q, x_last)[0]
    live = live_signal(X.index[-1], float(rate.iloc[-1]), q_live, cfg.decision, h, freq)
    sv_live, base_live = shap_values(prod_point, x_last)
    drivers = explain_row(sv_live[0], x_last.iloc[0], base_live)
    live["top_drivers"] = [{"feature": f, "value": None if pd.isna(r.value) else round(float(r.value), 4),
                            "shap": round(float(r.shap), 4)} for f, r in drivers.iterrows()]

    conclusions = verdict(table, cv, maj)

    # ---- persist
    stat.to_csv(out / "stationarity.csv", index=False)
    table.to_csv(out / "metrics_test.csv")
    cv.to_csv(out / "metrics_cv.csv", index=False)
    imp.to_csv(out / "feature_importance.csv")
    bt.to_csv(out / "decision_backtest.csv")
    breaches.to_csv(out / "band_breaches.csv")
    pd.DataFrame({"date": Xt.index, "rate_t": rt, "actual_logret": yt, **{f"pred_{k}": v for k, v in preds.items()},
                  "q_lo": q_test[:, 0], "q_hi": q_test[:, 2]}).to_csv(out / "predictions_test.csv", index=False)
    if not tuned.trials.empty:
        tuned.trials.to_csv(out / "optuna_trials.csv", index=False)
    (out / "best_params.json").write_text(json.dumps({**tuned.params, "n_estimators_point": n_point,
                                                      "n_estimators_quantile": n_q, "cv_rmse_logret": tuned.cv_rmse}, indent=2))
    (out / "latest_signal.json").write_text(json.dumps(live, indent=2))
    prod_point.save_model(out / "model_point.json")
    prod_q.save_model(out / "model_quantile.json")

    figs = {
        "history": plots.history(rate, Xt.index[0], out / "plots" / "history.png"),
        "forecast": plots.forecast_band(bt, h, freq, cfg.model.quantiles, out / "plots" / "forecast_band.png"),
        "cv": plots.cv_folds(cv, out / "plots" / "cv_folds.png"),
        "importance": plots.importance(imp, out / "plots" / "importance.png"),
        "beeswarm": plots.shap_beeswarm(sv, Xt, out / "plots" / "shap_beeswarm.png"),
        "decision": plots.decision_pnl(bt, out / "plots" / "decision_pnl.png"),
    }
    summary = {
        "data_source": cfg.data.source, "rows": len(panel), "features": X.shape[1],
        "dev_period": [str(Xd.index[0].date()), str(Xd.index[-1].date())],
        "test_period": [str(Xt.index[0].date()), str(Xt.index[-1].date())],
        "horizon": h, "freq": freq, "conclusions": conclusions, "quantile_metrics": qm,
        "majority_direction_rate": maj, "decision": bt_sum, "live": live, "params": tuned.params,
        "n_estimators": n_point,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_report(out / "report.html", cfg, summary, stat, table, cv, imp, breaches, figs)
    return summary
