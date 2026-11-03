"""Forecast evaluation: level errors, directional accuracy, skill vs naive,
Diebold-Mariano significance, and quantile calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray, rate_t: np.ndarray) -> dict[str, float]:
    """``y_*`` are h-period log returns; ``rate_t`` is the level at forecast time."""
    y_true, y_pred, rate_t = map(np.asarray, (y_true, y_pred, rate_t))
    actual = rate_t * np.exp(y_true)
    pred = rate_t * np.exp(y_pred)
    err = pred - actual
    nz = (y_true != 0) & (y_pred != 0)
    da = float(np.mean(np.sign(y_true[nz]) == np.sign(y_pred[nz]))) if nz.any() else np.nan
    return {
        "rmse_usd": float(np.sqrt(np.mean(err**2))),
        "mae_usd": float(np.mean(np.abs(err))),
        "mape_pct": float(100 * np.mean(np.abs(err) / actual)),
        "rmse_logret": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "directional_acc": da,
        "n": int(len(y_true)),
    }


def majority_direction_rate(y_true: np.ndarray) -> float:
    """Directional accuracy of always calling the more frequent direction - the bar DA must clear."""
    y = np.asarray(y_true)
    y = y[y != 0]
    return float(max(np.mean(y > 0), np.mean(y < 0))) if len(y) else np.nan


def diebold_mariano(e_model: np.ndarray, e_bench: np.ndarray, horizon: int) -> tuple[float, float]:
    """DM test (squared loss) with Newey-West variance and the Harvey-Leybourne-Newbold
    small-sample correction. Negative stat = model better. Returns (stat, two-sided p).

    h-step forecasts overlap, so errors are autocorrelated up to lag h-1; ignoring
    that (a plain t-test) overstates significance.
    """
    d = np.asarray(e_model) ** 2 - np.asarray(e_bench) ** 2
    n = len(d)
    if n < 10 or np.allclose(d, 0):
        return 0.0, 1.0
    dbar = d.mean()
    dc = d - dbar
    lr_var = np.dot(dc, dc) / n
    for k in range(1, horizon):
        w = 1 - k / horizon
        lr_var += 2 * w * np.dot(dc[k:], dc[:-k]) / n
    if lr_var <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(lr_var / n)
    hln = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = float(dm * hln)
    p = float(2 * stats.t.sf(abs(stat), df=n - 1))
    return stat, p


def pinball(y: np.ndarray, q_pred: np.ndarray, alpha: float) -> float:
    diff = y - q_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1) * diff)))


def quantile_metrics(y_true: np.ndarray, q: np.ndarray, quantiles: list[float]) -> dict[str, float]:
    lo, hi = q[:, 0], q[:, -1]
    nominal = quantiles[-1] - quantiles[0]
    out = {
        "interval_nominal": nominal,
        "interval_coverage": float(np.mean((y_true >= lo) & (y_true <= hi))),
        "mean_interval_width_logret": float(np.mean(hi - lo)),
    }
    for i, a in enumerate(quantiles):
        out[f"pinball_q{int(a * 100)}"] = pinball(y_true, q[:, i], a)
    return out


def compare_models(y_true: np.ndarray, preds: dict[str, np.ndarray], rate_t: np.ndarray, horizon: int,
                   benchmark: str = "naive_last_value") -> pd.DataFrame:
    """One row per model: errors, skill vs benchmark (1 - RMSE/RMSE_bench), DM test vs benchmark."""
    rows = []
    bench_err = rate_t * np.exp(preds[benchmark]) - rate_t * np.exp(y_true)
    bench_rmse = np.sqrt(np.mean(bench_err**2))
    for name, p in preds.items():
        m = point_metrics(y_true, p, rate_t)
        err = rate_t * np.exp(p) - rate_t * np.exp(y_true)
        m["skill_vs_naive"] = float(1 - m["rmse_usd"] / bench_rmse)
        if name == benchmark:
            m["dm_stat"], m["dm_pvalue"] = np.nan, np.nan
        else:
            m["dm_stat"], m["dm_pvalue"] = diebold_mariano(err, bench_err, horizon)
        rows.append({"model": name, **m})
    return pd.DataFrame(rows).set_index("model").sort_values("rmse_usd")
