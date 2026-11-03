"""Turn forecasts into chartering actions and price them in dollars.

Decision at time t, for a fixture that must be done within the next h periods:
FIX_NOW (lock today's rate) or WAIT (fix at the rate h periods later).

* charterer pays freight -> WAIT only if rates are expected to FALL by > threshold
* owner earns freight    -> WAIT only if rates are expected to RISE by > threshold
Default when the signal is weak is FIX_NOW (the cargo/ship has to move anyway).

With ``require_confidence`` the quantile band must agree too (e.g. a charterer
waits only if even the P90 forecast is a fall). The backtest reports dollars
saved vs always-fix-now and the share of the perfect-foresight saving captured -
the number a chartering desk actually cares about, not RMSE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vessel.config import DecisionConfig


def signals(q: np.ndarray, cfg: DecisionConfig) -> np.ndarray:
    """q: (n, 3) quantile forecasts of the h-period log return (P_lo, P50, P_hi)."""
    exp_move = np.exp(q[:, 1]) - 1
    if cfg.perspective == "charterer":
        wait = exp_move < -cfg.threshold
        if cfg.require_confidence:
            wait &= q[:, 2] < 0
    else:
        wait = exp_move > cfg.threshold
        if cfg.require_confidence:
            wait &= q[:, 0] > 0
    return np.where(wait, "WAIT", "FIX_NOW")


def backtest(dates: pd.Index, rate_now: np.ndarray, rate_later: np.ndarray, q: np.ndarray, cfg: DecisionConfig) -> tuple[pd.DataFrame, dict]:
    sig = signals(q, cfg)
    sign = 1.0 if cfg.perspective == "charterer" else -1.0
    # $/day gained by waiting instead of fixing now (charterer: pay less; owner: earn more)
    gain_if_wait = sign * (rate_now - rate_later)
    realised = np.where(sig == "WAIT", gain_if_wait, 0.0) * cfg.charter_days
    perfect = np.maximum(gain_if_wait, 0.0) * cfg.charter_days
    df = pd.DataFrame({
        "rate_now": rate_now, "rate_later": rate_later,
        "p_lo": rate_now * np.exp(q[:, 0]), "p50": rate_now * np.exp(q[:, 1]), "p_hi": rate_now * np.exp(q[:, 2]),
        "signal": sig, "pnl_vs_fix_now_usd": realised, "perfect_foresight_usd": perfect,
    }, index=dates)
    waits = sig == "WAIT"
    total_perfect = perfect.sum()
    summary = {
        "perspective": cfg.perspective,
        "decisions": int(len(sig)),
        "wait_signals": int(waits.sum()),
        "wait_hit_rate": float(np.mean(gain_if_wait[waits] > 0)) if waits.any() else np.nan,
        "total_pnl_vs_fix_now_usd": float(realised.sum()),
        "avg_pnl_per_wait_usd": float(realised[waits].mean()) if waits.any() else 0.0,
        "perfect_foresight_usd": float(total_perfect),
        "capture_ratio": float(realised.sum() / total_perfect) if total_perfect > 0 else np.nan,
        "worst_wait_usd": float(realised[waits].min()) if waits.any() else 0.0,
    }
    return df, summary


def band_breaches(bt: pd.DataFrame) -> pd.DataFrame:
    """Forecasts whose realised rate landed outside the P_lo-P_hi band: the market did
    something the model did not expect - review these with the desk."""
    out = bt[(bt["rate_later"] < bt["p_lo"]) | (bt["rate_later"] > bt["p_hi"])].copy()
    out["breach"] = np.where(out["rate_later"] > out["p_hi"], "ABOVE_BAND", "BELOW_BAND")
    return out


def live_signal(as_of: pd.Timestamp, rate_now: float, q: np.ndarray, cfg: DecisionConfig, horizon: int, freq: str) -> dict:
    sig = signals(q[None, :], cfg)[0]
    move = float(np.exp(q[1]) - 1)
    return {
        "as_of": str(as_of.date()),
        "horizon_periods": horizon,
        "freq": freq,
        "rate_now": round(float(rate_now), 0),
        "forecast_p_lo": round(float(rate_now * np.exp(q[0])), 0),
        "forecast_p50": round(float(rate_now * np.exp(q[1])), 0),
        "forecast_p_hi": round(float(rate_now * np.exp(q[2])), 0),
        "expected_move_pct": round(100 * move, 2),
        "perspective": cfg.perspective,
        "signal": str(sig),
        "alert": bool(abs(move) > cfg.threshold),
    }
