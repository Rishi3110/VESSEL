"""Feature engineering. Every feature at row t uses only data stamped <= t.

Target: y_t = log(rate_{t+h}) - log(rate_t), the h-period-ahead log return.
Predicting returns instead of levels matters for trees: XGBoost cannot
extrapolate beyond the target range it saw in training, and freight levels
wander into new regimes (2008, 2021). Returns are stationary; the level forecast
is rebuilt as rate_t * exp(y_hat).

Leakage-safety is tested in tests/test_features.py by truncating the input at
random dates and checking past feature rows do not change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vessel.config import Config

# Chinese New Year (Spring Festival day 1). Dry-bulk demand dips around it:
# steel mills slow, ports close, and rates typically soften into the holiday.
CNY_DATES = [
    "2005-02-09", "2006-01-29", "2007-02-18", "2008-02-07", "2009-01-26", "2010-02-14",
    "2011-02-03", "2012-01-23", "2013-02-10", "2014-01-31", "2015-02-19", "2016-02-08",
    "2017-01-28", "2018-02-16", "2019-02-05", "2020-01-25", "2021-02-12", "2022-02-01",
    "2023-01-22", "2024-02-10", "2025-01-29", "2026-02-17", "2027-02-06", "2028-01-26",
    "2029-02-13", "2030-02-03",
]


def make_target(rate: pd.Series, horizon: int) -> pd.Series:
    lr = np.log(rate)
    return (lr.shift(-horizon) - lr).rename(f"target_logret_{horizon}")


def _rolling_z(s: pd.Series, w: int) -> pd.Series:
    mu = s.rolling(w, min_periods=w // 2).mean()
    sd = s.rolling(w, min_periods=w // 2).std()
    return (s - mu) / sd.replace(0, np.nan)


def _target_features(rate: pd.Series, fc) -> dict[str, pd.Series]:
    f: dict[str, pd.Series] = {}
    lr = np.log(rate)
    ret1 = lr.diff()
    f["log_rate"] = lr
    for k in fc.target_lags:
        f[f"ret_{k}"] = lr - lr.shift(k)  # == log(rate_t / rate_{t-k})
    for k in fc.return_lags:
        f[f"ret1_lag{k}"] = ret1.shift(k)
    for w in fc.rolling_windows:
        roll = lr.rolling(w, min_periods=w)
        f[f"ma_gap_{w}"] = lr - roll.mean()
        f[f"vol_{w}"] = ret1.rolling(w, min_periods=w).std()
        lo = rate.rolling(w, min_periods=w).min()
        hi = rate.rolling(w, min_periods=w).max()
        f[f"range_pos_{w}"] = (rate - lo) / (hi - lo).replace(0, np.nan)
    zw = fc.zscore_window
    f[f"z_{zw}"] = _rolling_z(lr, zw)
    prev_vol = ret1.rolling(zw, min_periods=zw // 2).std().shift(1)
    f["spike_flag"] = (ret1.abs() / prev_vol > fc.spike_z).astype(float).where(prev_vol.notna())
    return f


def _exog_features(df: pd.DataFrame, fc) -> dict[str, pd.Series]:
    f: dict[str, pd.Series] = {}
    for c in df.columns:
        s = df[c]
        f[c] = s
        if s.dropna().nunique() <= 2:  # binary flags: level only
            continue
        positive = bool((s.dropna() > 0).all())
        for w in fc.exog_change_windows:
            f[f"{c}_chg{w}"] = (np.log(s) - np.log(s.shift(w))) if positive else (s - s.shift(w))
        f[f"{c}_z{fc.zscore_window}"] = _rolling_z(s, fc.zscore_window)
    return f


def _domain_features(panel: pd.DataFrame, rate: pd.Series, fc) -> dict[str, pd.Series]:
    f: dict[str, pd.Series] = {}
    demand_cols = [c for c in fc.demand_cols if c in panel]
    if demand_cols and fc.supply_col in panel:
        demand = panel[demand_cols].sum(axis=1, min_count=len(demand_cols))
        supply = panel[fc.supply_col]
        f["utilization_proxy"] = demand / supply
        # smooth demand over a quarter to remove harvest/CNY seasonality, then
        # compare year-on-year growth of demand vs fleet: >0 = market tightening
        dm = demand.rolling(13, min_periods=8).mean()
        f["demand_minus_supply_growth_yoy"] = (np.log(dm) - np.log(dm.shift(52))) - (np.log(supply) - np.log(supply.shift(52)))
    if fc.orderbook_col in panel and fc.supply_col in panel:
        f["orderbook_to_fleet"] = panel[fc.orderbook_col] / panel[fc.supply_col]
    if fc.bunker_col in panel:
        # voyage fuel cost per day as a share of the day rate earned (both known at t)
        f["bunker_cost_share"] = panel[fc.bunker_col] * fc.fuel_tpd / rate
    return f


def _calendar_features(idx: pd.DatetimeIndex, fc) -> dict[str, pd.Series]:
    f: dict[str, pd.Series] = {}
    if fc.calendar:
        m = idx.month.to_numpy()
        w = idx.isocalendar().week.to_numpy().astype(float)
        f["month_sin"] = pd.Series(np.sin(2 * np.pi * (m - 1) / 12), idx)
        f["month_cos"] = pd.Series(np.cos(2 * np.pi * (m - 1) / 12), idx)
        f["woy_sin"] = pd.Series(np.sin(2 * np.pi * (w - 1) / 52), idx)
        f["woy_cos"] = pd.Series(np.cos(2 * np.pi * (w - 1) / 52), idx)
        f["quarter"] = pd.Series(idx.quarter.to_numpy(), idx).astype(float)
    if fc.day_of_week:
        d = idx.dayofweek.to_numpy()
        f["dow_sin"] = pd.Series(np.sin(2 * np.pi * d / 7), idx)
        f["dow_cos"] = pd.Series(np.cos(2 * np.pi * d / 7), idx)
    if fc.chinese_new_year:
        cny = pd.to_datetime(CNY_DATES).values
        delta = (cny[None, :] - idx.values[:, None]).astype("timedelta64[D]").astype(float)
        nearest = delta[np.arange(len(idx)), np.abs(delta).argmin(axis=1)]
        # signed days to the nearest CNY (+ = holiday ahead); calendar is known in advance
        f["days_to_cny"] = pd.Series(np.clip(nearest, -90, 90), idx)
    return f


def build_features(panel: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, rate). Rows where y is NaN (the last ``horizon`` rows) are kept:
    they are the live-forecast rows."""
    fc = cfg.features
    tcol = cfg.data.target_col
    rate = panel[tcol]
    exog = panel.drop(columns=[tcol])

    feats: dict[str, pd.Series] = {}
    feats.update(_target_features(rate, fc))
    feats.update(_exog_features(exog, fc))
    feats.update(_domain_features(panel, rate, fc))
    feats.update(_calendar_features(panel.index, fc))
    X = pd.DataFrame(feats, index=panel.index)

    for pair in fc.interactions:
        a, b = pair
        if a in X and b in X:
            X[f"{a}_x_{b}"] = X[a] * X[b]
        else:
            raise KeyError(f"Interaction {pair}: feature not found. Available: {sorted(X.columns)[:20]}...")

    X = X.replace([np.inf, -np.inf], np.nan)
    y = make_target(rate, cfg.horizon)

    burn_in = max([*fc.target_lags, *fc.rolling_windows, fc.zscore_window, *fc.exog_change_windows, 1])
    keep = rate.notna()
    keep.iloc[:burn_in] = False
    return X[keep], y[keep], rate[keep]
