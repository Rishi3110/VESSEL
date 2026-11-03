"""Load raw CSVs and align them on one point-in-time-correct modelling grid.

Rules enforced here (each one is a common source of fake accuracy):

1. Publication lag: a value observed for period P is stamped at the date it was
   *released* (obs date + ``publication_lag_days``), not the period end.
2. Resampling takes the *last* value released inside each bin, so a bin only
   contains information available at its close.
3. Gaps are forward-filled only (never interpolated - interpolation uses the
   next observation, i.e. the future), and only for ``max_ffill_periods``.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from vessel.config import Config


def _read_csv(path: Path, date_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}. See data/raw/README.md for the expected schema.")
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df.apply(pd.to_numeric, errors="coerce")


def to_grid(df: pd.DataFrame, grid: pd.DatetimeIndex, freq: str, lag_days: int, max_ffill: int) -> pd.DataFrame:
    """Shift by publication lag, resample to ``freq`` (last), reindex, bounded ffill."""
    df = df.copy()
    if lag_days:
        df.index = df.index + pd.Timedelta(days=lag_days)
    binned = df.resample(freq).last()
    return binned.reindex(grid).ffill(limit=max_ffill)


def load_panel(cfg: Config) -> pd.DataFrame:
    """Return one DataFrame on the modelling grid: target column + all exogenous columns."""
    d = cfg.data
    root = Path(d.data_dir)
    if d.source == "synthetic" and not (root / d.target_file).exists():
        from vessel.synthetic import generate

        generate(root)

    target = _read_csv(root / d.target_file, d.date_col)[[d.target_col]]
    if d.start:
        target = target[target.index >= pd.Timestamp(d.start)]
    if d.end:
        target = target[target.index <= pd.Timestamp(d.end)]
    if (target[d.target_col] <= 0).any():
        raise ValueError("Freight rates must be strictly positive (model works in log space).")

    target_w = target.resample(d.freq).last()
    grid = target_w.index
    panel = [target_w]  # the rate is never forward-filled: a missing bin stays missing

    seen = {d.target_col}
    for src in d.exog:
        raw = _read_csv(root / src.file, d.date_col)
        cols = src.columns or list(raw.columns)
        missing = [c for c in cols if c not in raw.columns]
        if missing:
            raise KeyError(f"{src.file}: columns not found {missing}")
        dup = seen.intersection(cols)
        if dup:
            raise ValueError(f"Duplicate column names across sources: {sorted(dup)}")
        seen.update(cols)
        panel.append(to_grid(raw[cols], grid, d.freq, src.publication_lag_days, d.max_ffill_periods))

    out = pd.concat(panel, axis=1)
    out.index.name = "date"
    return out


def stationarity_report(rate: pd.Series) -> pd.DataFrame:
    """ADF test on log level and log returns. Expect: level non-stationary, returns stationary.

    This is why the model predicts log *returns* over the horizon, not levels.
    """
    log_r = np.log(rate.dropna())
    rows = []
    for name, s in {"log_rate": log_r, "log_return_1p": log_r.diff().dropna()}.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)  # statsmodels return-type deprecation notice
            stat, p, *_ = adfuller(s, autolag="AIC")
        rows.append({"series": name, "adf_stat": stat, "p_value": p, "stationary_5pct": p < 0.05})
    return pd.DataFrame(rows)
