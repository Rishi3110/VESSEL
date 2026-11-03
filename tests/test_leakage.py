"""Point-in-time tests. If any of these fail, backtest numbers are fiction."""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from vessel.data import load_panel, to_grid
from vessel.features import build_features


@pytest.mark.parametrize("as_of", ["2014-06-13", "2016-02-12", "2018-11-30"])  # Fridays
def test_features_do_not_change_when_future_data_is_removed(cfg, truncated_copy, as_of):
    """Rebuild the whole pipeline input from raw files cut at ``as_of``. Every feature
    row up to ``as_of`` must equal the full-history version - so no feature at t can
    depend on data released after t (covers resampling, publication lags, ffill,
    rolling windows, z-scores, domain features)."""
    X_full, _, _ = build_features(load_panel(cfg), cfg)

    cut_cfg = dataclasses.replace(cfg, data=dataclasses.replace(cfg.data, data_dir=str(truncated_copy(as_of))))
    X_cut, _, _ = build_features(load_panel(cut_cfg), cut_cfg)

    assert X_cut.index[-1] == pd.Timestamp(as_of)
    pd.testing.assert_frame_equal(X_full.loc[:as_of], X_cut, check_exact=False, rtol=1e-10, atol=1e-12)


def test_target_is_future_return_and_last_rows_unlabelled(cfg):
    X, y, rate = build_features(load_panel(cfg), cfg)
    h = cfg.horizon
    expected = np.log(rate.shift(-h) / rate)
    pd.testing.assert_series_equal(y, expected, check_names=False)
    assert y.iloc[-h:].isna().all() and y.iloc[:-h].notna().all()


def test_publication_lag_delays_availability():
    monthly = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.to_datetime(["2020-01-31", "2020-02-29"]))
    grid = pd.date_range("2020-01-03", "2020-04-03", freq="W-FRI")
    out = to_grid(monthly, grid, "W-FRI", lag_days=15, max_ffill=8)["x"]
    # January value is released 2020-02-15 (Sat) -> first usable Friday is 2020-02-21
    assert out.loc[:"2020-02-14"].isna().all()
    assert out.loc["2020-02-21"] == 1.0
    # February value released 2020-03-15 (Sun) -> visible from Friday 2020-03-20
    assert out.loc["2020-03-13"] == 1.0 and out.loc["2020-03-20"] == 2.0


def test_forward_fill_is_bounded():
    s = pd.DataFrame({"x": [5.0]}, index=pd.to_datetime(["2020-01-03"]))
    grid = pd.date_range("2020-01-03", periods=12, freq="W-FRI")
    out = to_grid(s, grid, "W-FRI", lag_days=0, max_ffill=3)["x"]
    assert out.iloc[:4].eq(5.0).all() and out.iloc[4:].isna().all()
