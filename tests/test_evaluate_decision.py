import numpy as np
import pandas as pd
import pytest

from vessel.config import DecisionConfig, load_config
from vessel.decision import backtest, band_breaches, signals
from vessel.evaluate import diebold_mariano, majority_direction_rate, point_metrics, quantile_metrics
from vessel.models import ClipToTrainRange, TargetWinsorizer, baseline_predictions


def test_point_metrics_perfect_forecast():
    y = np.array([0.1, -0.05, 0.02])
    m = point_metrics(y, y, np.array([10000, 12000, 9000]))
    assert m["rmse_usd"] == pytest.approx(0) and m["mape_pct"] == pytest.approx(0)
    assert m["directional_acc"] == 1.0


def test_naive_has_undefined_direction():
    m = point_metrics(np.array([0.1, -0.1]), np.zeros(2), np.array([1e4, 1e4]))
    assert np.isnan(m["directional_acc"])


def test_majority_direction():
    assert majority_direction_rate(np.array([0.1, 0.2, -0.1, 0.0])) == pytest.approx(2 / 3)


def test_diebold_mariano():
    rng = np.random.default_rng(0)
    e = rng.normal(size=300)
    assert diebold_mariano(e, e, 4) == (0.0, 1.0)
    stat, p = diebold_mariano(0.3 * e, e + rng.normal(size=300), 4)
    assert stat < 0 and p < 0.01  # model clearly better -> negative and significant


def test_quantile_coverage():
    y = np.linspace(-1, 1, 101)
    q = np.column_stack([np.full(101, -0.5), np.zeros(101), np.full(101, 0.5)])
    assert quantile_metrics(y, q, [0.1, 0.5, 0.9])["interval_coverage"] == pytest.approx(51 / 101)


def test_baselines_use_only_past():
    rate = pd.Series(np.exp(np.linspace(9, 10, 200)), index=pd.date_range("2015-01-02", periods=200, freq="W-FRI"))
    b = baseline_predictions(rate, horizon=4, freq="W-FRI", ma_window=4)
    lr = np.log(rate)
    assert (b["naive_last_value"] == 0).all()
    assert b["momentum"].iloc[10] == pytest.approx(lr.iloc[10] - lr.iloc[6])
    assert b["seasonal_naive"].iloc[60] == pytest.approx(lr.iloc[60 - 48] - lr.iloc[60 - 52])


def test_winsorizer_and_clipper():
    w = TargetWinsorizer([0.1, 0.9]).fit(np.arange(11.0))
    assert w.transform(np.array([-5.0, 5.0, 50.0])).tolist() == [1.0, 5.0, 9.0]
    assert TargetWinsorizer(None).fit(np.arange(3.0)).transform(np.array([99.0]))[0] == 99.0
    c = ClipToTrainRange().fit(np.array([[0.0], [10.0]]))
    assert c.transform(np.array([[-3.0], [30.0]])).ravel().tolist() == [0.0, 10.0]


def test_signals_by_perspective():
    q = np.log(np.array([[0.80, 0.85, 0.95],   # expected -15 %
                         [0.98, 1.00, 1.02],   # flat
                         [1.05, 1.15, 1.30]]))  # expected +15 %
    assert signals(q, DecisionConfig(perspective="charterer")).tolist() == ["WAIT", "FIX_NOW", "FIX_NOW"]
    assert signals(q, DecisionConfig(perspective="owner")).tolist() == ["FIX_NOW", "FIX_NOW", "WAIT"]
    # confidence: charterer waits only if even P_hi is a fall
    q2 = q.copy()
    q2[0, 2] = np.log(1.02)
    assert signals(q2, DecisionConfig(perspective="charterer", require_confidence=True))[0] == "FIX_NOW"


def test_backtest_pnl():
    q = np.log(np.array([[0.8, 0.85, 0.9], [1.0, 1.0, 1.0]]))
    idx = pd.date_range("2024-01-05", periods=2, freq="W-FRI")
    bt, s = backtest(idx, np.array([10000.0, 10000.0]), np.array([9000.0, 11000.0]), q,
                     DecisionConfig(perspective="charterer", charter_days=30))
    assert bt["signal"].tolist() == ["WAIT", "FIX_NOW"]
    assert s["total_pnl_vs_fix_now_usd"] == pytest.approx(30 * 1000)  # waited and paid 1000/day less
    assert s["perfect_foresight_usd"] == pytest.approx(30 * 1000)
    assert s["capture_ratio"] == pytest.approx(1.0) and s["wait_hit_rate"] == 1.0
    assert band_breaches(bt).index.tolist() == [idx[1]]  # 11000 is above the flat band


def test_config_rejects_unknown_keys():
    with pytest.raises(KeyError):
        load_config(None, {"model": {"n_trails": 5}})
    with pytest.raises(ValueError):
        load_config(None, {"decision": {"perspective": "broker"}})
