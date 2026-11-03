import json
from pathlib import Path

from vessel.pipeline import run


def test_end_to_end(cfg):
    s = run(cfg)
    out = Path(cfg.output_dir)
    for f in ["report.html", "metrics_test.csv", "metrics_cv.csv", "feature_importance.csv", "decision_backtest.csv",
              "latest_signal.json", "best_params.json", "model_point.json", "model_quantile.json", "summary.json"]:
        assert (out / f).exists(), f
    live = json.loads((out / "latest_signal.json").read_text())
    assert live["signal"] in {"FIX_NOW", "WAIT"}
    assert live["forecast_p_lo"] <= live["forecast_p50"] <= live["forecast_p_hi"]
    assert s["conclusions"] and s["test_period"][1] <= "2019-12-31"
    assert "Synthetic data" not in (out / "report.html").read_text()  # source=csv in this fixture
