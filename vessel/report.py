"""Self-contained HTML report (images embedded) - open it in any browser or attach it to an email."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd

from vessel.config import Config

CSS = """
:root { --bg:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --rule:#e4e3df; --accent:#2a78d6; --warn:#b54708; --good:#067647; }
@media (prefers-color-scheme: dark) { :root { --bg:#1a1a19; --ink:#f0efec; --ink2:#c3c2b7; --rule:#3a3a37; --accent:#3987e5; --warn:#f79009; --good:#47cd89; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width: 1040px; margin: 0 auto; padding: 32px 16px 64px; }
h1 { font-size: 26px; margin: 0 0 4px; } h2 { font-size: 19px; margin: 36px 0 10px; border-bottom:1px solid var(--rule); padding-bottom:6px; }
.sub { color: var(--ink2); margin: 0 0 20px; }
.banner { border:1px solid var(--warn); color:var(--warn); padding:10px 14px; border-radius:6px; margin:16px 0; }
.tiles { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap:12px; }
.tile { border:1px solid var(--rule); border-radius:8px; padding:12px 14px; }
.tile .k { color:var(--ink2); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }
.tile .v { font-size:24px; font-weight:600; font-variant-numeric: tabular-nums; }
.tile .n { color:var(--ink2); font-size:12px; }
ul.verdict li { margin: 4px 0; }
table { border-collapse: collapse; width:100%; font-size:13px; font-variant-numeric: tabular-nums; }
th, td { padding:6px 8px; border-bottom:1px solid var(--rule); text-align:right; white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
.scroll { overflow-x:auto; }
img { max-width:100%; height:auto; border-radius:6px; margin: 8px 0; }
code { background: color-mix(in srgb, var(--rule) 60%, transparent); padding: 1px 4px; border-radius: 4px; }
"""


def _img(path: Path) -> str:
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img alt="{html.escape(Path(path).stem)}" src="data:image/png;base64,{b64}">'


def _table(df: pd.DataFrame, floatfmt: str = "{:,.3f}") -> str:
    return '<div class="scroll">' + df.to_html(float_format=lambda v: floatfmt.format(v), border=0, na_rep="–") + "</div>"


def write_report(path: Path, cfg: Config, s: dict, stat, table, cv, imp, breaches, figs: dict) -> Path:
    live, dec, qm = s["live"], s["decision"], s["quantile_metrics"]
    synth = cfg.data.source == "synthetic"
    cv_summary = cv.groupby("model")[["rmse_usd", "mape_pct", "directional_acc"]].agg(["mean", "std"]).sort_values(("rmse_usd", "mean"))
    cv_summary.columns = [f"{a}_{b}" for a, b in cv_summary.columns]
    drivers = pd.DataFrame(live["top_drivers"]).set_index("feature")
    move_cls = "good" if live["signal"] == "FIX_NOW" else "warn"

    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Freight Forecast Report</title><style>{CSS}</style></head><body><main>",
             "<h1>Freight rate forecast</h1>",
             f"<p class='sub'>XGBoost, {s['horizon']}-period horizon ({s['freq']}) · dev {s['dev_period'][0]} → {s['dev_period'][1]} · "
             f"hold-out {s['test_period'][0]} → {s['test_period'][1]} · {s['features']} features</p>"]
    if synth:
        parts.append("<div class='banner'><b>Synthetic data.</b> These numbers come from the built-in market simulator, "
                     "not real Baltic/Clarksons data. They demonstrate the pipeline; they say nothing about the real market.</div>")

    parts += ["<h2>Live signal</h2><div class='tiles'>",
              f"<div class='tile'><div class='k'>Rate now ({live['as_of']})</div><div class='v'>${live['rate_now']:,.0f}</div><div class='n'>$/day</div></div>",
              f"<div class='tile'><div class='k'>Forecast P50 (+{s['horizon']})</div><div class='v'>${live['forecast_p50']:,.0f}</div>"
              f"<div class='n'>{live['expected_move_pct']:+.1f}% · band ${live['forecast_p_lo']:,.0f}–${live['forecast_p_hi']:,.0f}</div></div>",
              f"<div class='tile'><div class='k'>Action ({live['perspective']})</div><div class='v' style='color:var(--{move_cls})'>{live['signal'].replace('_', ' ')}</div>"
              f"<div class='n'>threshold ±{100 * cfg.decision.threshold:.0f}%{' · ALERT' if live['alert'] else ''}</div></div>",
              "</div>",
              "<p class='sub' style='margin-top:12px'>Top drivers of this forecast (SHAP, log-return units; + pushes the forecast up):</p>",
              _table(drivers, "{:,.4f}"),
              "<h2>Verdict</h2><ul class='verdict'>", *[f"<li>{html.escape(c)}</li>" for c in s["conclusions"]], "</ul>",
              "<h2>Hold-out test (touched once)</h2>",
              "<p class='sub'>Errors on rate levels in $/day. <code>skill_vs_naive</code> = 1 − RMSE/RMSE<sub>naive</sub>. "
              "<code>dm_pvalue</code>: Diebold-Mariano test vs naive (HAC, HLN-corrected). "
              f"Majority-direction baseline for directional accuracy: {100 * s['majority_direction_rate']:.1f}%.</p>",
              _table(table), _img(figs["forecast"]),
              f"<p class='sub'>Quantile band P{int(cfg.model.quantiles[0] * 100)}–P{int(cfg.model.quantiles[-1] * 100)}: nominal "
              f"{100 * qm['interval_nominal']:.0f}% coverage, realised {100 * qm['interval_coverage']:.1f}%.</p>",
              "<h2>Walk-forward cross-validation (dev set)</h2>", _table(cv_summary), _img(figs["cv"]),
              "<h2>Decision backtest</h2>",
              f"<p class='sub'>{dec['perspective'].title()} view, {cfg.decision.charter_days}-day fixtures: WAIT when the P50 move exceeds "
              f"±{100 * cfg.decision.threshold:.0f}% in the favourable direction, else FIX NOW. P&amp;L is vs always fixing now.</p><div class='tiles'>",
              f"<div class='tile'><div class='k'>P&amp;L vs fix-now</div><div class='v'>${dec['total_pnl_vs_fix_now_usd']:,.0f}</div><div class='n'>{dec['decisions']} decisions</div></div>",
              f"<div class='tile'><div class='k'>WAIT signals</div><div class='v'>{dec['wait_signals']}</div><div class='n'>hit rate {100 * (dec['wait_hit_rate'] or 0):.0f}%</div></div>",
              f"<div class='tile'><div class='k'>Capture vs perfect foresight</div><div class='v'>{100 * (dec['capture_ratio'] or 0):.1f}%</div>"
              f"<div class='n'>of ${dec['perfect_foresight_usd']:,.0f}</div></div>",
              f"<div class='tile'><div class='k'>Worst single WAIT</div><div class='v'>${dec['worst_wait_usd']:,.0f}</div><div class='n'>downside check</div></div>",
              "</div>", _img(figs["decision"]),
              f"<p class='sub'>{len(breaches)} forecasts where the realised rate fell outside the band (review list in <code>band_breaches.csv</code>).</p>",
              "<h2>What drives the model</h2>", _img(figs["importance"]), _img(figs["beeswarm"]),
              _table(imp.head(20), "{:,.4f}"),
              "<h2>Data diagnostics</h2>", _img(figs["history"]), _table(stat),
              "<p class='sub'>ADF test. If the log level is non-stationary (p &gt; 0.05) a model trained on levels cannot generalise to new "
              "price regimes - the model therefore always targets h-period log returns, which are stationary either way.</p>",
              "</main></body></html>"]
    path.write_text("\n".join(parts))
    return path
