"""Static charts for the report (matplotlib, headless)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3 (validated order)


def _style(ax, title: str, ylabel: str | None = None):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def _usd(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def history(rate: pd.Series, test_start: pd.Timestamp, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(rate.index, rate.values, color=S1, linewidth=1.6)
    ax.axvspan(test_start, rate.index[-1], color=GRID, alpha=0.6, linewidth=0)
    ax.text(test_start, ax.get_ylim()[1] * 0.97, "  hold-out test", color=INK_2, fontsize=9, va="top")
    _style(ax, "Freight rate history", "$/day")
    _usd(ax)
    return _save(fig, path)


def forecast_band(bt: pd.DataFrame, horizon: int, freq: str, quantiles: list[float], path: Path) -> Path:
    """Forecast made at t, plotted at its target date t+h, against what actually printed."""
    target_dates = bt.index + horizon * pd.tseries.frequencies.to_offset(freq)
    fig, ax = plt.subplots(figsize=(10, 4))
    lo, hi = int(quantiles[0] * 100), int(quantiles[-1] * 100)
    ax.fill_between(target_dates, bt["p_lo"], bt["p_hi"], color=S1, alpha=0.15, linewidth=0, label=f"P{lo}-P{hi} band")
    ax.plot(target_dates, bt["p50"], color=S1, linewidth=2, label="Forecast P50")
    ax.plot(target_dates, bt["rate_later"], color=INK, linewidth=1.6, label="Actual")
    _style(ax, f"Hold-out: {horizon}-period-ahead forecast vs actual", "$/day")
    _usd(ax)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_2)
    return _save(fig, path)


def importance(imp: pd.DataFrame, path: Path, top: int = 15) -> Path:
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(d) + 1.2))
    ax.barh(d.index, d["mean_abs_shap"], color=S1, height=0.6)
    _style(ax, "Top drivers (mean |SHAP|, hold-out)")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_xlabel("mean |SHAP| (log-return units)", color=INK_2, fontsize=9)
    return _save(fig, path)


def shap_beeswarm(sv: np.ndarray, X: pd.DataFrame, path: Path, top: int = 15) -> Path:
    import shap

    shap.summary_plot(sv, X, max_display=top, show=False, plot_size=(8, 0.35 * top + 1.5))
    fig = plt.gcf()
    fig.axes[0].set_title("SHAP: direction and size of each driver", loc="left", fontsize=12, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def cv_folds(cv: pd.DataFrame, path: Path, models: tuple[str, ...] = ("xgb_tuned", "ridge", "naive_last_value")) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    for name, color in zip(models, (S1, S2, S3)):
        d = cv[cv["model"] == name].sort_values("fold")
        ax.plot(d["fold"], d["rmse_usd"], color=color, linewidth=2, marker="o", markersize=5, label=name)
    _style(ax, "Walk-forward CV: RMSE per fold (lower is better)", "RMSE $/day")
    _usd(ax)
    ax.set_xlabel("fold (later = more recent)", color=INK_2, fontsize=9)
    ax.set_xticks(sorted(cv["fold"].unique()))
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    return _save(fig, path)


def decision_pnl(bt: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(bt.index, bt["perfect_foresight_usd"].cumsum(), color=MUTED, linewidth=1.6, linestyle="--", label="Perfect foresight")
    ax.plot(bt.index, bt["pnl_vs_fix_now_usd"].cumsum(), color=S1, linewidth=2, label="Model signal")
    ax.axhline(0, color=INK_2, linewidth=0.8)
    _style(ax, "Decision backtest: cumulative $ vs always fixing now", "USD per fixture, cumulative")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1e6:,.1f}M" if abs(v) >= 1e6 else f"${v / 1e3:,.0f}k"))
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK_2)
    return _save(fig, path)
