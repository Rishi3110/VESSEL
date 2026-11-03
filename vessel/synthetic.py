"""Synthetic dry-bulk market simulator.

Real freight data (Baltic Exchange indices, Clarksons fleet data) is licensed,
so the repo ships with a simulator that reproduces the *mechanics* of the
market well enough to exercise the whole pipeline:

* Demand: China steel output (logistic growth, CNY dip) drives iron-ore imports;
  PMI leads steel by ~2 months; grain has two harvest peaks; coal has a winter peak.
* Supply: fleet grows from an orderbook (~30-month delivery lag). Owners order
  when rates are high and scrap when they are low -> the classic shipping cycle.
* Price: a convex (hockey-stick) supply curve maps utilisation to an equilibrium
  rate with a lay-up floor. The spot rate mean-reverts to it with fat-tailed
  noise and jumps. Port congestion and canal restrictions remove effective supply.

Files are written with the same schema and release conventions the CSV loader
expects for real data, so swapping in real data is a config change only.
Numbers are *not* real market history - never use them for decisions.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _piecewise_log(dates: pd.DatetimeIndex, knots: dict[str, float]) -> np.ndarray:
    kx = pd.to_datetime(list(knots)).values.astype("int64")
    ky = np.log(list(knots.values()))
    return np.exp(np.interp(dates.values.astype("int64"), kx, ky))


def generate(out_dir: str | Path, start: str = "2008-01-01", end: str = "2026-10-30", seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    days = pd.bdate_range(start, end)
    n = len(days)
    months = pd.period_range(days[0], days[-1], freq="M")
    nm = len(months)
    year_frac = months.year + (months.month - 1) / 12.0

    # ---------------- monthly demand drivers (exogenous to freight) ----------
    pmi = np.empty(nm)
    pmi[0] = 52.0
    shocks = {pd.Period("2008-11", "M"): -9.0, pd.Period("2020-02", "M"): -14.0, pd.Period("2015-08", "M"): -3.0}
    for i in range(1, nm):
        pmi[i] = 50 + 0.85 * (pmi[i - 1] - 50) + rng.normal(0, 0.9) + shocks.get(months[i], 0.0)

    steel_trend = 40 + 47 / (1 + np.exp(-(year_frac - 2013) / 2.5))
    pmi_lag2 = np.r_[np.full(2, 50.0), pmi[:-2]]
    cny_month = np.where(months.month == 2, -0.08, 0.0)
    steel = steel_trend * (1 + 0.012 * (pmi_lag2 - 50) + cny_month + rng.normal(0, 0.025, nm))

    iron = 1.15 * steel * (1 + rng.normal(0, 0.04, nm))
    coal_season = 0.08 * np.cos(2 * np.pi * (months.month - 1) / 12)  # winter peak
    coal = (15 + 0.6 * (year_frac - 2008)) * (1 + coal_season + rng.normal(0, 0.06, nm))
    grain_season = 0.12 * np.exp(-((months.month - 5) ** 2) / 2) + 0.10 * np.exp(-((months.month - 11) ** 2) / 2)
    grain = 30 * (1.015 ** (year_frac - 2008)) * (1 + grain_season + rng.normal(0, 0.05, nm))
    restock = np.where((year_frac >= 2020.8) & (year_frac < 2022.0), 1.05, 1.0)  # post-COVID restocking
    minor = 35 * (1.03 ** (year_frac - 2008)) * (1 + rng.normal(0, 0.03, nm))  # unobserved minor bulks
    demand = (iron + 0.6 * coal + 1.3 * grain + minor) * restock  # tonne-mile proxy (hidden)

    # ---------------- daily drivers -------------------------------------------
    bunker_anchor = _piecewise_log(days, {
        "2008-01-01": 480, "2008-07-01": 760, "2009-01-01": 280, "2011-06-01": 660,
        "2014-06-01": 610, "2016-01-15": 180, "2018-10-01": 460, "2020-04-20": 230,
        "2022-06-15": 1000, "2023-06-01": 590, "2026-10-30": 560,
    })
    dev = np.zeros(n)
    for t in range(1, n):
        dev[t] = 0.98 * dev[t - 1] + rng.normal(0, 0.012)
    bunker = bunker_anchor * np.exp(dev)

    cong = np.empty(n)
    cong[0] = 4.0
    covid_bump = np.where((days >= "2021-03-01") & (days < "2022-07-01"), 5.0, 0.0)
    for t in range(1, n):
        target = 4.0 + covid_bump[t]
        cong[t] = cong[t - 1] + 0.03 * (target - cong[t - 1]) + rng.normal(0, 0.12) + (rng.random() < 0.004) * rng.uniform(1, 3)
    cong = np.clip(cong, 0.5, None)

    panama = ((days >= "2023-08-01") & (days < "2024-05-01")).astype(int)

    # seasonality on the price itself: Chinese New Year lull, Q4 firming
    from vessel.features import CNY_DATES

    cny = pd.to_datetime(CNY_DATES)
    dist = np.min(np.abs((days.values[:, None] - cny.values[None, :]).astype("timedelta64[D]").astype(float)), axis=1)
    season = -0.18 * np.exp(-((dist / 14.0) ** 2)) + 0.05 * np.cos(2 * np.pi * (days.dayofyear - 320) / 365.25)

    # ---------------- coupled supply / price simulation -----------------------
    prod = demand[0] / (420.0 * 0.97)  # tonne-miles per dwt; calibrated so 2008 util ~0.97
    fleet, orderbook = 420.0, 200.0
    fleet_m, ob_m, deliv_m, scrap_m = np.zeros(nm), np.zeros(nm), np.zeros(nm), np.zeros(nm)
    rate = np.empty(n)
    util_d = np.empty(n)
    x = np.log(60000.0)
    month_idx = days.to_period("M")
    m_ptr = -1
    month_rates: list[float] = []
    prev_avg = 60000.0
    for t in range(n):
        m = months.get_loc(month_idx[t])
        if m != m_ptr:  # month roll: update supply with last month's average rate
            if month_rates:
                prev_avg = float(np.mean(month_rates))
            month_rates = []
            deliveries = orderbook / 30.0
            scrapping = fleet * (0.0004 + 0.006 * np.clip(1 - prev_avg / 13000.0, 0, 1))
            orders = fleet * 0.0045 * np.clip(prev_avg / 15000.0, 0.1, 4.0) ** 2
            fleet += deliveries - scrapping
            orderbook += orders - deliveries
            fleet_m[m], ob_m[m], deliv_m[m], scrap_m[m] = fleet, orderbook, deliveries, scrapping
            m_ptr = m
        eff_fleet = fleet * (1 - 0.012 * (cong[t] - 4.0) - 0.02 * panama[t])
        u = demand[m] / (eff_fleet * prod)
        util_d[t] = u
        f = 7.0 * (u - 0.88) + 60.0 * max(u - 0.90, 0.0) ** 2
        eq = np.log(3500.0 + 10500.0 * np.exp(f)) + 0.20 * np.log(bunker[t] / 500.0) + season[t]
        jump = rng.normal(0, 0.12) if rng.random() < 0.004 else 0.0
        x = x + 0.03 * (eq - x) + 0.02 * rng.standard_t(4) / np.sqrt(2) + jump
        x = float(np.clip(x, np.log(2500), np.log(150000)))
        rate[t] = np.exp(x)
        month_rates.append(rate[t])

    # ---------------- write files with real-world release conventions ---------
    month_end = months.to_timestamp(how="end").normalize()
    pd.DataFrame({"date": days, "rate": rate.round(0)}).to_csv(out / "rates.csv", index=False)
    pd.DataFrame({"date": days, "bunker_usd_t": bunker.round(1)}).to_csv(out / "bunker.csv", index=False)
    weekly_cong = pd.Series(cong, index=days).resample("W-FRI").mean()
    pd.DataFrame({"date": weekly_cong.index, "port_wait_days": weekly_cong.values.round(2)}).to_csv(out / "congestion.csv", index=False)
    pd.DataFrame({"date": days, "panama_restricted": panama}).to_csv(out / "canal.csv", index=False)
    pd.DataFrame({"date": month_end, "china_pmi": pmi.round(1), "china_steel_mt": steel.round(2)}).to_csv(out / "macro.csv", index=False)
    pd.DataFrame({
        "date": month_end, "iron_ore_imports_mt": iron.round(2),
        "coal_imports_mt": coal.round(2), "grain_exports_mt": grain.round(2),
    }).to_csv(out / "trade.csv", index=False)
    pd.DataFrame({
        "date": month_end, "fleet_dwt_m": fleet_m.round(2), "orderbook_dwt_m": ob_m.round(2),
        "deliveries_dwt_m": deliv_m.round(3), "scrapping_dwt_m": scrap_m.round(3),
    }).to_csv(out / "fleet.csv", index=False)
    (out / "README.txt").write_text(
        "SYNTHETIC DATA generated by vessel.synthetic - not real market history.\n"
        f"seed={seed} start={start} end={end}\n"
    )
    return out
