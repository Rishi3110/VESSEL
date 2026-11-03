"""Typed configuration loaded from YAML.

Every knob of the pipeline lives here so experiments are reproducible from a
single file. Unknown keys raise, so a typo in the YAML fails loudly instead of
silently falling back to a default.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass
class ExogSource:
    """One exogenous CSV file (date column + one or more value columns)."""

    name: str
    file: str
    columns: list[str] | None = None  # None = every non-date column
    # Days between the observation date and the day the number is actually
    # public. Monthly China data, fleet stats etc. are released weeks after the
    # period ends; using them at the period-end date is look-ahead leakage.
    publication_lag_days: int = 0


@dataclass
class DataConfig:
    source: str = "synthetic"  # "synthetic" (generate then load) or "csv"
    data_dir: str = "data/synthetic"
    target_file: str = "rates.csv"
    target_col: str = "rate"
    date_col: str = "date"
    freq: str = "W-FRI"  # modelling frequency (pandas offset alias)
    max_ffill_periods: int = 8  # stop carrying a stale value forward after this
    start: str | None = None
    end: str | None = None
    exog: list[ExogSource] = field(default_factory=list)


@dataclass
class FeatureConfig:
    target_lags: list[int] = field(default_factory=lambda: [1, 2, 4, 8, 13, 26, 52])
    return_lags: list[int] = field(default_factory=lambda: [1, 2, 3])
    rolling_windows: list[int] = field(default_factory=lambda: [4, 13, 26, 52])
    exog_change_windows: list[int] = field(default_factory=lambda: [4, 13, 52])
    zscore_window: int = 52
    supply_col: str | None = "fleet_dwt_m"
    orderbook_col: str | None = "orderbook_dwt_m"
    demand_cols: list[str] = field(default_factory=list)
    bunker_col: str | None = "bunker_usd_t"
    fuel_tpd: float = 35.0  # tonnes/day at sea; turns $/t bunker into $/day cost
    calendar: bool = True
    day_of_week: bool = False  # only meaningful for daily data
    chinese_new_year: bool = True
    interactions: list[list[str]] = field(default_factory=list)
    spike_z: float = 3.0


@dataclass
class SplitConfig:
    test_periods: int = 104  # final hold-out, touched once
    cv_splits: int = 5
    cv_test_periods: int = 26
    val_periods: int = 26  # tail of each train fold used for early stopping


@dataclass
class ModelConfig:
    n_trials: int = 40  # Optuna trials; 0 = skip tuning, use defaults
    max_estimators: int = 2000
    early_stopping_rounds: int = 50
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    winsorize_target: list[float] | None = field(default_factory=lambda: [0.01, 0.99])
    seed: int = 42
    params: dict[str, Any] = field(default_factory=dict)  # fixed overrides


@dataclass
class DecisionConfig:
    perspective: str = "charterer"  # "charterer" (pays freight) or "owner" (earns it)
    threshold: float = 0.05  # act only if expected move exceeds 5 %
    require_confidence: bool = False  # also require the quantile band to agree
    charter_days: int = 30  # converts $/day savings into $ per fixture
    alert_band: bool = True  # alert when actual breaks the forecast P10-P90 band


@dataclass
class Config:
    horizon: int = 4  # forecast horizon in modelling periods (4 weeks)
    output_dir: str = "reports"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    def validate(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.decision.perspective not in {"charterer", "owner"}:
            raise ValueError("decision.perspective must be 'charterer' or 'owner'")
        if self.data.source not in {"synthetic", "csv"}:
            raise ValueError("data.source must be 'synthetic' or 'csv'")
        qs = self.model.quantiles
        if len(qs) != 3 or not (0 < qs[0] < qs[1] < qs[2] < 1):
            raise ValueError("model.quantiles must be three increasing values in (0, 1)")


def _build(cls, raw: dict[str, Any]):
    if raw is None:
        return cls()
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise KeyError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for name, value in raw.items():
        tp = hints[name]
        if is_dataclass(tp) and isinstance(value, dict):
            kwargs[name] = _build(tp, value)
        elif get_origin(tp) is list and value is not None:
            (inner,) = get_args(tp) or (Any,)
            if is_dataclass(inner):
                kwargs[name] = [_build(inner, v) for v in value]
            else:
                kwargs[name] = value
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    cfg = _build(Config, raw)
    cfg.validate()
    return cfg


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
