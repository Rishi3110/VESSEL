"""Models: XGBoost (point + quantile), a ridge benchmark, and naive baselines.

All models predict the h-period log return. Baselines are expressed the same
way so every model is scored on identical rows and identical targets.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from vessel.config import Config
from vessel.split import inner_val_split, walk_forward

log = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.6,
    "min_child_weight": 5.0,
    "gamma": 1e-6,  # ~0; kept >0 so it can seed the log-scale Optuna search
    "reg_alpha": 1e-6,
    "reg_lambda": 1.0,
}


# --------------------------------------------------------------------- utils
class TargetWinsorizer:
    """Clip training targets at quantiles learned from the training fold only.

    Freight spikes are real, so they are never deleted and never clipped in
    evaluation; clipping only limits how hard a handful of crisis weeks pull a
    squared-error model. (Winsorizing *features* is pointless for trees - splits
    are invariant to monotone transforms - so it is not done.)
    """

    def __init__(self, limits: list[float] | None):
        self.limits = limits
        self.lo = self.hi = None

    def fit(self, y: np.ndarray) -> "TargetWinsorizer":
        if self.limits:
            self.lo, self.hi = np.nanquantile(y, self.limits)
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return y if self.lo is None else np.clip(y, self.lo, self.hi)


def periods_per_year(freq: str) -> int:
    f = freq.upper()
    if f.startswith("W"):
        return 52
    if f.startswith(("B", "D")):
        return 260 if f.startswith("B") else 365
    if f.startswith(("M", "ME", "BM")):
        return 12
    if f.startswith("Q"):
        return 4
    raise ValueError(f"Unsupported frequency {freq}")


# ----------------------------------------------------------------- baselines
def baseline_predictions(rate: pd.Series, horizon: int, freq: str, ma_window: int) -> pd.DataFrame:
    """h-period log-return forecasts from simple rules, indexed like ``rate``."""
    lr = np.log(rate)
    p = periods_per_year(freq)
    return pd.DataFrame({
        "naive_last_value": 0.0,  # rate_{t+h} = rate_t
        "moving_average": np.log(rate.rolling(ma_window, min_periods=ma_window).mean()) - lr,
        "momentum": lr - lr.shift(horizon),  # last h-period move repeats
        "seasonal_naive": lr.shift(p - horizon) - lr.shift(p),  # same move as last year
    }, index=rate.index)


# --------------------------------------------------------------------- ridge
class ClipToTrainRange(BaseEstimator, TransformerMixin):
    """Clip each feature to its training [min, max].

    Trending level features (fleet size, steel output) leave their training range
    in the test period; a linear model then extrapolates them into absurd
    forecasts. Trees clip implicitly - this gives the linear benchmark the same
    protection so the comparison is fair.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_, self.hi_ = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def fit_ridge(X: pd.DataFrame, y: np.ndarray, horizon: int):
    """Linear benchmark. Alpha is picked with purged time-series CV: RidgeCV's default
    leave-one-out leaks through the autocorrelation of overlapping h-step targets and
    selects almost no regularisation."""
    cv = TimeSeriesSplit(n_splits=5, gap=horizon)
    model = make_pipeline(SimpleImputer(strategy="median"), ClipToTrainRange(), StandardScaler(),
                          RidgeCV(alphas=np.logspace(-1, 5, 30), cv=cv))
    return model.fit(X, y)


# ------------------------------------------------------------------- xgboost
def _xgb(params: dict, n_estimators: int, seed: int, quantiles: list[float] | None = None, es_rounds: int | None = None):
    kw = dict(params)
    kw.update(n_estimators=n_estimators, random_state=seed, tree_method="hist", n_jobs=-1)
    if quantiles is not None:
        kw.update(objective="reg:quantileerror", quantile_alpha=np.asarray(quantiles))
    else:
        kw.update(objective="reg:squarederror", eval_metric="rmse")
    if es_rounds:
        kw["early_stopping_rounds"] = es_rounds
    return xgb.XGBRegressor(**kw)


def fit_xgb(X: pd.DataFrame, y: np.ndarray, params: dict, cfg: Config, quantiles: list[float] | None = None):
    """Early-stop on the purged tail of (X, y), then refit on all of (X, y) with the found tree count.

    Returns (model, n_estimators).
    """
    m = cfg.model
    idx = np.arange(len(X))
    fit_idx, val_idx = inner_val_split(idx, cfg.split.val_periods, cfg.horizon)
    w = TargetWinsorizer(m.winsorize_target).fit(y[fit_idx])
    es = _xgb(params, m.max_estimators, m.seed, quantiles, m.early_stopping_rounds)
    es.fit(X.iloc[fit_idx], w.transform(y[fit_idx]), eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=False)
    n_est = max(int(es.best_iteration) + 1, 10)

    w_full = TargetWinsorizer(m.winsorize_target).fit(y)
    final = _xgb(params, n_est, m.seed, quantiles)
    final.fit(X, w_full.transform(y), verbose=False)
    return final, n_est


def fit_xgb_default(X: pd.DataFrame, y: np.ndarray, seed: int):
    """Methodology step 6 baseline: XGBRegressor with library defaults, no tuning."""
    return xgb.XGBRegressor(random_state=seed, tree_method="hist").fit(X, y)


def predict_quantiles(model, X: pd.DataFrame) -> np.ndarray:
    q = np.asarray(model.predict(X))
    return np.sort(q, axis=1)  # enforce monotone quantiles (no crossing)


# -------------------------------------------------------------------- tuning
@dataclass
class TuningResult:
    params: dict
    cv_rmse: float
    trials: pd.DataFrame


def _suggest(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
        # targets are log returns (~0.1 scale) so gamma/alpha live on small scales
        "gamma": trial.suggest_float("gamma", 1e-6, 0.5, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
    }


def cv_score(X: pd.DataFrame, y: np.ndarray, params: dict, cfg: Config) -> tuple[float, list[int]]:
    s, m = cfg.split, cfg.model
    errs, iters = [], []
    for tr, te in walk_forward(len(X), s.cv_splits, s.cv_test_periods, cfg.horizon):
        fit_idx, val_idx = inner_val_split(tr, s.val_periods, cfg.horizon)
        w = TargetWinsorizer(m.winsorize_target).fit(y[fit_idx])
        model = _xgb(params, m.max_estimators, m.seed, None, m.early_stopping_rounds)
        model.fit(X.iloc[fit_idx], w.transform(y[fit_idx]), eval_set=[(X.iloc[val_idx], y[val_idx])], verbose=False)
        pred = model.predict(X.iloc[te])
        errs.append(np.sqrt(np.mean((y[te] - pred) ** 2)))
        iters.append(int(model.best_iteration) + 1)
    return float(np.mean(errs)), iters


def tune(X: pd.DataFrame, y: np.ndarray, cfg: Config) -> TuningResult:
    """Optuna TPE search, scored by mean walk-forward RMSE on the development set only."""
    base = {**DEFAULT_PARAMS, **cfg.model.params}
    if cfg.model.n_trials <= 0:
        score, _ = cv_score(X, y, base, cfg)
        return TuningResult(base, score, pd.DataFrame())

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.model.seed))
    study.enqueue_trial({k: v for k, v in base.items() if k in DEFAULT_PARAMS})  # start from the sane default

    def objective(trial: optuna.Trial) -> float:
        params = {**_suggest(trial), **cfg.model.params}
        score, iters = cv_score(X, y, params, cfg)
        trial.set_user_attr("median_trees", int(np.median(iters)))
        return score

    study.optimize(objective, n_trials=cfg.model.n_trials, show_progress_bar=False)
    best = {**study.best_params, **cfg.model.params}
    log.info("Best CV RMSE %.4f with %s", study.best_value, best)
    return TuningResult(best, float(study.best_value), study.trials_dataframe())
