"""SHAP explanations for the point XGBoost model."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def shap_values(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X)
    base = explainer.expected_value
    return np.asarray(vals), float(np.ravel(base)[0])


def importance_table(model, X: pd.DataFrame, sv: np.ndarray) -> pd.DataFrame:
    gain = model.get_booster().get_score(importance_type="total_gain")
    tot = sum(gain.values()) or 1.0
    df = pd.DataFrame({
        "mean_abs_shap": np.abs(sv).mean(axis=0),
        "gain_share": [gain.get(c, 0.0) / tot for c in X.columns],
    }, index=X.columns)
    return df.sort_values("mean_abs_shap", ascending=False)


def explain_row(sv_row: np.ndarray, x_row: pd.Series, base: float, top: int = 8) -> pd.DataFrame:
    """Top drivers of one forecast: what moved it away from the average prediction."""
    df = pd.DataFrame({"value": x_row.values, "shap": sv_row}, index=x_row.index)
    df = df.reindex(df["shap"].abs().sort_values(ascending=False).index).head(top)
    df.attrs["base"] = base
    return df
