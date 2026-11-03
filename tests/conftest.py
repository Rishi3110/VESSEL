import shutil
from pathlib import Path

import pytest

from vessel.config import load_config
from vessel.synthetic import generate

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("synthetic")
    generate(d, start="2010-01-01", end="2019-12-31", seed=11)
    return d


@pytest.fixture()
def cfg(synth_dir, tmp_path):
    return load_config(ROOT / "configs" / "default.yaml", {
        "output_dir": str(tmp_path / "out"),
        "data": {"source": "csv", "data_dir": str(synth_dir)},
        "split": {"test_periods": 52, "cv_splits": 3, "cv_test_periods": 26, "val_periods": 26},
        "model": {"n_trials": 2, "max_estimators": 300},
    })


@pytest.fixture()
def truncated_copy(synth_dir, tmp_path):
    """Return a function that writes the dataset as it would have looked on ``as_of``
    (rows with observation date <= as_of) and returns the directory."""
    import pandas as pd

    def make(as_of: str) -> Path:
        out = tmp_path / f"asof_{as_of}"
        out.mkdir()
        for f in synth_dir.glob("*.csv"):
            df = pd.read_csv(f, parse_dates=["date"])
            df[df["date"] <= pd.Timestamp(as_of)].to_csv(out / f.name, index=False)
        shutil.copy(synth_dir / "README.txt", out / "README.txt")
        return out

    return make
