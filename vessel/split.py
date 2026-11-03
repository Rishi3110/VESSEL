"""Chronological splits with a purge gap.

With an h-step target, the label of row i is built from rate_{i+h}. If a
training row sits within h periods of the first test row, its label contains
prices from inside the test window - a leak that plain TimeSeriesSplit (gap=0)
misses. Every split here drops ``gap = horizon`` rows between train and test.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def holdout_split(n: int, test_periods: int, gap: int) -> tuple[np.ndarray, np.ndarray]:
    if test_periods + gap >= n:
        raise ValueError(f"Not enough rows ({n}) for test_periods={test_periods} + gap={gap}")
    test = np.arange(n - test_periods, n)
    train = np.arange(0, n - test_periods - gap)
    return train, test


def walk_forward(n: int, n_splits: int, test_size: int, gap: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward folds (train always strictly before test)."""
    tss = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    return [(tr, te) for tr, te in tss.split(np.arange(n))]


def inner_val_split(train_idx: np.ndarray, val_periods: int, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Carve an early-stopping set off the END of a training fold - never the test fold."""
    if len(train_idx) <= val_periods + gap + 10:
        raise ValueError("Training fold too short for an early-stopping validation slice")
    return train_idx[: len(train_idx) - val_periods - gap], train_idx[-val_periods:]
