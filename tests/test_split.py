import numpy as np
import pytest

from vessel.split import holdout_split, inner_val_split, walk_forward


def test_holdout_purges_horizon():
    tr, te = holdout_split(200, test_periods=40, gap=4)
    assert te[0] == 160 and te[-1] == 199
    assert tr[-1] + 4 < te[0]  # label of last train row (uses t+4) ends before test starts


def test_walk_forward_is_chronological_with_gap():
    folds = walk_forward(300, n_splits=4, test_size=30, gap=4)
    assert len(folds) == 4
    prev_end = -1
    for tr, te in folds:
        assert tr[-1] + 4 < te[0]
        assert te[0] > prev_end
        assert tr[0] == 0  # expanding window
        prev_end = te[-1]


def test_inner_val_comes_from_train_tail():
    tr = np.arange(100)
    fit, val = inner_val_split(tr, val_periods=20, gap=4)
    assert val[0] == 80 and fit[-1] == 75
    with pytest.raises(ValueError):
        inner_val_split(np.arange(30), val_periods=20, gap=4)


def test_holdout_too_small_raises():
    with pytest.raises(ValueError):
        holdout_split(50, test_periods=48, gap=4)
