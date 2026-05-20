import math

import pytest

from bimamba2_proteindta.training.metrics import (
    compute_regression_metrics,
    concordance_index,
    mae,
    mse,
    rm2_score,
    rmse,
)


def test_basic_regression_metrics() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.5, 2.0]

    assert mse(y_true, y_pred) == pytest.approx((0.0 + 0.25 + 1.0) / 3)
    assert rmse(y_true, y_pred) == pytest.approx(math.sqrt((0.0 + 0.25 + 1.0) / 3))
    assert mae(y_true, y_pred) == pytest.approx(0.5)


def test_concordance_index_counts_ties_as_half_credit() -> None:
    y_true = [1.0, 2.0, 3.0]
    y_pred = [1.0, 2.0, 2.0]

    assert concordance_index(y_true, y_pred) == pytest.approx(2.5 / 3.0)


def test_concordance_index_ignores_equal_truth_pairs() -> None:
    y_true = [1.0, 1.0, 2.0]
    y_pred = [2.0, 1.0, 3.0]

    assert concordance_index(y_true, y_pred) == pytest.approx(1.0)


def test_rm2_score_is_one_for_perfect_prediction() -> None:
    y_true = [1.0, 2.0, 3.0, 4.0]
    y_pred = [1.0, 2.0, 3.0, 4.0]

    assert rm2_score(y_true, y_pred) == pytest.approx(1.0)


def test_compute_regression_metrics_returns_named_values() -> None:
    metrics = compute_regression_metrics([1.0, 2.0], [1.5, 1.5])

    assert set(metrics.as_dict()) == {"mse", "rmse", "mae", "ci", "rm2"}


def test_metric_inputs_must_have_matching_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        mse([1.0], [1.0, 2.0])


def test_metric_inputs_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="at least one"):
        mse([], [])
