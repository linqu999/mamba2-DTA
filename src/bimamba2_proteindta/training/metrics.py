"""Regression metrics used by DTA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable


NumberSeq = Iterable[float]


@dataclass(frozen=True)
class RegressionMetrics:
    mse: float
    rmse: float
    mae: float
    ci: float
    rm2: float

    def as_dict(self) -> dict[str, float]:
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "ci": self.ci,
            "rm2": self.rm2,
        }


def _to_float_list(values: NumberSeq, name: str) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError(f"{name} must contain at least one value")
    return result


def _validate_pair(y_true: NumberSeq, y_pred: NumberSeq) -> tuple[list[float], list[float]]:
    true_values = _to_float_list(y_true, "y_true")
    pred_values = _to_float_list(y_pred, "y_pred")
    if len(true_values) != len(pred_values):
        raise ValueError(
            f"y_true and y_pred must have the same length, got "
            f"{len(true_values)} and {len(pred_values)}"
        )
    return true_values, pred_values


def mse(y_true: NumberSeq, y_pred: NumberSeq) -> float:
    true_values, pred_values = _validate_pair(y_true, y_pred)
    return sum((truth - pred) ** 2 for truth, pred in zip(true_values, pred_values)) / len(true_values)


def rmse(y_true: NumberSeq, y_pred: NumberSeq) -> float:
    return sqrt(mse(y_true, y_pred))


def mae(y_true: NumberSeq, y_pred: NumberSeq) -> float:
    true_values, pred_values = _validate_pair(y_true, y_pred)
    return sum(abs(truth - pred) for truth, pred in zip(true_values, pred_values)) / len(true_values)


def concordance_index(y_true: NumberSeq, y_pred: NumberSeq) -> float:
    """Return the concordance index used in DTA regression papers.

    Ties in predictions receive half credit. Pairs with identical ground-truth
    affinity are ignored because they carry no ordering information.
    """
    true_values, pred_values = _validate_pair(y_true, y_pred)
    if len(true_values) < 2:
        return 0.0

    pred_ranks = {value: rank + 1 for rank, value in enumerate(sorted(set(pred_values)))}
    tree = _FenwickTree(len(pred_ranks))
    pairs = sorted(zip(true_values, pred_values), key=lambda item: item[0])

    concordant = 0.0
    comparable = 0
    previous_count = 0
    group_start = 0
    while group_start < len(pairs):
        group_end = group_start + 1
        while group_end < len(pairs) and pairs[group_end][0] == pairs[group_start][0]:
            group_end += 1

        for _, pred in pairs[group_start:group_end]:
            rank = pred_ranks[pred]
            lower = tree.prefix_sum(rank - 1)
            equal = tree.range_sum(rank, rank)
            concordant += lower + 0.5 * equal
            comparable += previous_count

        for _, pred in pairs[group_start:group_end]:
            tree.add(pred_ranks[pred], 1)
            previous_count += 1
        group_start = group_end

    if comparable == 0:
        return 0.0
    return concordant / comparable


class _FenwickTree:
    def __init__(self, size: int) -> None:
        self.values = [0] * (size + 1)

    def add(self, index: int, delta: int) -> None:
        while index < len(self.values):
            self.values[index] += delta
            index += index & -index

    def prefix_sum(self, index: int) -> int:
        total = 0
        while index > 0:
            total += self.values[index]
            index -= index & -index
        return total

    def range_sum(self, start: int, end: int) -> int:
        return self.prefix_sum(end) - self.prefix_sum(start - 1)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _r_squared(y_true: list[float], y_pred: list[float]) -> float:
    true_mean = _mean(y_true)
    ss_tot = sum((value - true_mean) ** 2 for value in y_true)
    if ss_tot == 0:
        return 0.0
    ss_res = sum((truth - pred) ** 2 for truth, pred in zip(y_true, y_pred))
    return 1.0 - ss_res / ss_tot


def _r0_squared(y_true: list[float], y_pred: list[float]) -> float:
    denominator = sum(pred * pred for pred in y_pred)
    if denominator == 0:
        return 0.0
    slope = sum(truth * pred for truth, pred in zip(y_true, y_pred)) / denominator
    y_pred_zero_intercept = [slope * pred for pred in y_pred]
    return _r_squared(y_true, y_pred_zero_intercept)


def rm2_score(y_true: NumberSeq, y_pred: NumberSeq) -> float:
    """Return the RM2 metric used in QSAR/DTA model evaluation."""
    true_values, pred_values = _validate_pair(y_true, y_pred)
    r2 = _r_squared(true_values, pred_values)
    r02 = _r0_squared(true_values, pred_values)
    penalty = sqrt(abs(r2 * r2 - r02 * r02))
    return r2 * (1.0 - penalty)


def compute_regression_metrics(y_true: NumberSeq, y_pred: NumberSeq) -> RegressionMetrics:
    return RegressionMetrics(
        mse=mse(y_true, y_pred),
        rmse=rmse(y_true, y_pred),
        mae=mae(y_true, y_pred),
        ci=concordance_index(y_true, y_pred),
        rm2=rm2_score(y_true, y_pred),
    )
