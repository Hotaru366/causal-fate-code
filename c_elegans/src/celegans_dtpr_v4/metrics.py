"""Masked matrix metrics for C. elegans propagation patterns."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _xy(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = mask & np.isfinite(predicted) & np.isfinite(target)
    return predicted[valid].astype(float), target[valid].astype(float)


def masked_relative_frobenius(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    x, y = _xy(predicted, target, mask)
    denom = float(np.linalg.norm(y))
    if denom <= 0.0:
        return float("nan")
    return float(np.linalg.norm(x - y) / denom)


def masked_rmse(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    x, y = _xy(predicted, target, mask)
    if x.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((x - y) ** 2)))


def masked_pearson(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    x, y = _xy(predicted, target, mask)
    if x.size < 3 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def masked_spearman(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    x, y = _xy(predicted, target, mask)
    if x.size < 3 or np.std(x) <= 0.0 or np.std(y) <= 0.0:
        return float("nan")
    rho = stats.spearmanr(x, y).correlation
    return float(rho)


def masked_sign_agreement(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    x, y = _xy(predicted, target, mask)
    keep = (np.abs(x) > 0.0) & (np.abs(y) > 0.0)
    if np.count_nonzero(keep) == 0:
        return float("nan")
    return float(np.mean(np.sign(x[keep]) == np.sign(y[keep])))


def masked_balanced_sign_accuracy(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    x, y = _xy(predicted, target, mask)
    keep = (np.abs(x) > 0.0) & (np.abs(y) > 0.0)
    if np.count_nonzero(keep) == 0:
        return float("nan")
    sx = np.sign(x[keep])
    sy = np.sign(y[keep])
    scores = []
    for sign in (-1.0, 1.0):
        cls = sy == sign
        if np.any(cls):
            scores.append(float(np.mean(sx[cls] == sy[cls])))
    return float(np.mean(scores)) if scores else float("nan")


def majority_sign_baseline(target: np.ndarray, mask: np.ndarray) -> float:
    _, y = _xy(target, target, mask)
    keep = np.abs(y) > 0.0
    if np.count_nonzero(keep) == 0:
        return float("nan")
    signs, counts = np.unique(np.sign(y[keep]), return_counts=True)
    return float(np.max(counts) / np.sum(counts))


def _filled_centered(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid = mask & np.isfinite(matrix)
    out = np.zeros_like(matrix, dtype=float)
    if np.count_nonzero(valid) == 0:
        return out
    out[valid] = matrix[valid] - float(np.mean(matrix[valid]))
    return out


def leading_mode_similarity(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    x = _filled_centered(predicted, mask)
    y = _filled_centered(target, mask)
    if not np.any(x) or not np.any(y):
        return float("nan")
    ux, _, vxh = np.linalg.svd(x, full_matrices=False)
    uy, _, vyh = np.linalg.svd(y, full_matrices=False)
    left = abs(float(np.dot(ux[:, 0], uy[:, 0])))
    right = abs(float(np.dot(vxh[0], vyh[0])))
    return float(np.sqrt(left * right))


def strongest_response_recovery(
    predicted: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    top_fraction: float = 0.05,
) -> float:
    x, y = _xy(predicted, target, mask)
    if x.size == 0:
        return float("nan")
    k = max(1, int(round(top_fraction * x.size)))
    target_top = set(np.argpartition(np.abs(y), -k)[-k:].tolist())
    pred_top = set(np.argpartition(np.abs(x), -k)[-k:].tolist())
    return float(len(target_top & pred_top) / k)


def matrix_metrics(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    return {
        "relative_frobenius_error": masked_relative_frobenius(predicted, target, mask),
        "rmse": masked_rmse(predicted, target, mask),
        "pearson": masked_pearson(predicted, target, mask),
        "spearman": masked_spearman(predicted, target, mask),
        "sign_agreement": masked_sign_agreement(predicted, target, mask),
        "balanced_sign_accuracy": masked_balanced_sign_accuracy(predicted, target, mask),
        "majority_sign_baseline": majority_sign_baseline(target, mask),
        "leading_mode_similarity": leading_mode_similarity(predicted, target, mask),
        "strongest_5pct_recovery": strongest_response_recovery(predicted, target, mask, 0.05),
        "strongest_10pct_recovery": strongest_response_recovery(predicted, target, mask, 0.10),
    }


def transform_like_empirical(raw_matrix: np.ndarray, scale: float) -> np.ndarray:
    return np.tanh(raw_matrix / scale)


def inverse_transform_to_raw(transformed_matrix: np.ndarray, scale: float) -> np.ndarray:
    clipped = np.clip(transformed_matrix, -0.999999, 0.999999)
    return scale * np.arctanh(clipped)


def pair_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))
