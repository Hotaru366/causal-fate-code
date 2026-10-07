from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .safeguards import CANDIDATE_ROOT, peak_rss


TARGET = "target_change_30"
CATEGORICAL = ["collector", "remote_collector", "prefix"]
VISIBLE = [
    "collector",
    "remote_collector",
    "prefix",
    "peer_as",
    "current_route_present",
    "current_path_length",
    "current_path_bucket",
    "current_origin",
    "current_community_count",
    "time_sin",
    "time_cos",
]
LATEST = VISIBLE + ["latest_local_type", "latest_local_age", "latest_local_changed"]
LAGS = [
    "lag_local_events_30",
    "lag_local_events_120",
    "lag_local_announcements_120",
    "lag_local_withdrawals_120",
]
CUMULATIVE = [
    "hist_remote_since_local_30",
    "hist_remote_since_local_120",
    "hist_remote_announcements_120",
    "hist_remote_withdrawals_120",
    "hist_outstanding_net",
]
UNORDERED = CUMULATIVE + [
    "hist_outstanding_count",
    "hist_outstanding_unique",
    "hist_route_presence_mismatch",
    "hist_path_mismatch",
    "hist_origin_mismatch",
    "hist_remote_path_length",
    "hist_remote_path_bucket",
]
ORDERED_STATE = [
    "hist_outstanding_oldest_age",
    "hist_outstanding_latest_age",
    "hist_order_transitions",
    "hist_remote_latest_type",
    "hist_remote_latest_age",
]
RAW_ORDERED = [
    f"raw_{index}_{suffix}"
    for index in range(1, 6)
    for suffix in ("type", "age", "path_bucket")
]
CONTENT_ONLY = [name for name in RAW_ORDERED if not name.endswith("_age")]
FATE = UNORDERED + ORDERED_STATE


def feature_sets() -> dict[str, list[str]]:
    return {
        "current_visible_state_only": VISIBLE,
        "current_state_plus_latest_event": LATEST,
        "current_state_plus_ordinary_short_lags": LATEST + LAGS,
        "current_state_plus_simple_cumulative_history": LATEST + CUMULATIVE,
        "current_state_plus_interpretable_fate_state": LATEST + FATE,
        "current_state_plus_shuffled_history": LATEST + UNORDERED + RAW_ORDERED,
        "current_state_plus_unordered_summary": LATEST + UNORDERED,
        "current_state_plus_full_raw_ordered_history": LATEST + UNORDERED + RAW_ORDERED,
        "front_content_ordered": LATEST + UNORDERED + CONTENT_ONLY,
        "front_content_shuffled": LATEST + UNORDERED + CONTENT_ONLY,
        "front_content_inventory": LATEST + UNORDERED + CONTENT_ONLY,
        "fate_state_without_outstanding_queue": LATEST
        + [name for name in FATE if not name.startswith("hist_outstanding")],
    }


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(CANDIDATE_ROOT / "data" / "processed" / "stage1_dataset.parquet")


def _shuffle_order_state(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Permute whole event tuples within each row, retaining padding and context.

    With event ages present this is only an encoding control: times reveal order.
    Order diagnostics use the content-only feature set, omitting those ages.
    """
    result = frame.copy()
    values = result[RAW_ORDERED].to_numpy(copy=True).reshape(-1, 5, 3)
    rng = np.random.default_rng(seed)
    for row in values:
        valid = np.flatnonzero(row[:, 0] != 0)
        row[valid] = row[rng.permutation(valid)].copy()
    result[RAW_ORDERED] = values.reshape(-1, 15)
    return result


def _inventory_state(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonical multiset of (type,path) pairs, with no chronological position."""
    result = frame.copy()
    values = result[RAW_ORDERED].to_numpy(copy=True).reshape(-1, 5, 3)
    for row in values:
        valid = np.flatnonzero(row[:, 0] != 0)
        order = sorted(valid, key=lambda i: (row[i, 0], row[i, 2]))
        row[valid] = row[order].copy()
    result[RAW_ORDERED] = values.reshape(-1, 15)
    return result


def _pipeline(family: str, features: list[str], params: dict, seed: int) -> Pipeline:
    categorical = [name for name in features if name in CATEGORICAL]
    numeric = [name for name in features if name not in categorical]
    preprocess = ColumnTransformer(
        [
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ],
        remainder="drop",
    )
    if family == "logistic":
        estimator = LogisticRegression(
            C=float(params["C"]), max_iter=400, solver="lbfgs", random_state=seed
        )
    elif family == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=80,
            max_leaf_nodes=int(params["max_leaf_nodes"]),
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=seed,
        )
    else:
        raise ValueError(family)
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def parameter_grid(family: str) -> list[dict]:
    if family == "logistic":
        return [{"C": 0.1}, {"C": 1.0}]
    if family == "hist_gradient_boosting":
        return [{"max_leaf_nodes": 15}, {"max_leaf_nodes": 31}]
    raise ValueError(family)


def metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    predicted = probability >= 0.5
    return {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, probability)) if y.nunique() > 1 else float("nan"),
        "accuracy": float(accuracy_score(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
    }


@dataclass
class FitResult:
    summary: dict
    predictions: pd.DataFrame


def fit_variant(
    frame: pd.DataFrame,
    variant: str,
    family: str,
    seed: int,
    fixed_params: dict | None = None,
    target: str = TARGET,
) -> FitResult:
    if variant in {"current_state_plus_shuffled_history", "front_content_shuffled"}:
        frame = _shuffle_order_state(frame, seed)
    elif variant == "front_content_inventory":
        frame = _inventory_state(frame)
    features = feature_sets()[variant]
    train = frame[frame["split"] == "train"]
    validation = frame[frame["split"] == "validation"]
    test = frame[frame["split"] == "test"]
    grid = [fixed_params] if fixed_params is not None else parameter_grid(family)
    trials: list[dict] = []
    best: tuple[float, dict] | None = None
    for params in grid:
        model = _pipeline(family, features, params, seed)
        model.fit(train[features], train[target])
        probability = model.predict_proba(validation[features])[:, 1]
        risk = float(log_loss(validation[target], probability, labels=[0, 1]))
        trials.append({"params": params, "validation_log_loss": risk})
        if best is None or risk < best[0]:
            best = (risk, params)
    assert best is not None
    train_validation = frame[frame["split"].isin(["train", "validation"])]
    model = _pipeline(family, features, best[1], seed)
    model.fit(train_validation[features], train_validation[target])
    probability = model.predict_proba(test[features])[:, 1]
    test_metrics = metrics(test[target], probability)
    prediction = test[
        ["sample_timestamp", "prefix", "collector", "episode_id", target]
    ].copy()
    prediction["probability"] = probability
    prediction["variant"] = variant
    prediction["family"] = family
    prediction["seed"] = seed
    prediction["target"] = target
    summary = {
        "variant": variant,
        "family": family,
        "seed": seed,
        "target": target,
        "features": features,
        "feature_count": len(features),
        "best_params": best[1],
        "validation_log_loss": best[0],
        "search_trials": trials,
        **{f"test_{name}": value for name, value in test_metrics.items()},
        "train_samples": len(train),
        "validation_samples": len(validation),
        "test_samples": len(test),
    }
    return FitResult(summary, prediction)


def block_bootstrap_gain(
    baseline: pd.DataFrame,
    history: pd.DataFrame,
    seed: int,
    repetitions: int = 1000,
    block_seconds: int = 1800,
    target: str = TARGET,
) -> dict[str, float]:
    keys = ["sample_timestamp", "prefix", "collector", "episode_id", target]
    merged = baseline[keys + ["probability"]].merge(
        history[keys + ["probability"]], on=keys, suffixes=("_m0", "_m1"), validate="one_to_one"
    )
    merged["block"] = merged["sample_timestamp"] // block_seconds
    # A block bootstrap depends only on block sizes and sums of per-row losses.
    # Vectorizing these sufficient statistics preserves the row-resampling draws.
    y = merged[target].to_numpy(dtype=float)
    eps = np.finfo(float).eps
    for suffix in ("m0", "m1"):
        p = np.clip(merged[f"probability_{suffix}"].to_numpy(dtype=float), eps, 1 - eps)
        merged[f"loss_{suffix}"] = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    blocks = merged.groupby("block", sort=True)[["loss_m0", "loss_m1"]].sum().to_numpy()
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(blocks), size=(repetitions, len(blocks)))
    sampled = blocks[indexes].sum(axis=1)
    gains = 1 - sampled[:, 1] / sampled[:, 0]
    direct_m0 = log_loss(merged[target], merged["probability_m0"], labels=[0, 1])
    direct_m1 = log_loss(merged[target], merged["probability_m1"], labels=[0, 1])
    return {
        "gain": float(1 - direct_m1 / direct_m0),
        "ci_low": float(np.quantile(gains, 0.025)),
        "ci_high": float(np.quantile(gains, 0.975)),
        "bootstrap_repetitions": repetitions,
        "block_seconds": block_seconds,
        "block_count": len(blocks),
    }


def run_primary(seed: int = 1729) -> dict:
    started = time.perf_counter()
    frame = load_dataset()
    variants = [
        "current_state_plus_latest_event",
        "current_state_plus_interpretable_fate_state",
    ]
    summaries: list[dict] = []
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    gains: list[dict] = []
    for family in ("logistic", "hist_gradient_boosting"):
        for variant in variants:
            result = fit_variant(frame, variant, family, seed)
            summaries.append(result.summary)
            predictions[(family, variant)] = result.predictions
        gain = block_bootstrap_gain(
            predictions[(family, variants[0])], predictions[(family, variants[1])], seed
        )
        gains.append({"family": family, "seed": seed, **gain})
    tables = CANDIDATE_ROOT / "results" / "tables"
    pd.DataFrame(
        [
            {key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            for row in summaries
        ]
    ).to_csv(tables / "model_comparison.csv", index=False)
    pd.DataFrame(gains).to_csv(tables / "primary_gain_ci.csv", index=False)
    all_predictions = pd.concat(predictions.values(), ignore_index=True)
    all_predictions.to_parquet(CANDIDATE_ROOT / "data" / "processed" / "primary_predictions.parquet", index=False)
    payload = {
        "stage": "primary_identification",
        "status": "PASS",
        "seed": seed,
        "summaries": summaries,
        "gains": gains,
        "peak_rss_bytes": peak_rss(),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (CANDIDATE_ROOT / "results" / "logs" / "06_identification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload
