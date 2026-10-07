from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pandas as pd

from .models import TARGET, load_dataset
from .safeguards import CANDIDATE_ROOT, peak_rss


M0_VARIANT = "current_state_plus_latest_event"
DISCRETE_SURFACE = [
    "prefix",
    "collector",
    "peer_as",
    "current_route_present",
    "current_path_length",
    "current_path_bucket",
    "current_origin",
    "current_community_count",
    "latest_local_type",
    "latest_local_changed",
]
FATE_COLUMNS = [
    "hist_outstanding_count",
    "hist_outstanding_net",
    "hist_outstanding_unique",
    "hist_order_transitions",
    "hist_route_presence_mismatch",
    "hist_path_mismatch",
    "hist_origin_mismatch",
    "hist_remote_latest_type",
]


def _iso(timestamp: int | float) -> str:
    return datetime.fromtimestamp(int(timestamp), timezone.utc).isoformat()


def find_matched_witnesses(frame: pd.DataFrame, maximum_pairs: int = 100) -> tuple[pd.DataFrame, dict]:
    test = frame[frame["split"] == "test"].copy()
    predictions = pd.read_parquet(
        CANDIDATE_ROOT / "data" / "processed" / "primary_predictions.parquet"
    )
    baseline = predictions[
        (predictions["family"] == "hist_gradient_boosting")
        & (predictions["variant"] == M0_VARIANT)
    ][["sample_timestamp", "prefix", "collector", "episode_id", "probability"]]
    test = test.merge(
        baseline,
        on=["sample_timestamp", "prefix", "collector", "episode_id"],
        how="left",
        validate="one_to_one",
    )
    pairs: list[dict] = []
    used: set[int] = set()
    exact_count = 0
    for _, group in test.sort_values("sample_timestamp").groupby(DISCRETE_SURFACE, sort=False):
        indexes = list(group.index)
        for left_position, left_index in enumerate(indexes):
            if left_index in used:
                continue
            left = test.loc[left_index]
            for right_index in indexes[left_position + 1 :]:
                if right_index in used:
                    continue
                right = test.loc[right_index]
                delta = int(right["sample_timestamp"] - left["sample_timestamp"])
                if delta > 300:
                    break
                if abs(float(right["latest_local_age"] - left["latest_local_age"])) > 5:
                    continue
                if int(right[TARGET]) == int(left[TARGET]):
                    continue
                fate_differences = [
                    name for name in FATE_COLUMNS if right[name] != left[name]
                ]
                if not fate_differences:
                    continue
                if abs(float(right["probability"] - left["probability"])) > 0.02:
                    continue
                exact = bool(
                    delta == 0
                    and right["latest_local_age"] == left["latest_local_age"]
                    and right["time_sin"] == left["time_sin"]
                    and right["time_cos"] == left["time_cos"]
                )
                exact_count += int(exact)
                pairs.append(
                    {
                        "pair_id": len(pairs) + 1,
                        "prefix": left["prefix"],
                        "collector": left["collector"],
                        "episode_id_a": left["episode_id"],
                        "episode_id_b": right["episode_id"],
                        "time_a_utc": _iso(left["sample_timestamp"]),
                        "time_b_utc": _iso(right["sample_timestamp"]),
                        "wall_clock_delta_seconds": delta,
                        "latest_age_a": left["latest_local_age"],
                        "latest_age_b": right["latest_local_age"],
                        "m0_probability_a": left["probability"],
                        "m0_probability_b": right["probability"],
                        "outstanding_a": left["hist_outstanding_count"],
                        "outstanding_b": right["hist_outstanding_count"],
                        "outstanding_net_a": left["hist_outstanding_net"],
                        "outstanding_net_b": right["hist_outstanding_net"],
                        "path_mismatch_a": left["hist_path_mismatch"],
                        "path_mismatch_b": right["hist_path_mismatch"],
                        "target_a": int(left[TARGET]),
                        "target_b": int(right[TARGET]),
                        "different_fate_fields": ";".join(fate_differences),
                        "exact_surface_state": exact,
                    }
                )
                used.update({left_index, right_index})
                break
            if len(pairs) >= maximum_pairs:
                break
        if len(pairs) >= maximum_pairs:
            break
    pair_frame = pd.DataFrame(pairs)
    balance = {
        "candidate_pairs": len(pair_frame),
        "exact_pairs": exact_count,
        "unique_prefixes": int(pair_frame["prefix"].nunique()) if not pair_frame.empty else 0,
        "unique_collectors": int(pair_frame["collector"].nunique()) if not pair_frame.empty else 0,
        "maximum_wall_clock_delta": int(pair_frame["wall_clock_delta_seconds"].max())
        if not pair_frame.empty
        else None,
        "maximum_latest_age_delta": float(
            (pair_frame["latest_age_a"] - pair_frame["latest_age_b"]).abs().max()
        )
        if not pair_frame.empty
        else None,
        "maximum_m0_probability_delta": float(
            (pair_frame["m0_probability_a"] - pair_frame["m0_probability_b"]).abs().max()
        )
        if not pair_frame.empty
        else None,
    }
    return pair_frame, balance


def find_state_cases(frame: pd.DataFrame, maximum_per_type: int = 10) -> pd.DataFrame:
    rows: list[dict] = []
    ordered = frame.sort_values(["prefix", "collector", "sample_timestamp"])
    for (prefix, collector), group in ordered.groupby(["prefix", "collector"], sort=False):
        prior = None
        for item in group.to_dict("records"):
            if prior is not None:
                case_type = None
                if prior["hist_outstanding_net"] * item["hist_outstanding_net"] < 0:
                    case_type = "direction_reversal"
                elif item["hist_outstanding_count"] > prior["hist_outstanding_count"]:
                    case_type = "enhancement"
                elif (
                    prior["hist_outstanding_count"] > 0
                    and item["hist_outstanding_count"] < prior["hist_outstanding_count"]
                ):
                    case_type = "consumption_or_expiry"
                if case_type and sum(row["case_type"] == case_type for row in rows) < maximum_per_type:
                    rows.append(
                        {
                            "case_id": len(rows) + 1,
                            "case_type": case_type,
                            "prefix": prefix,
                            "collector": collector,
                            "start_utc": _iso(prior["sample_timestamp"]),
                            "end_utc": _iso(item["sample_timestamp"]),
                            "outstanding_before": prior["hist_outstanding_count"],
                            "outstanding_after": item["hist_outstanding_count"],
                            "net_before": prior["hist_outstanding_net"],
                            "net_after": item["hist_outstanding_net"],
                            "path_mismatch_before": prior["hist_path_mismatch"],
                            "path_mismatch_after": item["hist_path_mismatch"],
                            "future_change_30_before": prior[TARGET],
                            "future_change_30_after": item[TARGET],
                        }
                    )
            prior = item
    return pd.DataFrame(rows)


def run_witness_analysis() -> dict:
    started = time.perf_counter()
    frame = load_dataset()
    pairs, balance = find_matched_witnesses(frame)
    cases = find_state_cases(frame)
    tables = CANDIDATE_ROOT / "results" / "tables"
    pairs.to_csv(tables / "matched_witnesses.csv", index=False)
    cases.to_csv(tables / "case_catalog.csv", index=False)
    payload = {
        "stage": "witness_analysis",
        "status": "PASS",
        "matched_balance": balance,
        "case_counts": cases["case_type"].value_counts().to_dict() if not cases.empty else {},
        "peak_rss_bytes": peak_rss(),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (CANDIDATE_ROOT / "results" / "logs" / "08_witnesses.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload
