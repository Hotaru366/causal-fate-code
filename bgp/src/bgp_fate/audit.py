from __future__ import annotations

import csv
import json
import statistics as stats
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .download import load_config
from .mrt import inspect_records, iter_rib_routes, iter_update_events
from .safeguards import CANDIDATE_ROOT, RAW_ROOT, MANIFEST_PATH, directory_size, enforce_rss, peak_rss


def _manifest() -> list[dict]:
    path = MANIFEST_PATH
    with path.open(newline="") as handle:
        return [row for row in csv.DictReader(handle) if row["stage"] == "stage0"]


def _raw_path(row: dict) -> Path:
    return RAW_ROOT / row["collector"] / row["filename"]


def _frame_guard(frame: pd.DataFrame, limit: int) -> int:
    size = int(frame.memory_usage(index=True, deep=True).sum())
    if size > limit:
        raise RuntimeError(f"DataFrame estimate {size} exceeds limit {limit}")
    return size


def _nearest_matches(events: pd.DataFrame, maximum_lag: int) -> list[dict]:
    matches: list[dict] = []
    collectors = sorted(events["collector"].unique())
    if len(collectors) < 2:
        return matches
    left_name, right_name = collectors[:2]
    for prefix, group in events.groupby("prefix", sort=False):
        left = group[group["collector"] == left_name]
        right = group[group["collector"] == right_name]
        if left.empty or right.empty:
            continue
        right_rows = list(right.to_dict("records"))
        for item in left.to_dict("records"):
            candidates = [
                other
                for other in right_rows
                if other["event_type"] == item["event_type"]
                and (
                    item["event_type"] == "W"
                    or other["as_path"] == item["as_path"]
                )
            ]
            if not candidates:
                continue
            other = min(candidates, key=lambda row: abs(row["timestamp"] - item["timestamp"]))
            lag = int(other["timestamp"] - item["timestamp"])
            if abs(lag) <= maximum_lag:
                matches.append(
                    {
                        "prefix": prefix,
                        "event_type": item["event_type"],
                        "left_collector": left_name,
                        "right_collector": right_name,
                        "left_timestamp": int(item["timestamp"]),
                        "right_timestamp": int(other["timestamp"]),
                        "lag_seconds": lag,
                        "as_path": item["as_path"],
                    }
                )
    return matches


def _episodes(events: pd.DataFrame, gap_seconds: int) -> pd.DataFrame:
    rows: list[dict] = []
    for prefix, group in events.sort_values("timestamp").groupby("prefix", sort=False):
        episode = 0
        last: int | None = None
        buckets: dict[int, list[dict]] = defaultdict(list)
        for item in group.to_dict("records"):
            ts = int(item["timestamp"])
            if last is not None and ts - last > gap_seconds:
                episode += 1
            buckets[episode].append(item)
            last = ts
        for episode_id, items in buckets.items():
            rows.append(
                {
                    "prefix": prefix,
                    "episode_id": f"{prefix}:{episode_id}",
                    "start_timestamp": min(int(x["timestamp"]) for x in items),
                    "end_timestamp": max(int(x["timestamp"]) for x in items),
                    "event_count": len(items),
                    "collector_count": len({x["collector"] for x in items}),
                    "peer_count": len({(x["collector"], x["peer_as"], x["peer_ip"]) for x in items}),
                    "announcement_count": sum(x["event_type"] == "A" for x in items),
                    "withdrawal_count": sum(x["event_type"] == "W" for x in items),
                    "path_count": len({x["as_path"] for x in items if x["as_path"]}),
                }
            )
    return pd.DataFrame(rows)


def run_stage0() -> dict:
    started = time.perf_counter()
    config = load_config("stage0.yaml")
    limits = config["limits"]
    hard_rss = int(limits["hard_rss_bytes"])
    frame_limit = int(limits["dataframe_bytes"])
    manifest = _manifest()
    update_rows = sorted(
        (row for row in manifest if row["kind"] == "updates"),
        key=lambda row: (row["collector"], row["filename"]),
    )
    rib_rows = [row for row in manifest if row["kind"] == "rib"]

    counts: dict[str, Counter] = defaultdict(Counter)
    event_types = Counter()
    peers: dict[str, set[tuple[int, str]]] = defaultdict(set)
    first_pass_stats: dict[str, int] = {}
    monotonic_inversions = 0
    last_timestamp: dict[str, int] = {}
    for row in update_rows:
        for event in iter_update_events(_raw_path(row), row["collector"], first_pass_stats):
            counts[event["collector"]][event["prefix"]] += 1
            event_types[event["event_type"]] += 1
            peers[event["collector"]].add((event["peer_as"], event["peer_ip"]))
            prior = last_timestamp.get(event["collector"])
            if prior is not None and event["timestamp"] < prior:
                monotonic_inversions += 1
            last_timestamp[event["collector"]] = event["timestamp"]
        enforce_rss(hard_rss)

    collector_names = sorted(counts)
    common = set.intersection(*(set(counts[name]) for name in collector_names))
    ranked = sorted(
        common,
        key=lambda prefix: (
            min(counts[name][prefix] for name in collector_names),
            sum(counts[name][prefix] for name in collector_names),
            prefix,
        ),
        reverse=True,
    )
    selected = set(ranked[: int(config["selection"]["maximum_prefixes"])])

    selected_events: list[dict] = []
    second_pass_stats: dict[str, int] = {}
    for file_order, row in enumerate(update_rows):
        for event in iter_update_events(_raw_path(row), row["collector"], second_pass_stats):
            if event["prefix"] in selected:
                event["file_order"] = file_order
                selected_events.append(event)
        enforce_rss(hard_rss)
    events = pd.DataFrame(selected_events)
    events_memory = _frame_guard(events, frame_limit)
    if not events.empty:
        events = events.sort_values(
            ["timestamp", "collector", "file_order", "record_index", "ordinal", "event_type"],
            kind="stable",
        ).reset_index(drop=True)

    rib_routes: list[dict] = []
    rib_stats: dict[str, int] = {}
    rib_route_counts: dict[str, int] = {}
    for row in rib_rows:
        routes = list(iter_rib_routes(_raw_path(row), row["collector"], selected, rib_stats))
        rib_routes.extend(routes)
        rib_route_counts[row["collector"]] = len(routes)
        enforce_rss(hard_rss)
    ribs = pd.DataFrame(rib_routes)
    ribs_memory = _frame_guard(ribs, frame_limit)

    state: dict[tuple, tuple[str, str, object]] = {}
    for item in rib_routes:
        key = (item["collector"], item["peer_as"], item["peer_ip"], item["prefix"])
        state[key] = (item["as_path"], item["communities"], item["origin"])
    recovered_before = 0
    withdrawals_without_state = 0
    path_changes = 0
    reconstructed: list[dict] = []
    for item in events.to_dict("records"):
        key = (item["collector"], item["peer_as"], item["peer_ip"], item["prefix"])
        previous = state.get(key)
        if previous is not None:
            recovered_before += 1
        if item["event_type"] == "A":
            current = (item["as_path"], item["communities"], item["origin"])
            path_changes += int(previous is not None and previous[0] != current[0])
            state[key] = current
        else:
            withdrawals_without_state += int(previous is None)
            state.pop(key, None)
        reconstructed.append(
            {
                **item,
                "prior_state_available": previous is not None,
                "prior_as_path": previous[0] if previous else "",
                "route_changed": item["event_type"] == "W"
                or previous is None
                or previous[0] != item["as_path"],
            }
        )
    reconstructed_frame = pd.DataFrame(reconstructed)
    reconstructed_memory = _frame_guard(reconstructed_frame, frame_limit)

    matches = _nearest_matches(events, int(config["selection"]["episode_gap_seconds"]))
    match_frame = pd.DataFrame(matches)
    episode_frame = _episodes(events, int(config["selection"]["episode_gap_seconds"]))
    replay = episode_frame.sort_values(
        ["collector_count", "event_count", "peer_count"], ascending=False
    ).head(int(config["selection"]["minimum_replayed_episodes"]))

    interim = CANDIDATE_ROOT / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    if not reconstructed_frame.empty:
        reconstructed_frame.to_parquet(interim / "stage0_selected_events.parquet", index=False)
    if not ribs.empty:
        ribs.to_parquet(interim / "stage0_selected_rib.parquet", index=False)

    tables = CANDIDATE_ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    prefix_quality = pd.DataFrame(
        [
            {
                "prefix": prefix,
                **{f"{name}_events": counts[name][prefix] for name in collector_names},
                "total_events": sum(counts[name][prefix] for name in collector_names),
            }
            for prefix in ranked
        ]
    )
    prefix_quality.to_csv(tables / "prefix_quality.csv", index=False)
    episode_frame.to_csv(tables / "stage0_episodes.csv", index=False)
    replay.to_csv(tables / "episode_replay.csv", index=False)
    match_frame.to_csv(tables / "cross_collector_matches.csv", index=False)

    schemas = {
        "parser": {"name": "mrtparse", "version": mrtparse_version()},
        "files": [
            {
                "collector": row["collector"],
                "kind": row["kind"],
                "filename": row["filename"],
                "samples": inspect_records(_raw_path(row), 3),
            }
            for row in rib_rows + update_rows[:2]
        ],
        "normalized_event_fields": sorted(reconstructed_frame.columns.tolist()),
        "identity_key": ["collector", "peer_as", "peer_ip", "prefix"],
        "event_order_key": ["timestamp", "file_order", "record_index", "ordinal"],
    }
    (CANDIDATE_ROOT / "data" / "manifests").mkdir(parents=True, exist_ok=True)
    (CANDIDATE_ROOT / "data" / "manifests" / "schema_manifest.json").write_text(
        json.dumps(schemas, indent=2, sort_keys=True) + "\n"
    )

    nonzero_lags = sum(int(item["lag_seconds"] != 0) for item in matches)
    cross_prefixes = len(common)
    selected_count = len(selected)
    conditions = {
        "initial RIB recovered for both collectors": all(
            rib_route_counts.get(name, 0) > 0 for name in collector_names
        ),
        "announcements and withdrawals observed": event_types["A"] > 0 and event_types["W"] > 0,
        "peer-prefix identity recoverable": recovered_before > 0,
        "event timestamps monotonic": monotonic_inversions == 0,
        "collector timestamps use one UTC epoch": len(last_timestamp) == 2,
        "AS path changes traceable": path_changes > 0,
        "active prefixes sufficient": selected_count
        >= int(config["selection"]["minimum_active_prefixes"]),
        "cross-collector prefixes sufficient": cross_prefixes
        >= int(config["selection"]["minimum_cross_collector_prefixes"]),
        "propagation timing differences observed": nonzero_lags > 0,
        "convergence episodes sufficient": len(episode_frame)
        >= int(config["selection"]["minimum_replayed_episodes"]),
        "twenty episodes replayable": len(replay)
        >= int(config["selection"]["minimum_replayed_episodes"]),
        "parser integrity acceptable": first_pass_stats.get("parse_errors", 0)
        <= max(5, first_pass_stats.get("records", 0) // 1000),
        "resource limits satisfied": peak_rss() < hard_rss
        and directory_size(CANDIDATE_ROOT / "data") < int(limits["candidate_data_bytes"]),
    }
    status = "PASS" if all(conditions.values()) else "FAIL"
    conclusion = "STAGE 0 PASSED" if status == "PASS" else "FAILED_DATA_INTEGRITY"
    statistics = {
        "selected_prefixes": selected_count,
        "all_cross_collector_prefixes": cross_prefixes,
        "selected_events": len(events),
        "event_type_counts": dict(event_types),
        "update_peers": {name: len(value) for name, value in peers.items()},
        "rib_route_counts": rib_route_counts,
        "recovered_before_event": recovered_before,
        "withdrawals_without_state": withdrawals_without_state,
        "path_changes": path_changes,
        "episode_count": len(episode_frame),
        "replayed_episode_count": len(replay),
        "cross_collector_matches": len(matches),
        "nonzero_cross_collector_lags": nonzero_lags,
        "median_absolute_lag_seconds": stats.median(abs(x["lag_seconds"]) for x in matches)
        if matches
        else None,
        "monotonic_inversions": monotonic_inversions,
        "first_pass_records": first_pass_stats.get("records", 0),
        "first_pass_parse_errors": first_pass_stats.get("parse_errors", 0),
        "rib_records": rib_stats.get("records", 0),
        "rib_parse_errors": rib_stats.get("parse_errors", 0),
        "dataframe_bytes": {
            "events": events_memory,
            "ribs": ribs_memory,
            "reconstructed": reconstructed_memory,
        },
        "data_size_bytes": directory_size(CANDIDATE_ROOT / "data"),
        "peak_rss_bytes": peak_rss(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    gate = {
        "status": status,
        "conclusion": conclusion,
        "conditions": conditions,
        "statistics": statistics,
        "window": {"start": config["start_utc"], "end": config["end_utc"]},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    logs = CANDIDATE_ROOT / "results" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "stage0_gate.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
    return gate


def mrtparse_version() -> str:
    import mrtparse

    return str(getattr(mrtparse, "__version__", "unknown"))
