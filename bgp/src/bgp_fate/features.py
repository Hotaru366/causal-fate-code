from __future__ import annotations

import bisect
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .download import load_config
from .mrt import iter_rib_routes, iter_update_events
from .safeguards import CANDIDATE_ROOT, RAW_ROOT, MANIFEST_PATH, directory_size, enforce_rss, peak_rss


def _manifest_rows(kind: str) -> list[dict]:
    path = MANIFEST_PATH
    with path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["kind"] == kind]
    return sorted(rows, key=lambda row: (row["collector"], row["filename"]))


def _raw(row: dict) -> Path:
    return RAW_ROOT / row["collector"] / row["filename"]


def _stable_bucket(value: str, buckets: int = 2048) -> int:
    if not value:
        return 0
    digest = hashlib.blake2b(value.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets + 1


def _fingerprint(event: dict) -> tuple[str, str]:
    return (event["event_type"], event["as_path"] if event["event_type"] == "A" else "")


def _remove_match(queue: deque[dict], fingerprint: tuple[str, str]) -> bool:
    for index, item in enumerate(queue):
        if item["fingerprint"] == fingerprint:
            del queue[index]
            return True
    return False


def _expire(queue: deque[dict], now: int, maximum_age: int) -> int:
    removed = 0
    while queue and now - int(queue[0]["timestamp"]) > maximum_age:
        queue.popleft()
        removed += 1
    return removed


def _advance_fronts(outstanding, group, collectors, now, maximum_age):
    """Process a timestamp group symmetrically before reading either front.

    Each local event consumes the oldest matching remote occurrence, at most once.
    Events at the same timestamp are all available before matching; collector
    iteration order must not create a spurious one-sided unmatched item.
    """
    for queue in outstanding.values():
        _expire(queue, now, maximum_age)
    for event in group:
        other = next(name for name in collectors if name != event["collector"])
        outstanding[other].append({
            "timestamp": now, "event_type": event["event_type"],
            "as_path": event["as_path"], "fingerprint": _fingerprint(event),
        })
    for event in group:
        _remove_match(outstanding[event["collector"]], _fingerprint(event))


def _recent_count(history: deque[dict], now: int, seconds: int, key: str | None = None) -> int:
    return sum(
        item["timestamp"] >= now - seconds and (key is None or item["event_type"] == key)
        for item in history
    )


def _ordered_features(remote_history: deque[dict], now: int) -> dict:
    recent = list(remote_history)[-5:]
    values: dict[str, float | int] = {}
    for index in range(5):
        if index < len(recent):
            item = recent[-1 - index]
            values[f"raw_{index + 1}_type"] = 1 if item["event_type"] == "A" else -1
            values[f"raw_{index + 1}_age"] = now - int(item["timestamp"])
            values[f"raw_{index + 1}_path_bucket"] = _stable_bucket(item["as_path"])
        else:
            values[f"raw_{index + 1}_type"] = 0
            values[f"raw_{index + 1}_age"] = 3600
            values[f"raw_{index + 1}_path_bucket"] = 0
    return values


def _history_state(
    outstanding: deque[dict],
    remote_history: deque[dict],
    local_state: dict,
    remote_state: dict,
    now: int,
) -> dict:
    types = [item["event_type"] for item in outstanding]
    transitions = sum(left != right for left, right in zip(types, types[1:]))
    fingerprints = {item["fingerprint"] for item in outstanding}
    remote_last = remote_history[-1] if remote_history else None
    local_path = local_state.get("as_path", "") if local_state.get("present") else ""
    remote_path = remote_state.get("as_path", "") if remote_state.get("present") else ""
    return {
        "hist_outstanding_count": len(outstanding),
        "hist_outstanding_oldest_age": now - int(outstanding[0]["timestamp"])
        if outstanding
        else 0,
        "hist_outstanding_latest_age": now - int(outstanding[-1]["timestamp"])
        if outstanding
        else 0,
        "hist_outstanding_net": types.count("A") - types.count("W"),
        "hist_outstanding_unique": len(fingerprints),
        "hist_order_transitions": transitions,
        "hist_remote_since_local_30": _recent_count(remote_history, now, 30),
        "hist_remote_since_local_120": _recent_count(remote_history, now, 120),
        "hist_remote_announcements_120": _recent_count(remote_history, now, 120, "A"),
        "hist_remote_withdrawals_120": _recent_count(remote_history, now, 120, "W"),
        "hist_remote_latest_type": 1
        if remote_last and remote_last["event_type"] == "A"
        else (-1 if remote_last else 0),
        "hist_remote_latest_age": now - int(remote_last["timestamp"])
        if remote_last
        else 3600,
        "hist_route_presence_mismatch": int(
            bool(local_state.get("present")) != bool(remote_state.get("present"))
        ),
        "hist_path_mismatch": int(local_path != remote_path),
        "hist_origin_mismatch": int(
            local_state.get("origin") != remote_state.get("origin")
        ),
        "hist_remote_path_length": len(remote_path.split()) if remote_path else 0,
        "hist_remote_path_bucket": _stable_bucket(remote_path),
        **_ordered_features(outstanding, now),
    }


def _base_features(
    collector: str,
    prefix: str,
    peer: tuple[int, str],
    state: dict,
    history: deque[dict],
    now: int,
) -> dict:
    path = state.get("as_path", "") if state.get("present") else ""
    last_time = state.get("last_timestamp")
    hour = datetime.fromtimestamp(now, timezone.utc).hour + datetime.fromtimestamp(
        now, timezone.utc
    ).minute / 60
    return {
        "collector": collector,
        "prefix": prefix,
        "peer_as": peer[0],
        "peer_ip": peer[1],
        "sample_timestamp": now,
        "max_feature_timestamp": now,
        "current_route_present": int(bool(state.get("present"))),
        "current_path_length": len(path.split()) if path else 0,
        "current_path_bucket": _stable_bucket(path),
        "current_origin": int(state.get("origin") or 0),
        "current_community_count": int(state.get("community_count", 0)),
        "latest_local_type": 1
        if state.get("last_type") == "A"
        else (-1 if state.get("last_type") == "W" else 0),
        "latest_local_age": now - int(last_time) if last_time is not None else 86400,
        "latest_local_changed": int(bool(state.get("last_changed"))),
        "lag_local_events_30": _recent_count(history, now, 30),
        "lag_local_events_120": _recent_count(history, now, 120),
        "lag_local_announcements_120": _recent_count(history, now, 120, "A"),
        "lag_local_withdrawals_120": _recent_count(history, now, 120, "W"),
        "time_sin": math.sin(2 * math.pi * hour / 24),
        "time_cos": math.cos(2 * math.pi * hour / 24),
    }


def build_stage1_dataset() -> dict:
    started = time.perf_counter()
    stage0 = load_config("stage0.yaml")
    stage1 = load_config("stage1.yaml")
    hard_rss = int(stage0["limits"]["hard_rss_bytes"])
    frame_limit = int(stage0["limits"]["dataframe_bytes"])
    update_rows = _manifest_rows("updates")
    rib_rows = _manifest_rows("rib")
    counts: dict[str, Counter] = defaultdict(Counter)
    scan_stats: dict[str, int] = {}
    for row in update_rows:
        for event in iter_update_events(_raw(row), row["collector"], scan_stats):
            counts[event["collector"]][event["prefix"]] += 1
        enforce_rss(hard_rss)
    collectors = sorted(counts)
    common = set.intersection(*(set(counts[name]) for name in collectors))
    ranked = sorted(
        common,
        key=lambda prefix: (
            min(counts[name][prefix] for name in collectors),
            sum(counts[name][prefix] for name in collectors),
            prefix,
        ),
        reverse=True,
    )
    selected_count = int(stage1["prefix_count_min"])
    selected = set(ranked[:selected_count])
    if len(selected) < selected_count:
        raise RuntimeError(f"only {len(selected)} shared active prefixes")

    selected_events: list[dict] = []
    peer_prefixes: dict[str, dict[tuple[int, str], set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    peer_counts: dict[str, Counter] = defaultdict(Counter)
    second_stats: dict[str, int] = {}
    for file_order, row in enumerate(update_rows):
        for event in iter_update_events(_raw(row), row["collector"], second_stats):
            if event["prefix"] not in selected:
                continue
            peer = (event["peer_as"], event["peer_ip"])
            peer_prefixes[event["collector"]][peer].add(event["prefix"])
            peer_counts[event["collector"]][peer] += 1
            event["file_order"] = file_order
            selected_events.append(event)
        enforce_rss(hard_rss)
    anchors = {
        collector: max(
            peer_counts[collector],
            key=lambda peer: (
                len(peer_prefixes[collector][peer]), peer_counts[collector][peer], peer
            ),
        )
        for collector in collectors
    }
    anchor_events = [
        event
        for event in selected_events
        if (event["peer_as"], event["peer_ip"]) == anchors[event["collector"]]
    ]
    del selected_events
    anchor_events.sort(
        key=lambda row: (
            row["prefix"], row["timestamp"], row["collector"], row["file_order"], row["record_index"], row["ordinal"]
        )
    )

    rib_routes: list[dict] = []
    rib_stats: dict[str, int] = {}
    for row in rib_rows:
        for route in iter_rib_routes(_raw(row), row["collector"], selected, rib_stats):
            if (route["peer_as"], route["peer_ip"]) == anchors[row["collector"]]:
                rib_routes.append(route)
        enforce_rss(hard_rss)
    initial = {
        (route["prefix"], route["collector"]): {
            "present": True,
            "as_path": route["as_path"],
            "origin": route["origin"],
            "community_count": len(route["communities"].split()) if route["communities"] else 0,
            "last_timestamp": None,
            "last_type": None,
            "last_changed": False,
        }
        for route in rib_routes
    }

    samples: list[dict] = []
    changes: dict[tuple[str, str], list[int]] = defaultdict(list)
    episode_ranges: dict[str, list[int]] = {}
    by_prefix: dict[str, list[dict]] = defaultdict(list)
    for event in anchor_events:
        by_prefix[event["prefix"]].append(event)
    for prefix, prefix_events in by_prefix.items():
        local_states = {
            collector: dict(initial.get((prefix, collector), {"present": False}))
            for collector in collectors
        }
        histories = {collector: deque(maxlen=10000) for collector in collectors}
        outstanding = {collector: deque(maxlen=10000) for collector in collectors}
        episode_index = 0
        episode_start = int(prefix_events[0]["timestamp"])
        previous_ts: int | None = None
        cursor = 0
        while cursor < len(prefix_events):
            now = int(prefix_events[cursor]["timestamp"])
            if previous_ts is not None and now - previous_ts > int(stage1["episode_gap_seconds"]):
                prior_id = f"{prefix}:{episode_index}"
                episode_ranges[prior_id][1] = previous_ts
                episode_index += 1
                episode_start = now
                for queue in outstanding.values():
                    queue.clear()
                for history in histories.values():
                    history.clear()
            for queue in outstanding.values():
                _expire(queue, now, int(stage1["episode_gap_seconds"]))
            episode_id = f"{prefix}:{episode_index}"
            episode_ranges.setdefault(episode_id, [episode_start, now])
            episode_ranges[episode_id][1] = now
            group: list[dict] = []
            while cursor < len(prefix_events) and int(prefix_events[cursor]["timestamp"]) == now:
                group.append(prefix_events[cursor])
                cursor += 1
            _advance_fronts(outstanding, group, collectors, now, int(stage1["episode_gap_seconds"]))
            source_collectors = {event["collector"] for event in group}
            for event in group:
                source = event["collector"]
                other = next(name for name in collectors if name != source)
                fingerprint = _fingerprint(event)
                histories[source].append(
                    {
                        "timestamp": now,
                        "event_type": event["event_type"],
                        "as_path": event["as_path"],
                        "fingerprint": fingerprint,
                    }
                )
                state = local_states[source]
                old_present = bool(state.get("present"))
                old_path = state.get("as_path", "") if old_present else ""
                if event["event_type"] == "A":
                    new_present = True
                    new_path = event["as_path"]
                    state.update(
                        {
                            "present": True,
                            "as_path": new_path,
                            "origin": event["origin"],
                            "community_count": len(event["communities"].split())
                            if event["communities"]
                            else 0,
                        }
                    )
                else:
                    new_present = False
                    new_path = ""
                    state.update({"present": False, "as_path": "", "origin": None, "community_count": 0})
                changed = old_present != new_present or old_path != new_path
                state.update(
                    {
                        "last_timestamp": now,
                        "last_type": event["event_type"],
                        "last_changed": changed,
                    }
                )
                if changed and (not changes[(prefix, source)] or changes[(prefix, source)][-1] != now):
                    changes[(prefix, source)].append(now)
            for source in source_collectors:
                local = next(name for name in collectors if name != source)
                local_state = local_states[local]
                if local_state.get("last_timestamp") is None and not local_state.get("present"):
                    continue
                row = _base_features(
                    local, prefix, anchors[local], local_state, histories[local], now
                )
                row.update(
                    _history_state(
                        outstanding[local], histories[source], local_state, local_states[source], now
                    )
                )
                row["remote_collector"] = source
                row["episode_id"] = episode_id
                samples.append(row)
            previous_ts = now
        episode_ranges[f"{prefix}:{episode_index}"][1] = int(prefix_events[-1]["timestamp"])
        enforce_rss(hard_rss)

    end_ts = int(datetime.fromisoformat(stage1["end_utc"]).timestamp())
    for row in samples:
        future = changes[(row["prefix"], row["collector"])]
        index = bisect.bisect_right(future, int(row["sample_timestamp"]))
        next_change = future[index] if index < len(future) else None
        row["next_change_timestamp"] = next_change
        row["target_change_30"] = int(
            next_change is not None and next_change <= int(row["sample_timestamp"]) + 30
        )
        row["target_change_120"] = int(
            next_change is not None and next_change <= int(row["sample_timestamp"]) + 120
        )

    frame = pd.DataFrame(samples)
    del samples
    frame = frame[frame["sample_timestamp"] <= end_ts - 120].copy()
    start_ts = int(datetime.fromisoformat(stage1["start_utc"]).timestamp())
    boundary1 = start_ts + int((end_ts - start_ts) * float(stage1["train_fraction"]))
    boundary2 = start_ts + int(
        (end_ts - start_ts)
        * (float(stage1["train_fraction"]) + float(stage1["validation_fraction"]))
    )
    split_by_episode: dict[str, str] = {}
    for episode_id, (episode_start, episode_end) in episode_ranges.items():
        if episode_end < boundary1:
            split_by_episode[episode_id] = "train"
        elif episode_start >= boundary1 and episode_end < boundary2:
            split_by_episode[episode_id] = "validation"
        elif episode_start >= boundary2:
            split_by_episode[episode_id] = "test"
        else:
            split_by_episode[episode_id] = "dropped_boundary_episode"
    frame["split"] = frame["episode_id"].map(split_by_episode)
    dropped = int((frame["split"] == "dropped_boundary_episode").sum())
    frame = frame[frame["split"] != "dropped_boundary_episode"].reset_index(drop=True)
    frame_bytes = int(frame.memory_usage(index=True, deep=True).sum())
    if frame_bytes > frame_limit:
        raise RuntimeError(f"Stage 1 DataFrame {frame_bytes} exceeds {frame_limit}")
    if (frame["max_feature_timestamp"] > frame["sample_timestamp"]).any():
        raise RuntimeError("future feature timestamp detected")
    if (frame["next_change_timestamp"].dropna() <= frame.loc[frame["next_change_timestamp"].notna(), "sample_timestamp"]).any():
        raise RuntimeError("non-future target detected")
    overlap = frame.groupby("episode_id")["split"].nunique().max()
    if overlap != 1:
        raise RuntimeError("episode crosses split")

    processed = CANDIDATE_ROOT / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(processed / "stage1_dataset.parquet", index=False)
    tables = CANDIDATE_ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "prefix": prefix,
                **{f"{collector}_events": counts[collector][prefix] for collector in collectors},
                "selection_rank": index + 1,
            }
            for index, prefix in enumerate(ranked[:selected_count])
        ]
    ).to_csv(tables / "selected_prefixes.csv", index=False)
    pd.DataFrame(
        [
            {
                "collector": collector,
                "peer_as": peer[0],
                "peer_ip": peer[1],
                "selected_prefix_coverage": len(peer_prefixes[collector][peer]),
                "selected_events": peer_counts[collector][peer],
            }
            for collector, peer in anchors.items()
        ]
    ).to_csv(tables / "anchor_peers.csv", index=False)
    counts_table = (
        frame.groupby("split", observed=True)
        .agg(samples=("target_change_30", "size"), positives=("target_change_30", "sum"), episodes=("episode_id", "nunique"), prefixes=("prefix", "nunique"))
        .reset_index()
    )
    counts_table.to_csv(tables / "sample_counts.csv", index=False)
    payload = {
        "stage": "build_stage1_dataset",
        "status": "PASS",
        "selected_prefixes": len(selected),
        "common_active_prefixes": len(common),
        "anchors": {
            collector: {"peer_as": peer[0], "peer_ip": peer[1]} for collector, peer in anchors.items()
        },
        "anchor_events": len(anchor_events),
        "rib_routes": len(rib_routes),
        "samples": len(frame),
        "dropped_boundary_samples": dropped,
        "split_counts": counts_table.to_dict("records"),
        "frame_bytes": frame_bytes,
        "scan_records": scan_stats.get("records", 0),
        "scan_parse_errors": scan_stats.get("parse_errors", 0),
        "second_pass_parse_errors": second_stats.get("parse_errors", 0),
        "rib_parse_errors": rib_stats.get("parse_errors", 0),
        "peak_rss_bytes": peak_rss(),
        "data_size_bytes": directory_size(CANDIDATE_ROOT / "data"),
        "elapsed_seconds": time.perf_counter() - started,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (CANDIDATE_ROOT / "results" / "logs" / "05_build_dataset.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload
