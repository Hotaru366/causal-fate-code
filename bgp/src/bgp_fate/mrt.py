from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import mrtparse


def coded_name(value: Any) -> str:
    if isinstance(value, dict) and value:
        return str(next(iter(value.values())))
    if isinstance(value, (list, tuple)) and len(value) > 1:
        return str(value[1])
    return str(value)


def coded_number(value: Any) -> int:
    if isinstance(value, dict) and value:
        return int(next(iter(value.keys())))
    if isinstance(value, (list, tuple)) and value:
        return int(value[0])
    return int(value)


def prefix_text(nlri: dict) -> str:
    return f"{nlri['prefix']}/{int(nlri['length'])}"


def timestamp_number(value: Any) -> int:
    return coded_number(value)


def attr_name(attr: dict) -> str:
    return coded_name(attr.get("type", ""))


def as_path(attributes: Iterable[dict]) -> tuple[int, ...]:
    values: list[int] = []
    for attr in attributes:
        if attr_name(attr) not in {"AS_PATH", "AS4_PATH"}:
            continue
        for segment in attr.get("value", []):
            values.extend(int(item) for item in segment.get("value", []))
        if attr_name(attr) == "AS_PATH":
            break
    return tuple(values)


def communities(attributes: Iterable[dict]) -> tuple[str, ...]:
    result: list[str] = []
    for attr in attributes:
        if attr_name(attr) in {"COMMUNITY", "LARGE_COMMUNITY"}:
            result.extend(str(item) for item in attr.get("value", []))
    return tuple(result)


def update_nlri(message: dict) -> tuple[list[dict], list[dict]]:
    announcements = list(message.get("nlri", []))
    withdrawals = list(message.get("withdrawn_routes", []))
    for attr in message.get("path_attributes", []):
        name = attr_name(attr)
        value = attr.get("value", {})
        if name == "MP_REACH_NLRI":
            announcements.extend(value.get("nlri", []))
        elif name == "MP_UNREACH_NLRI":
            withdrawals.extend(value.get("withdrawn_routes", []))
    return announcements, withdrawals


def iter_update_events(
    path: Path, collector: str, stats: dict[str, int] | None = None
) -> Iterator[dict]:
    reader = mrtparse.Reader(str(path))
    try:
        for record_index, entry in enumerate(reader):
            if entry.err:
                if stats is not None:
                    stats["parse_errors"] = stats.get("parse_errors", 0) + 1
                continue
            if stats is not None:
                stats["records"] = stats.get("records", 0) + 1
            data = entry.data
            message = data.get("bgp_message")
            if not message or coded_name(message.get("type")) != "UPDATE":
                continue
            attributes = message.get("path_attributes", [])
            path_value = as_path(attributes)
            comm_value = communities(attributes)
            announcements, withdrawals = update_nlri(message)
            base = {
                "collector": collector,
                "timestamp": timestamp_number(data["timestamp"]),
                "peer_as": int(data["peer_as"]),
                "peer_ip": str(data["peer_ip"]),
                "record_index": record_index,
                "as_path": " ".join(map(str, path_value)),
                "communities": " ".join(comm_value),
                "origin": path_value[-1] if path_value else None,
            }
            for ordinal, nlri in enumerate(withdrawals):
                yield {**base, "event_type": "W", "prefix": prefix_text(nlri), "ordinal": ordinal}
            for ordinal, nlri in enumerate(announcements):
                yield {**base, "event_type": "A", "prefix": prefix_text(nlri), "ordinal": ordinal}
    finally:
        try:
            reader.f.close()
        except Exception:
            pass


def iter_rib_routes(
    path: Path,
    collector: str,
    selected: set[str],
    stats: dict[str, int] | None = None,
) -> Iterator[dict]:
    peers: list[dict] = []
    reader = mrtparse.Reader(str(path))
    try:
        for record_index, entry in enumerate(reader):
            if entry.err:
                if stats is not None:
                    stats["parse_errors"] = stats.get("parse_errors", 0) + 1
                continue
            if stats is not None:
                stats["records"] = stats.get("records", 0) + 1
            data = entry.data
            if "peer_entries" in data:
                peers = list(data["peer_entries"])
                continue
            if "rib_entries" not in data or "prefix" not in data or "length" not in data:
                continue
            prefix = f"{data['prefix']}/{int(data['length'])}"
            if prefix not in selected:
                continue
            for rib_entry in data["rib_entries"]:
                index = int(rib_entry["peer_index"])
                if index >= len(peers):
                    continue
                peer = peers[index]
                attrs = rib_entry.get("path_attributes", [])
                path_value = as_path(attrs)
                yield {
                    "collector": collector,
                    "prefix": prefix,
                    "peer_as": int(peer["peer_as"]),
                    "peer_ip": str(peer["peer_ip"]),
                    "as_path": " ".join(map(str, path_value)),
                    "communities": " ".join(communities(attrs)),
                    "origin": path_value[-1] if path_value else None,
                    "originated_time": timestamp_number(rib_entry["originated_time"]),
                    "record_index": record_index,
                }
    finally:
        try:
            reader.f.close()
        except Exception:
            pass


def inspect_records(path: Path, limit: int = 3) -> list[dict]:
    samples: list[dict] = []
    reader = mrtparse.Reader(str(path))
    try:
        for entry in reader:
            if entry.err:
                continue
            data = entry.data
            samples.append(
                {
                    "mrt_type": coded_name(data.get("type")),
                    "mrt_subtype": coded_name(data.get("subtype")),
                    "top_level_fields": sorted(data),
                    "bgp_message_type": coded_name(data["bgp_message"]["type"])
                    if data.get("bgp_message")
                    else None,
                    "bgp_message_fields": sorted(data.get("bgp_message", {})),
                }
            )
            if len(samples) >= limit:
                break
    finally:
        try:
            reader.f.close()
        except Exception:
            pass
    return samples
