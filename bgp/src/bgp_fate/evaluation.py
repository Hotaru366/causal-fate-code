from __future__ import annotations

import pandas as pd
from sklearn.metrics import log_loss
from .models import TARGET

def _group_gain(baseline: pd.DataFrame, history: pd.DataFrame, group: str) -> pd.DataFrame:
    keys = ["sample_timestamp", "prefix", "collector", "episode_id", TARGET]
    merged = baseline[keys + ["probability"]].merge(
        history[keys + ["probability"]], on=keys, suffixes=("_m0", "_m1"), validate="one_to_one"
    )
    if group == "time_block":
        merged[group] = merged["sample_timestamp"] // 1800
    rows: list[dict] = []
    for value, item in merged.groupby(group, sort=True):
        m0 = log_loss(item[TARGET], item["probability_m0"], labels=[0, 1])
        m1 = log_loss(item[TARGET], item["probability_m1"], labels=[0, 1])
        rows.append(
            {
                group: value,
                "samples": len(item),
                "positives": int(item[TARGET].sum()),
                "m0_log_loss": m0,
                "m1_log_loss": m1,
                "gain": 1 - m1 / m0,
            }
        )
    return pd.DataFrame(rows)
