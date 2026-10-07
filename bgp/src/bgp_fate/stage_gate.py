from __future__ import annotations

import json
from pathlib import Path

from .safeguards import CANDIDATE_ROOT


def require_stage0_pass() -> dict:
    path = CANDIDATE_ROOT / "results" / "logs" / "stage0_gate.json"
    if not path.exists():
        raise RuntimeError("Stage 0 gate is missing")
    gate = json.loads(path.read_text())
    if gate.get("status") != "PASS":
        raise RuntimeError("Stage 0 did not pass; Stage 1 is prohibited")
    return gate
