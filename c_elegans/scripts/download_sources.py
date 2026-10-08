#!/usr/bin/env python3
"""Obtain the pinned upstream source in the external cache."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from celegans_dtpr_v4.data import ensure_official_repo
if __name__ == "__main__":
    config = json.loads((ROOT / "config/default.json").read_text())["config"]
    print(ensure_official_repo(ROOT, config["official_repo_url"], config["official_repo_commit"]))
