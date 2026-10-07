"""Portable paths and resource limits for the standalone routing experiment."""
import os
import resource
import sys
from pathlib import Path

import psutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_ROOT = Path(os.environ.get('BGP_OUTPUT_DIR', PROJECT_ROOT/'outputs')).expanduser().resolve()
RAW_ROOT = Path(os.environ.get('BGP_RAW_DIR', Path.home()/'.cache/causal-fate/bgp/raw')).expanduser().resolve()
MANIFEST_PATH = PROJECT_ROOT/'config/sources.csv'


def directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink()) if path.exists() else 0


def peak_rss() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == 'darwin' else value * 1024


def enforce_rss(hard_limit: int) -> int:
    rss = psutil.Process(os.getpid()).memory_info().rss
    if rss > hard_limit:
        raise RuntimeError(f'RSS {rss} exceeds hard limit {hard_limit}')
    return rss
