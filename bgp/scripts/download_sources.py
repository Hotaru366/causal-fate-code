"""Download only missing MRT archives; verify every archive against fixed hashes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from bgp_fate.download import fetch_sources

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Check existing files without downloading')
    args = parser.parse_args()
    print(json.dumps(fetch_sources(args.verify_only), indent=2))
