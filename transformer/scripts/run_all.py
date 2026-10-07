#!/usr/bin/env python3
"""Run exact finite transport with external Hugging Face caches."""
import argparse
import os
from pathlib import Path
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['full','smoke'], default='full')
    parser.add_argument('--config', type=Path, default=project/'config/default.json')
    parser.add_argument('--output-dir', type=Path, default=project/'outputs')
    args = parser.parse_args()
    cache = Path(os.environ.get('HF_HOME', Path.home()/'.cache/huggingface')).expanduser()
    os.environ.setdefault('HF_HOME', str(cache))
    os.environ.setdefault('HF_DATASETS_CACHE', str(cache/'datasets'))
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    sys.path.insert(0, str(project/'src'))
    from experiment import run_all
    run_all(root=args.output_dir.expanduser().resolve(), repo_root=project.parent,
            config_path=args.config.expanduser().resolve(), mode=args.mode)


if __name__ == '__main__':
    main()
