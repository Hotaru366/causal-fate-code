"""Populate the user's external Hugging Face cache with the fixed model and data."""
import argparse
import json
import os
from pathlib import Path


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=project/'config/default.json')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    cache = Path(os.environ.get('HF_HOME', Path.home()/'.cache/huggingface')).expanduser()
    os.environ.setdefault('HF_HOME', str(cache))
    os.environ.setdefault('HF_DATASETS_CACHE', str(cache/'datasets'))
    from huggingface_hub import snapshot_download
    from datasets import load_dataset
    snapshot_download(cfg['model_id'], revision=cfg['model_revision'],
                      allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
    load_dataset(cfg['dataset_name'], cfg['dataset_config'], revision=cfg['dataset_revision'])
    print(f'Model and dataset cached outside the code tree: {cache}')


if __name__ == '__main__':
    main()
