"""Read fixed source metadata and fetch upstream archives into an external cache."""
import csv
import hashlib
from pathlib import Path

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .safeguards import PROJECT_ROOT, RAW_ROOT, MANIFEST_PATH


def load_config(name: str) -> dict:
    return yaml.safe_load((PROJECT_ROOT/'config'/name).read_text())


def source_rows(stage: str | None = None) -> list[dict]:
    with MANIFEST_PATH.open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if stage is None or row['stage'] == stage]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_sources(verify_only: bool = False) -> dict:
    rows = source_rows()
    limit = int(load_config('stage0.yaml')['limits']['compressed_download_bytes'])
    if sum(int(row['bytes']) for row in rows) > limit:
        raise RuntimeError('Source manifest exceeds the compressed download limit')
    with requests.Session() as session:
        retry = Retry(total=4, backoff_factor=1, status_forcelist=[429,500,502,503,504])
        session.mount('https://', HTTPAdapter(max_retries=retry))
        for row in rows:
            path = RAW_ROOT/row['collector']/row['filename']
            if not path.exists():
                if verify_only:
                    raise FileNotFoundError(f'{path}: run scripts/download_sources.py first')
                path.parent.mkdir(parents=True, exist_ok=True)
                partial = path.with_suffix(path.suffix+'.part')
                with session.get(row['url'], stream=True, timeout=(30,180)) as response:
                    response.raise_for_status()
                    with partial.open('wb') as output:
                        for chunk in response.iter_content(1024*1024):
                            output.write(chunk)
                if partial.stat().st_size != int(row['bytes']) or checksum(partial) != row['sha256']:
                    raise RuntimeError(f'Download checksum/size mismatch: {partial}')
                partial.replace(path)
            if path.stat().st_size != int(row['bytes']) or checksum(path) != row['sha256']:
                raise RuntimeError(f'Existing source checksum/size mismatch: {path}')
    return {'status':'PASS', 'verified_archives':len(rows), 'raw_directory':str(RAW_ROOT)}
