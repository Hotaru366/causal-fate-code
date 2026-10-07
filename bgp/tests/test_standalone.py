"""Synthetic regression checks; no upstream observations or network calls."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bgp_fate import audit, download, models, safeguards


def test_source_manifest_is_complete_and_stage0_is_bounded():
    rows = download.source_rows()
    assert len(rows) == 386
    assert len({r['url'] for r in rows}) == 386
    assert len([r for r in rows if r['kind'] == 'rib']) == 2
    assert {r['collector'] for r in rows} == {'route-views.chile','rrc06'}
    assert all(len(r['sha256']) == 64 and int(r['bytes']) > 0 for r in rows)
    initial = audit._manifest()
    assert len(initial) == 34
    assert all(r['stage'] == 'stage0' for r in initial)
    assert all('20250706.0' in r['filename'] for r in initial)
    assert all(r['filename'].split('.')[2] < '0200' for r in initial if r['kind'] == 'updates')


def test_verified_downloads_need_no_network_and_reject_changed_files(tmp_path, monkeypatch):
    raw = tmp_path/'raw';raw.mkdir();(raw/'collector').mkdir()
    content = b'synthetic checksum fixture'
    file = raw/'collector/update';file.write_bytes(content)
    row = {'collector':'collector','filename':'update','url':'https://example.invalid/update',
           'bytes':str(len(content)),'sha256':hashlib.sha256(content).hexdigest()}
    monkeypatch.setattr(download,'RAW_ROOT',raw)
    monkeypatch.setattr(download,'source_rows',lambda:[row])
    assert download.fetch_sources(verify_only=True)['verified_archives'] == 1
    file.write_bytes(b'changed')
    with pytest.raises(RuntimeError,match='checksum/size mismatch'):
        download.fetch_sources(verify_only=True)


def test_block_bootstrap_matches_explicit_row_resampling():
    size = 20
    base = pd.DataFrame({'sample_timestamp':np.arange(size)*1000,'prefix':'synthetic',
                         'collector':'test','episode_id':np.arange(size),
                         'target_change_30':np.arange(size)%2,
                         'probability':np.linspace(.1,.9,size)})
    alternate = base.copy();alternate['probability'] = .7*base.probability + .15
    actual = models.block_bootstrap_gain(base,alternate,1729,repetitions=25)
    # The reference independently expands the sampled blocks into observation rows.
    block = ((base.sample_timestamp-base.sample_timestamp.min())//1800).to_numpy()
    unique = np.unique(block);rng=np.random.default_rng(1729)
    from sklearn.metrics import log_loss
    draws=[]
    for _ in range(25):
        chosen=rng.choice(unique,size=len(unique),replace=True)
        indices=np.concatenate([np.flatnonzero(block==b) for b in chosen])
        l0=log_loss(base.target_change_30.iloc[indices],base.probability.iloc[indices],labels=[0,1])
        l1=log_loss(alternate.target_change_30.iloc[indices],alternate.probability.iloc[indices],labels=[0,1])
        draws.append(1-l1/l0)
    assert actual['ci_low'] == pytest.approx(np.quantile(draws,.025),abs=1e-12)
    assert actual['ci_high'] == pytest.approx(np.quantile(draws,.975),abs=1e-12)
