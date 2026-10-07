"""Reproduce the unmatched-front experiment from verified upstream MRT archives."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import warnings

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / 'src'))

import pandas as pd
from bgp_fate.features import build_stage1_dataset
from bgp_fate.models import fit_variant, feature_sets, block_bootstrap_gain, TARGET, FitResult
from bgp_fate.evaluation import _group_gain
from bgp_fate.safeguards import CANDIDATE_ROOT
from bgp_fate.download import fetch_sources
from bgp_fate.audit import run_stage0
ROOT = CANDIDATE_ROOT
from bgp_fate.stage_gate import require_stage0_pass
from bgp_fate.witness import run_witness_analysis

M0 = 'current_state_plus_latest_event'
FULL = 'current_state_plus_full_raw_ordered_history'
SUMMARY = 'current_state_plus_interpretable_fate_state'
SHUFFLE = 'current_state_plus_shuffled_history'
CONTENT = 'front_content_ordered'
CONTENT_SHUFFLE = 'front_content_shuffled'
INVENTORY = 'front_content_inventory'
COMPARISONS = {
    'front_vs_current': (M0, FULL),
    'front_vs_tuple_shuffle': (SHUFFLE, FULL),
    'content_order_vs_shuffle': (CONTENT_SHUFFLE, CONTENT),
    'content_order_vs_inventory': (INVENTORY, CONTENT),
    'front_vs_unordered_summary': ('current_state_plus_unordered_summary', FULL),
    'front_vs_queue_removed': ('fate_state_without_outstanding_queue', FULL),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def source_audit():
    write_json(ROOT / 'results/logs/raw_integrity.json', fetch_sources(verify_only=True))


def dataset_audit(frame):
    reference = json.loads((CODE_ROOT / 'config/sample_invariants.json').read_text())
    keys = ['sample_timestamp', 'prefix', 'collector', 'episode_id']
    fixed = keys + ['split', 'target_change_30', 'target_change_120', 'next_change_timestamp']
    fixed += feature_sets()[M0] + [c for c in frame if c.startswith('lag_local_')]
    fixed = list(dict.fromkeys(fixed))
    assert fixed == reference['columns'] and keys == reference['sort_keys']
    canonical = frame[fixed].sort_values(keys).reset_index(drop=True)
    invariant_digest = hashlib.sha256(canonical.to_csv(
        index=False, float_format='%.17g', lineterminator='\n').encode()).hexdigest()
    assert len(frame) == reference['samples']
    assert invariant_digest == reference['canonical_csv_sha256'], 'Fixed sample invariants changed'
    assert (frame.max_feature_timestamp <= frame.sample_timestamp).all()
    assert frame.groupby('episode_id')['split'].nunique().max() == 1
    for i in range(1, 6):
        active = frame[f'raw_{i}_type'] != 0
        assert frame.loc[active, f'raw_{i}_age'].between(0, 300).all()
        assert (frame.loc[~active, f'raw_{i}_age'] == 3600).all()
    active_count = sum(frame[f'raw_{i}_type'].ne(0).astype(int) for i in range(1, 6))
    assert active_count.equals(frame.hist_outstanding_count.clip(upper=5))
    write_json(ROOT / 'results/logs/dataset_audit.json', {
        'status': 'PASS', 'samples': len(frame), 'unchanged_columns': fixed,
        'sample_invariant_digest': invariant_digest,
        'dataset_sha256': sha(ROOT / 'data/processed/stage1_dataset.parquet'),
        'nonempty_front_samples': int(active_count.gt(0).sum()),
        'max_front_count': int(frame.hist_outstanding_count.max()),
        'five_slot_counts': active_count.value_counts().sort_index().to_dict(),
    })


def canonicalize_samples(frame):
    """Keep fit/permutation row order independent of Python set iteration."""
    return frame.sort_values(['prefix', 'sample_timestamp', 'collector'], kind='stable').reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--analysis-only', action='store_true', help='Reuse an already rebuilt dataset')
    args = parser.parse_args()
    started = time.perf_counter()
    source_audit()
    if not args.analysis_only:
        print(json.dumps(run_stage0()), flush=True)
    require_stage0_pass()
    if not args.analysis_only:
        print(json.dumps(build_stage1_dataset()), flush=True)
    dataset_path = ROOT / 'data/processed/stage1_dataset.parquet'
    original = pd.read_parquet(dataset_path)
    frame = canonicalize_samples(original)
    if not frame.equals(original):
        frame.to_parquet(dataset_path, index=False)
    dataset_audit(frame)
    tables = ROOT / 'results/tables'
    fits = ROOT / 'results/logs/fits'
    prediction_root = ROOT / 'data/processed/fits'
    fits.mkdir(parents=True, exist_ok=True)
    prediction_root.mkdir(parents=True, exist_ok=True)
    dependencies = {name: importlib.metadata.version(name) for name in
                    ['numpy', 'pandas', 'scikit-learn', 'pyarrow', 'mrtparse']}
    scientific_inputs = {
        'dataset': sha(ROOT / 'data/processed/stage1_dataset.parquet'),
        'models': sha(CODE_ROOT / 'src/bgp_fate/models.py'),
        'features': sha(CODE_ROOT / 'src/bgp_fate/features.py'),
        'dependencies': dependencies,
    }
    signature = hashlib.sha256(json.dumps(scientific_inputs, sort_keys=True).encode()).hexdigest()
    cache = {}

    def fit(family, variant, seed=1729, target=TARGET):
        key = f'{family}__{variant}__{seed}__{target}'
        if key in cache:
            return cache[key]
        meta_path = fits / (key + '.json')
        pred_path = prediction_root / (key + '.parquet')
        if meta_path.exists() and pred_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta['signature'] != signature:
                raise RuntimeError('Scientific inputs changed; do not mix cached fits: ' + key)
            result = FitResult(meta['summary'], pd.read_parquet(pred_path))
        else:
            print('FIT ' + key, flush=True)
            tick = time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                result = fit_variant(frame, variant, family, seed, target=target)
            result.predictions.to_parquet(pred_path, index=False)
            write_json(meta_path, dict(signature=signature, summary=result.summary,
                                      elapsed_seconds=time.perf_counter()-tick,
                                      warnings=[str(w.message) for w in caught]))
            print(f"  log loss {result.summary['test_log_loss']:.6f}", flush=True)
        cache[key] = result
        return result

    summaries, ci_rows, group_rows, seed_rows, horizon_rows = [], [], [], [], []
    primary_predictions = []
    for family in ('logistic', 'hist_gradient_boosting'):
        for variant in feature_sets():
            result = fit(family, variant)
            baseline = fit(family, M0).summary['test_log_loss']
            summaries.append({**result.summary, 'gain_vs_strong_m0':
                              1-result.summary['test_log_loss']/baseline})
            if variant in (M0, SUMMARY, FULL):
                primary_predictions.append(result.predictions)
        for name, (reference, candidate) in COMPARISONS.items():
            stats = block_bootstrap_gain(fit(family, reference).predictions,
                                         fit(family, candidate).predictions, 1729)
            ci_rows.append(dict(family=family, comparison=name, reference=reference,
                                candidate=candidate, **stats))
        for group in ('prefix', 'collector', 'time_block'):
            grouped = _group_gain(fit(family, M0).predictions, fit(family, FULL).predictions, group)
            grouped['family'] = family
            grouped['group_type'] = group
            group_rows.append(grouped.rename(columns={group: 'group_value'}))
        for seed in (1729, 2718, 31415):
            for name in ('front_vs_current', 'front_vs_tuple_shuffle',
                         'content_order_vs_shuffle', 'content_order_vs_inventory'):
                reference, candidate = COMPARISONS[name]
                base, alt = fit(family, reference, seed), fit(family, candidate, seed)
                stats = block_bootstrap_gain(base.predictions, alt.predictions, seed, repetitions=500)
                seed_rows.append(dict(seed=seed, family=family, comparison=name,
                                      reference_log_loss=base.summary['test_log_loss'],
                                      candidate_log_loss=alt.summary['test_log_loss'], **stats))
        for target, horizon in ((TARGET, 30), ('target_change_120', 120)):
            for variant in (FULL, SUMMARY):
                base, alt = fit(family, M0, target=target), fit(family, variant, target=target)
                stats = block_bootstrap_gain(base.predictions, alt.predictions, 1729,
                                            repetitions=500, target=target)
                horizon_rows.append(dict(family=family, horizon_seconds=horizon, variant=variant,
                                         m0_log_loss=base.summary['test_log_loss'],
                                         m1_log_loss=alt.summary['test_log_loss'], **stats))
    pd.DataFrame([{k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                   for k, v in row.items()} for row in summaries]).to_csv(tables/'ablation_results.csv', index=False)
    pd.DataFrame(ci_rows).to_csv(tables/'gain_ci.csv', index=False)
    pd.concat(group_rows, ignore_index=True).to_csv(tables/'front_group_results.csv', index=False)
    pd.DataFrame(seed_rows).to_csv(tables/'seed_robustness.csv', index=False)
    pd.DataFrame(horizon_rows).to_csv(tables/'per_horizon_results.csv', index=False)
    pd.concat(primary_predictions, ignore_index=True).to_parquet(
        ROOT/'data/processed/primary_predictions.parquet', index=False)
    witnesses = run_witness_analysis()
    write_json(ROOT/'results/logs/run_manifest.json', {
        'status': 'PASS', 'protocol': 'unmatched-front-within-sample-controls',
        'scientific_inputs': scientific_inputs, 'signature': signature,
        'python': sys.version, 'elapsed_seconds': time.perf_counter()-started,
        'comparisons': ci_rows, 'witnesses': witnesses,
        'interpretation': 'Tuple shuffling preserves ages and tests encoding. '
                          'Age-free content comparisons test chronological content arrangement.'})
    print(pd.DataFrame(ci_rows)[['family', 'comparison', 'gain', 'ci_low', 'ci_high']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
