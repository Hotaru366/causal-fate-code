from collections import Counter, deque

import numpy as np
import pandas as pd
import pytest

from bgp_fate import features, models


def event(timestamp, kind='A', path='1 2', collector='a'):
    return dict(timestamp=timestamp, event_type=kind, as_path=path,
                fingerprint=(kind, path if kind == 'A' else ''), collector=collector)


def test_consumed_and_expired_events_leave_the_ordered_view():
    old, matched, retained = event(0), event(200, path='2 3'), event(300, 'W', '')
    history = deque([old, matched, retained])
    front = deque(history)
    features._expire(front, 301, 300)
    features._remove_match(front, matched['fingerprint'])
    row = features._history_state(front, history, {}, {}, 301)
    assert row['hist_outstanding_count'] == 1
    assert row['raw_1_type'] == -1
    assert row['raw_2_type'] == 0
    assert row['raw_1_age'] == 1


def test_expiry_boundary_and_newest_five_view():
    front = deque(event(t, path=str(t)) for t in [0, 1, 50, 100, 200, 250, 300])
    assert features._expire(front, 300, 300) == 0
    assert features._expire(front, 301, 300) == 1
    row = features._history_state(front, deque(), {}, {}, 301)
    assert [row[f'raw_{i}_age'] for i in range(1, 6)] == [1, 51, 101, 201, 251]


def test_simultaneous_matching_is_symmetric_and_consumes_one_occurrence():
    front = {'a': deque(), 'b': deque()}
    group = [event(100, collector='a'), event(100, collector='b')]
    features._advance_fronts(front, group, ['a', 'b'], 100, 300)
    assert not front['a'] and not front['b']
    front['a'].extend([event(100), event(101)])
    features._advance_fronts(front, [event(102, collector='a')], ['a', 'b'], 102, 300)
    assert [item['timestamp'] for item in front['a']] == [101]
    assert len(front['b']) == 1


def frame():
    rows = []
    for n in range(12):
        row = dict(split='test', prefix='p', collector='a', sample_timestamp=n,
                   target_change_30=n % 2, current_path_bucket=n + 900)
        for field in models.ORDERED_STATE:
            row[field] = n
        for i in range(1, 6):
            row[f'raw_{i}_type'] = (1 if i % 2 else -1) if i <= 3 else 0
            row[f'raw_{i}_age'] = i * 10 + n if i <= 3 else 3600
            row[f'raw_{i}_path_bucket'] = n * 10 + i if i <= 3 else 0
        rows.append(row)
    return pd.DataFrame(rows, index=np.arange(12) * 2)


def tuples(row):
    return Counter(tuple(row[f'raw_{i}_{s}'] for s in ('type', 'age', 'path_bucket'))
                   for i in range(1, 6) if row[f'raw_{i}_type'] != 0)


def test_shuffle_preserves_each_rows_content_and_padding():
    original = frame()
    shuffled = models._shuffle_order_state(original, 1729)
    assert shuffled.index.equals(original.index)
    for index in original.index:
        assert tuples(shuffled.loc[index]) == tuples(original.loc[index])
        for i in [4, 5]:
            for suffix in ('type', 'age', 'path_bucket'):
                assert shuffled.loc[index, f'raw_{i}_{suffix}'] == original.loc[index, f'raw_{i}_{suffix}']
    fixed = [c for c in original if not c.startswith('raw_')]
    pd.testing.assert_frame_equal(original[fixed], shuffled[fixed])
    pd.testing.assert_frame_equal(shuffled, models._shuffle_order_state(original, 1729))
    assert not original[models.RAW_ORDERED].equals(shuffled[models.RAW_ORDERED])


def test_age_free_order_pair_uses_identical_information_fields():
    sets = models.feature_sets()
    ordered = sets['front_content_ordered']
    assert ordered == sets['front_content_shuffled'] == sets['front_content_inventory']
    assert not any(c.startswith('raw_') and c.endswith('_age') for c in ordered)
    assert not any(c in models.ORDERED_STATE for c in ordered)


def test_empty_and_singleton_fronts_are_unchanged():
    original = frame().iloc[:2].copy()
    for j, index in enumerate(original.index):
        for i in range(j + 1, 6):
            original.loc[index, [f'raw_{i}_type', f'raw_{i}_age', f'raw_{i}_path_bucket']] = [0, 3600, 0]
    pd.testing.assert_frame_equal(original, models._shuffle_order_state(original, 1729))


def test_inventory_is_identical_for_reordered_content():
    original = frame()
    shuffled = models._shuffle_order_state(original, 2718)
    pd.testing.assert_frame_equal(
        models._inventory_state(original)[models.CONTENT_ONLY],
        models._inventory_state(shuffled)[models.CONTENT_ONLY],
    )
