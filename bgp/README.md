# BGP unmatched-front experiment

The implementation reconstructs current local routes and a remote unmatched
update queue at two routing observers. At each timestamp, remote observations
are added, the oldest matching occurrence is consumed once per local event,
and entries older than 300 seconds expire. The five newest eligible queue
entries supply the ordered slots; summaries describe the full eligible queue.

The controls include within-sample tuple permutations, age-free chronological
content, age-free shuffled content, and a canonical content inventory. Full
age-preserving permutations test encoding sensitivity; they do not erase
chronology. Positive and negative results are both retained by the runner.

## Install

From the repository root, using Python 3.12:

```bash
python3.12 -m venv .venv-bgp
.venv-bgp/bin/python -m pip install -r bgp/requirements.txt
```

Allow approximately 4 GB of RAM for the replay and fitting steps; the configured
hard process limit is 10 GB. Raw compressed sources total approximately 149 MB.
On the reference CPU, the raw replay takes minutes rather than seconds. The
full run includes data audits, 54 fitted model/target/seed combinations and
bootstrap analysis.

## Download and arrange the source data

The source window is `[2025-07-06 00:00, 2025-07-07 00:00)` UTC:

- RouteViews Chile: [July 2025 archive](https://archive.routeviews.org/route-views.chile/bgpdata/2025.07/), one RIB and 96 updates at 15-minute spacing.
- RIPE RIS `rrc06`: [July 2025 archive](https://data.ris.ripe.net/rrc06/2025.07/), one starting BVIEW and 288 updates at 5-minute spacing.

`config/sources.csv` lists the exact URL, provider, collector, archive kind,
filename, byte length and SHA-256 for every required file. The `stage` column
identifies the first two-hour integrity-check subset. This file is download
metadata, not a dataset of routing observations.

Set an external destination and download:

```bash
export BGP_RAW_DIR="$HOME/.cache/causal-fate/bgp/raw"
.venv-bgp/bin/python bgp/scripts/download_sources.py
.venv-bgp/bin/python bgp/scripts/download_sources.py --verify-only
```

Missing files are fetched from the recorded URLs. Existing files are reused
only when their lengths and checksums match. A mismatch fails explicitly;
the script does not silently replace the source definition.

The resulting layout must be:

```text
$BGP_RAW_DIR/
  route-views.chile/
    rib.20250706.0000.bz2
    updates.20250706.0000.bz2
    updates.20250706.0015.bz2
    ...
    updates.20250706.2345.bz2
  rrc06/
    bview.20250706.0000.gz
    updates.20250706.0000.gz
    updates.20250706.0005.gz
    ...
    updates.20250706.2355.gz
```

Keep the original compressed MRT files and names. Do not extract them to CSV
or merge them. The parser streams the `.bz2` and `.gz` archives directly. Manual
downloads into the same layout are supported; run `--verify-only` afterwards.
When `BGP_RAW_DIR` is unset, the default is `~/.cache/causal-fate/bgp/raw`.

## Run

```bash
.venv-bgp/bin/python bgp/scripts/run_all.py
```

The runner checks all source hashes, reruns the two-hour identity and integrity
gate, then reconstructs the full-day dataset. A failed gate prevents fitting.
No archived PASS log or prior processed dataset is bundled or required.

The fixed protocol uses a chronological 60/20/20 split and assigns whole
convergence episodes to one split. The sample-population check expects 80,267
rows: 56,941 training, 9,696 validation and 13,630 test rows. A fixed digest in
`config/sample_invariants.json` checks the sample identities, labels, splits
and current/local-lag features without distributing those observations.

Each model's preprocessing and hyperparameters are selected using validation
log loss. The primary outcome is a local route change in the next 30 seconds;
120 seconds is secondary. Seeds are 1729, 2718 and 31415. Primary paired
intervals use 1,000 resamples of consecutive 30-minute held-out blocks;
seed/horizon diagnostics use 500. The code reports all comparisons.

Outputs are written below `bgp/outputs/`:

```text
outputs/
  data/processed/       # rebuilt samples and per-model predictions
  results/logs/         # source checks, gate, fit records and run manifest
  results/tables/       # comparisons, ablations, groups, seeds, horizons and cases
```

To put outputs elsewhere, set `BGP_OUTPUT_DIR` before invoking any script.
To reuse an existing rebuilt dataset and gate, use `run_all.py --analysis-only`.
Cached fits require matching dataset, feature/model source and dependency
signatures. If these scientific inputs change, use a fresh output directory.

## Tests

```bash
.venv-bgp/bin/python -m pytest bgp/tests -c bgp/pytest.ini -q
```

Tests cover parsing, queue consumption and expiry, simultaneous matching,
within-row content conservation, padding, age-free comparisons, canonical
inventory invariance, source-manifest completeness, checksum rejection and
block-bootstrap parity against direct row resampling. All test observations
are synthetic.
