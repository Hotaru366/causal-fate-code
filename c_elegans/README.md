# C. elegans: model-based biological hypothesis

This experiment constructs finite transport and selective realization in a
connectome-constrained, 188-neuron head-network model. It asks whether selected
propagation structure can be retained when inter-neuronal influence is realized
selectively. The result motivates a scientific hypothesis about unresolved
influence in living neural systems; it does not identify or validate a physical
carrier in an animal.

Current scientific version: V4, exact nonlinear finite-displacement transport.
The current standalone packaging preserves its numerical kernels, fixed
substrate parameters, convergence protocol and realization selection rules.

## Version history

| Version | Status | Scope |
| --- | --- | --- |
| V4 standalone preparation, 2026-10-08 | Current | External provider cache, source reconstruction and independent execution; no bundled data or fitted coefficients |
| V4 research implementation | Numerical source | Same scientific model and results; its original precomputed input bundle is replaced here by preparation code |

## Environment and commands

Reference environment: Python 3.9 on CPU, with Git available. Use a separate
environment from BGP and Transformer.

```bash
# From the repository root
python3.9 -m venv .venv-celegans
.venv-celegans/bin/python -m pip install -r c_elegans/requirements.txt
export CELEGANS_RAW_DIR="$HOME/.cache/causal-fate/celegans/raw"
export CELEGANS_DATA_DIR="$HOME/.cache/causal-fate/celegans/prepared"
.venv-celegans/bin/python c_elegans/scripts/download_sources.py
.venv-celegans/bin/python c_elegans/scripts/prepare_data.py
.venv-celegans/bin/python c_elegans/scripts/run_all.py
```

`prepare_data.py` also fetches the source if necessary. It evaluates the
published kernels, applies the fixed observation-window rule and transform,
constructs masks and signed connectivity, then regenerates the response-local
readout. It uses the fixed selected substrate parameters in `config/default.json`;
it does not perform new parameter selection. The historical grid-calibration
routine is retained in `src/celegans_dtpr_v4/dynamics.py` for inspection.

The readout is reconstructed with the historical substrate calibration map
(explicit Euler, 0.1 s). The final V4 propagation/transport evaluation uses RK4
and its convergence-selected integration step; the readout is not refitted at
that step. This preserves the scientific definition of the original experiment.

`run_all.py` performs the integration-convergence check, realization-policy
selection and stimulus-stratified evaluation. It regenerates the seed-188 split
(118 selection stimuli and 50 evaluation stimuli). The full propagation object
was used to calibrate the substrate/readout before that split. Therefore the
50-stimulus evaluation is not an independent validation of the neural model.
The full run takes several minutes in the reference environment and retains
trajectories, so allow several GB of free memory and disk space.

## Obtain data directly from the provider

Use the official [Worm Neuro Atlas repository](https://github.com/francescorandi/wormneuroatlas)
at commit `b2e13d88b670efcb3438aeacba2ad4bd6c383933`.
The download script checks out this exact revision in the external cache.
The signal-propagation study is Randi et al. (Nature, 2023); its original
resources are the [online atlas](https://funconn.princeton.edu) and
[OSF deposit](https://doi.org/10.17605/OSF.IO/E2SYT). For this implementation use
the pinned integrated repository, which also supplies the default White/Witvliet
anatomical connectivity and Fenyves chemical-sign predictions.

Expected provider layout:

```text
$CELEGANS_RAW_DIR/wormneuroatlas/
  .git/
  wormneuroatlas/data/
    funatlas.h5
    aconnectome_default.h5
    aconnectome_ids_ganglia.json
    neuron_ids.txt
    journal.pcbi.1007974.s003.xlsx
```

The HDF5 files contain pairwise functional kernels/measurements and anatomical
matrices. The workbook supplies sign predictions. They must remain in the
provider checkout; they are not included in this repository. Git and these
files are sufficient; installing or copying the upstream Python package into
this code repository is unnecessary.

After preparation:

```text
$CELEGANS_DATA_DIR/
  operating_data/             # generated matrices, masks and neuron identifiers
  frozen_config.json          # generated readout coefficients plus configuration
  provenance_manifest.json   # source revision and reconstructed array hashes
  kernel_window_audit.json
```

Preparation yields 188 neurons, 23,264 observed off-diagonal pairs, 23,180 valid
observed kernels and 1,141 high-confidence pairs, with a 10-s window.
`config/expected_input_hashes.json` contains checksums only, permitting comparison
with the fixed research inputs without redistributing their contents. Raw data,
processed arrays, stimulus records and fitted readout coefficients are all
created locally, outside this repository by default.

## Outputs and interpretation

By default, results go to ignored `c_elegans/outputs/`; override with
`CELEGANS_OUTPUT_DIR` if desired. The runner never reads or writes a manuscript.
Its key artifacts are `results/metrics/held_out_test_metrics.json`,
`results/metrics/final_configs.json`, `results/scans/quality_realization_frontier.csv`
and `results/scans/integration_convergence.csv`. Keep generated outputs out of Git.

The reference evaluation has relative selective-versus-full propagation error
about 0.0351499, correlation above 0.99 and sign agreement 1.0. The local-autonomy
counterfactual has relative error 1.0. Fully realized and local-autonomy dynamics
are constitutive counterfactual references, not generic algorithmic baselines.

## Tests without downloads

```bash
.venv-celegans/bin/python -m pytest c_elegans/tests -c c_elegans/pytest.ini -q
```

The tests use a synthetic two-neuron network to check exact represented-state
closure, full/no-realization limits, the nonnegative local readout and signed
kernel integration. They require no empirical files.
