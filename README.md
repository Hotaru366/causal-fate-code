# Causal-fate dynamics: C. elegans, BGP and Transformer code

This repository contains three standalone implementations, ordered by their evidential role:

- [`c_elegans/`](c_elegans/README.md): a connectome-constrained neural model motivating a scientific hypothesis about unresolved inter-neuronal influence.
- [`bgp/`](bgp/README.md): an unmatched routing-update front, within-sample order controls, chronological prediction, ablations and uncertainty estimates.
- [`transformer/`](transformer/README.md): exact finite transport of latent contextual influence with selective token-level realization in a pretrained Transformer.

The repository contains implementation source, configuration, synthetic tests and
reproduction instructions. It includes no manuscript, manuscript source, figures,
experimental outputs, raw observations, pretrained model weights or third-party
source-code copies. The BGP source manifest contains only upstream URLs, file
names, lengths and checksums; it contains no routing records.

## Setup and reproduction

Use a separate environment for each experiment. The BGP reference environment
uses Python 3.12; the C. elegans and Transformer reference environments use
Python 3.9. Their dependency versions differ, so use the component-specific
requirements files in separate environments. All three experiments run on CPU. See the component READMEs for configuration and
resource details.

```bash
# C. elegans
python3.9 -m venv .venv-celegans
.venv-celegans/bin/python -m pip install -r c_elegans/requirements.txt
.venv-celegans/bin/python c_elegans/scripts/download_sources.py
.venv-celegans/bin/python c_elegans/scripts/prepare_data.py
.venv-celegans/bin/python c_elegans/scripts/run_all.py

# BGP
python3.12 -m venv .venv-bgp
.venv-bgp/bin/python -m pip install -r bgp/requirements.txt
export BGP_RAW_DIR="$HOME/.cache/causal-fate/bgp/raw"
.venv-bgp/bin/python bgp/scripts/download_sources.py
.venv-bgp/bin/python bgp/scripts/run_all.py

# Transformer
python3.9 -m venv .venv-transformer
.venv-transformer/bin/python -m pip install -r transformer/requirements.txt
export HF_HOME="$HOME/.cache/huggingface"
.venv-transformer/bin/python transformer/scripts/download_sources.py
.venv-transformer/bin/python transformer/scripts/run_all.py --mode full
```

Downloads go to external caches by default. Each experiment writes newly
computed outputs under its own ignored `outputs/` directory. Nothing in any
runner reads or writes a manuscript directory. No existing output or prior
research checkout is required.

## Obtain inputs from their providers

| Input | Provider | Exact selection |
| --- | --- | --- |
| Neural propagation, connectivity and sign predictions | [Worm Neuro Atlas](https://github.com/francescorandi/wormneuroatlas) | Revision `b2e13d88b670efcb3438aeacba2ad4bd6c383933`; source preparation generates the fixed 188-neuron operating object locally |
| Routing MRT archives | [RouteViews Chile](https://archive.routeviews.org/route-views.chile/bgpdata/2025.07/) and [RIPE RIS rrc06](https://data.ris.ripe.net/rrc06/2025.07/) | 6 July 2025, starting RIB plus 24 hours of updates; 386 archives listed in `bgp/config/sources.csv` |
| Pretrained Transformer | [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) | Revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2` |
| Language-model evaluation text | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) | `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`; validation and test splits |

The download scripts obtain these inputs directly from their original providers.
Their respective terms apply. This code repository does not redistribute them.
Detailed download layouts and formats appear in each component README.

## Tests without data downloads

```bash
.venv-celegans/bin/python -m pytest c_elegans/tests -c c_elegans/pytest.ini -q
.venv-bgp/bin/python -m pytest bgp/tests -c bgp/pytest.ini -q
.venv-transformer/bin/python -m pytest transformer/tests -c transformer/pytest.ini -q
```

Tests use synthetic observations and tensors. Running them does not fetch a
model or a dataset. Full experiment runs are separate from the unit tests.

## Scope and release status

This is a local preparation repository. No remote or publication workflow is
configured. The code license for a future public release has not yet been
selected. This preparation does not grant rights to redistribute third-party
inputs.

The C. elegans study motivates a biological hypothesis within an adopted model;
it does not establish a mechanism in living animals. The BGP study evaluates predictive information under fixed representations; it
does not establish a causal link between collectors. The Transformer study
implements a specified construction; it does not claim reduced attention cost
or deployment acceleration. Realization-policy selection precedes evaluation in both constructions. In the
neural study, substrate calibration and readout fitting use the full empirical
object; its stimulus split evaluates the realization policy, not independent
neural-model generalization.
