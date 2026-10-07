# Exact finite-transport Transformer

For layer `l`, the implementation maintains realized state `H_l` and latent
contextual state `xi_l`. It computes:

```text
h_l       = F_l(H_l + xi_l) - L_l(H_l)
H_{l+1}   = L_l(H_l) + psi_l
xi_{l+1}  = h_l - psi_l
```

`F_l` is the full pretrained layer; `L_l` retains token-local computation.
The token-level gate selects `psi_l` from `h_l`. Language-model logits use the
final realized state only; there is no forced terminal release. Pretrained
weights remain fixed.

## Install

The reference environment used Python 3.9, CPU, float32, PyTorch 2.8.0 and
Transformers 4.57.6. Use a separate environment from BGP:

```bash
python3.9 -m venv .venv-transformer
.venv-transformer/bin/python -m pip install -r transformer/requirements.txt
```

The layer wrapper uses the pinned Transformers API. Unpinned newer versions
may require adaptation. A CPU is sufficient for the fixed 135M-parameter
experiment. Model weights and evaluation caches require additional disk space
outside the repository.

## Obtain the model and dataset

Download directly from:

- [HuggingFaceTB/SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext), configuration `wikitext-2-raw-v1`, revision `b08601e04326c79dfdd32d625aee71d232d685c3`.

The dataset loader uses the provider's `wikitext` identifier, which resolves to
this dataset. The configuration pins the source revision as well as the model
revision. Validation and test splits are used; there is no training or
fine-tuning. The dataset package can cache the other split as part of its
normal download process.

From the repository root:

```bash
export HF_HOME="$HOME/.cache/huggingface"
.venv-transformer/bin/python transformer/scripts/download_sources.py
```

The script downloads model configuration, tokenizer files and Safetensors
weights using `huggingface_hub`, then loads the dataset using `datasets`.
Both libraries manage an external cache. The conceptual layout is:

```text
$HF_HOME/
  hub/
    models--HuggingFaceTB--SmolLM2-135M/
      snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2/
        config.json
        tokenizer.json
        ...
        model.safetensors
    datasets--wikitext/  # library alias; may use the Salesforce namespace
      snapshots/b08601e04326c79dfdd32d625aee71d232d685c3/
        ...
  datasets/             # library-managed Arrow cache
```

Keep model, tokenizer and dataset files in their library-managed format.
Do not concatenate raw dataset files by hand. `HF_DATASETS_CACHE` can override
the prepared dataset cache separately. No data, weights or tokenizer assets
are included in the source repository. Provider metadata requests may still
occur when a run starts with populated caches.

## Run

```bash
# Small end-to-end execution check (not the full result)
.venv-transformer/bin/python transformer/scripts/run_all.py --mode smoke

# Full fixed evaluation
.venv-transformer/bin/python transformer/scripts/run_all.py --mode full
```

Use `--config PATH` for a separate configuration and `--output-dir PATH` for
a separate run directory. The default is `transformer/outputs/`; use separate
output directories when retaining both smoke and full runs.

The full run removes blank text rows, joins remaining rows using two newlines,
tokenizes without special tokens and takes contiguous 128-token blocks from
the beginning of each split. It uses 33 validation blocks and 65 test blocks:
4,191 validation and 8,255 test next-token predictions. Token 0 is excluded
from the gate denominator.

Thresholds are proposed and selected on validation data. The primary rule
chooses the lowest non-trivial realization ratio with validation perplexity
at most 1.06 times the fully realized reference. The predefined fallback uses
1.10; the code explicitly reports whether either rule was met. Test evaluation
uses the selected threshold without reselection. The default seed is 20260618.

The runner writes numeric metrics, validation sweeps, reference trajectories,
closure checks, identity checks and CSV summaries under `outputs/results/`.
It records software/configuration information and generated sample metadata.
It generates no manuscript files or manuscript figures.

## Tests

```bash
.venv-transformer/bin/python -m pytest transformer/tests -c transformer/pytest.ini -q
```

Tests use synthetic tensors and text to verify residual closure, selective
realization, pinned dataset loading and contiguous sample selection. They do
not download or load pretrained weights.
