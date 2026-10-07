from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("experiment", ROOT / "src" / "experiment.py")
experiment = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["experiment"] = experiment
SPEC.loader.exec_module(experiment)


def test_token_gate_keeps_exact_residual_identity() -> None:
    candidate = torch.tensor([[[0.0, 0.0], [3.0, 4.0], [0.1, 0.0]]])
    local = torch.ones_like(candidate)
    valid = torch.tensor([False, True, True])
    release, latent_next, active, scores = experiment.apply_token_gate(candidate, local, valid, threshold=1.0)

    assert active.tolist() == [False, True, False]
    assert torch.allclose(release + latent_next, candidate * valid.view(1, -1, 1))
    assert scores[0].item() == -1.0


def test_exact_transport_closure_for_toy_nonlinear_map() -> None:
    hidden = torch.tensor([[[0.2, -0.1], [0.5, 0.7], [-0.3, 0.4]]])
    latent = torch.tensor([[[0.0, 0.0], [0.1, -0.2], [0.3, 0.1]]])
    valid = torch.tensor([False, True, True])

    def full(x: torch.Tensor) -> torch.Tensor:
        context = torch.cumsum(x, dim=1) / torch.arange(1, x.shape[1] + 1, dtype=x.dtype).view(1, -1, 1)
        return x + torch.tanh(context)

    def local(x: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(x)

    local_next = local(hidden)
    h = full(hidden + latent) - local_next
    release, latent_next, _, _ = experiment.apply_token_gate(h, local_next, valid, threshold=0.5)
    realized_next = local_next + release

    assert torch.allclose(realized_next + latent_next, full(hidden + latent), atol=1e-7)
    assert torch.allclose(latent_next, full(hidden + latent) - realized_next, atol=1e-7)


def test_fixed_dataset_revision_and_contiguous_sample_selection(tmp_path, monkeypatch):
    import io
    calls=[]
    def fake_load(name, configuration, **kwargs):
        calls.append((name,configuration,kwargs))
        return {split:[{'text':'synthetic input'}] for split in ['validation','test']}
    monkeypatch.setattr(experiment,'load_dataset',fake_load)
    def tokenizer(text, **kwargs):
        return {'input_ids':list(range(16))}
    cfg={'dataset_name':'synthetic','dataset_config':'example','dataset_revision':'fixed',
         'sequence_length':4,'smoke_validation_blocks':2,'smoke_test_blocks':1}
    samples=experiment.prepare_samples(cfg,tokenizer,tmp_path,'smoke',io.StringIO())
    assert calls == [('synthetic','example',{'revision':'fixed'})]
    assert samples['validation'][1]['input_ids'].tolist() == [4,5,6,7]
    assert samples['test'][0]['prediction_tokens'] == 3
    assert samples['test'][0]['gate_valid_mask'].tolist() == [False,True,True,True]
