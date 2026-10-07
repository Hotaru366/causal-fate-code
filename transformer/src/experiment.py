from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import model_info
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MASK_NEGATIVE = -1.0e4
EPS = 1.0e-8


@dataclass
class ReferenceSampleResult:
    sample_id: str
    split: str
    block_index: int
    n_tokens: int
    prediction_tokens: int
    continuous_nll: float
    local_nll: float
    continuous_count: int
    local_count: int
    top1_agreement_local_vs_continuous: float


@dataclass
class SelectiveSampleResult:
    sample_id: str
    split: str
    block_index: int
    n_tokens: int
    prediction_tokens: int
    threshold: float
    selective_nll: float
    selective_count: int
    token_layer_realization_ratio: float
    inactive_ratio: float
    active_layer_ratio: float
    mean_active_token_fraction_per_layer: float
    mean_active_layers_per_token: float
    mean_local_autonomy_layers_per_token: float
    max_consecutive_latent_depth: int
    terminal_latent_norm: float
    per_layer_active_fraction: List[float]
    per_layer_latent_norm: List[float]
    per_position_active_fraction: List[float]
    per_position_mean_active_layers: List[float]
    latent_depth_distribution: List[int]
    realization_matrix: List[List[int]]
    closure_relative_by_layer: Optional[List[float]]
    closure_absolute_by_layer: Optional[List[float]]
    latent_displacement_relative_by_layer: Optional[List[float]]
    reconstructed_logit_relative_error: Optional[float]
    reconstructed_logit_max_abs_error: Optional[float]
    main_logit_kl_vs_continuous: Optional[float]
    main_top1_agreement_vs_continuous: Optional[float]


def run_all(root: Path, repo_root: Path, config_path: Path, mode: str) -> None:
    t0 = time.time()
    cfg = read_json(config_path)
    ensure_dirs(root)
    set_seed(int(cfg["seed"]))

    log_path = root / "results" / "logs" / f"run_{mode}.log"
    with log_path.open("w", encoding="utf-8") as log:
        log_line(log, f"mode={mode}")
        log_line(log, f"root={root}")
        log_line(log, f"repo_root={repo_root}")
        log_line(log, f"config={config_path}")

        preflight = collect_preflight(repo_root, root)
        write_json(root / "results" / "metrics" / f"preflight_{mode}.json", preflight)
        log_line(log, "code revision recorded")

        env = collect_environment(cfg)
        write_json(root / "environment" / f"environment_{mode}.json", env)
        write_json(root / "results" / "metrics" / f"environment_{mode}.json", env)
        log_line(log, "environment check complete")

        model, tokenizer, model_meta = load_model_and_tokenizer(cfg, root, log)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        samples = prepare_samples(cfg, tokenizer, root, mode, log)
        validation_samples = samples["validation"]
        test_samples = samples["test"]
        log_line(
            log,
            "prepared samples: "
            f"validation_prediction_tokens={sum(s['prediction_tokens'] for s in validation_samples)}, "
            f"test_prediction_tokens={sum(s['prediction_tokens'] for s in test_samples)}",
        )

        log_line(log, "running custom-layer and decomposition checks")
        implementation_checks = run_implementation_checks(model, validation_samples, cfg, device, mode, log)
        write_json(
            root / "results" / "metrics" / f"implementation_checks_{mode}.json",
            implementation_checks,
        )

        log_line(log, "running threshold pilot on validation only")
        pilot = run_threshold_pilot(model, validation_samples, cfg, device, mode)
        write_json(root / "results" / "metrics" / f"threshold_pilot_{mode}.json", pilot)
        thresholds = formal_thresholds_from_pilot(pilot, cfg, mode)
        write_json(
            root / "results" / "metrics" / f"threshold_grid_{mode}.json",
            {"thresholds": thresholds, "source": "validation pilot only", "config": threshold_config(cfg, mode)},
        )
        log_line(log, "formal thresholds=" + ", ".join(f"{t:.6g}" for t in thresholds))

        log_line(log, "evaluating validation reference trajectories")
        validation_reference = evaluate_reference_split(
            model=model,
            samples=validation_samples,
            cfg=cfg,
            device=device,
            split="validation",
            mode=mode,
        )
        write_json(root / "results" / "metrics" / f"validation_reference_{mode}.json", validation_reference)

        log_line(log, "running validation exact-transport threshold sweep")
        validation_sweep = []
        for threshold in tqdm(thresholds, desc="validation thresholds"):
            validation_sweep.append(
                evaluate_selective_split(
                    model=model,
                    samples=validation_samples,
                    cfg=cfg,
                    threshold=float(threshold),
                    device=device,
                    split="validation",
                    run_tag=mode,
                    root=None,
                    audit=False,
                    reference_metrics=validation_reference,
                )
            )
        validation_sweep_payload = common_result_payload(
            cfg=cfg,
            model_meta=model_meta,
            split="validation",
            metrics={
                "rows": [strip_samples(row) for row in validation_sweep],
                "threshold_count": len(validation_sweep),
                "selection_rule": "lowest token-layer realization ratio satisfying <= 1.06x validation Continuous perplexity, fallback <= 1.10x",
            },
            metric_definitions=metric_definitions(),
            sample_manifest=sample_manifest_for_json(validation_samples),
        )
        write_json(root / "results" / "metrics" / "threshold_sweep_validation.json", validation_sweep_payload)
        write_json(root / "results" / "metrics" / f"threshold_sweep_validation_{mode}.json", validation_sweep_payload)
        write_threshold_tradeoff(root, validation_sweep, split="validation")

        selected = select_threshold(validation_sweep, cfg)
        write_json(root / "results" / "metrics" / f"selected_threshold_{mode}.json", selected)
        log_line(log, f"selected threshold={selected['threshold']:.6g} reason={selected['reason']}")

        log_line(log, "evaluating test reference trajectories")
        test_reference = evaluate_reference_split(
            model=model,
            samples=test_samples,
            cfg=cfg,
            device=device,
            split="test",
            mode=mode,
        )
        write_json(root / "results" / "metrics" / "full_reference.json", make_full_reference_payload(test_reference, cfg, model_meta, test_samples))
        write_json(root / "results" / "metrics" / "local_only.json", make_local_only_payload(test_reference, cfg, model_meta, test_samples))
        write_json(root / "results" / "metrics" / f"test_reference_{mode}.json", test_reference)

        log_line(log, "running frozen-threshold main test evaluation")
        main_test = evaluate_selective_split(
            model=model,
            samples=test_samples,
            cfg=cfg,
            threshold=float(selected["threshold"]),
            device=device,
            split="test",
            run_tag=mode,
            root=root,
            audit=True,
            reference_metrics=test_reference,
            representative_index=int(cfg["representative_test_sample_index"]),
        )
        main_result = make_main_test_payload(main_test, test_reference, selected, cfg, model_meta, test_samples)
        write_json(root / "results" / "metrics" / "main_test_result.json", main_result)

        log_line(log, "running fixed test tradeoff thresholds without reselecting")
        test_tradeoff_thresholds = choose_test_tradeoff_thresholds(thresholds, float(selected["threshold"]), cfg)
        test_tradeoff = []
        for threshold in tqdm(test_tradeoff_thresholds, desc="test tradeoff"):
            metrics = (
                main_test
                if abs(float(threshold) - float(selected["threshold"])) < 1e-12
                else evaluate_selective_split(
                    model=model,
                    samples=test_samples,
                    cfg=cfg,
                    threshold=float(threshold),
                    device=device,
                    split="test",
                    run_tag=mode,
                    root=None,
                    audit=False,
                    reference_metrics=test_reference,
                )
            )
            test_tradeoff.append(metrics)
        write_json(root / "results" / "metrics" / f"test_tradeoff_{mode}.json", [strip_samples(row) for row in test_tradeoff])
        write_threshold_tradeoff(root, test_tradeoff, split="test")

        closure_audit = make_exact_closure_audit(main_test, cfg, model_meta, test_samples)
        fate_audit = run_fate_identity_audit(
            model=model,
            samples=validation_samples[: int(cfg["audit_validation_blocks"])],
            cfg=cfg,
            model_meta=model_meta,
            threshold=float(selected["threshold"]),
            device=device,
            mode=mode,
        )
        write_json(root / "results" / "metrics" / "exact_closure_audit.json", closure_audit)
        write_json(root / "results" / "metrics" / "fate_identity_audit.json", fate_audit)

        write_summary_tables(root, main_test, test_reference, validation_sweep, selected)
        manifest = {
            "mode": mode,
            "timestamp_utc": utc_now(),
            "elapsed_seconds": time.time() - t0,
            "selected_threshold": selected,
            "full_reference": str(root / "results" / "metrics" / "full_reference.json"),
            "local_only": str(root / "results" / "metrics" / "local_only.json"),
            "threshold_sweep_validation": str(root / "results" / "metrics" / "threshold_sweep_validation.json"),
            "main_test_result": str(root / "results" / "metrics" / "main_test_result.json"),
            "exact_closure_audit": str(root / "results" / "metrics" / "exact_closure_audit.json"),
            "fate_identity_audit": str(root / "results" / "metrics" / "fate_identity_audit.json"),
        }
        write_json(root / "results" / "metrics" / f"run_manifest_{mode}.json", manifest)
        log_line(log, f"done elapsed_seconds={manifest['elapsed_seconds']:.2f}")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")


def ensure_dirs(root: Path) -> None:
    for rel in [
        "data",
        "environment",
        "results/metrics",
        "results/tables",
        "results/logs",
        "results/checkpoints_or_cache",
        "results/trajectories",
    ]:
        (root / rel).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)


def log_line(log, message: str) -> None:
    print(message)
    log.write(message + "\n")
    log.flush()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_output(cmd: List[str], cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.check_output(cmd, cwd=str(cwd or Path(__file__).resolve().parents[2]), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def collect_preflight(repo_root: Path, root: Path) -> Dict[str, Any]:
    return {"timestamp_utc": utc_now(),
            "git_commit": git_output(["git", "rev-parse", "HEAD"], cwd=repo_root),
            "git_branch": git_output(["git", "branch", "--show-current"], cwd=repo_root)}


def collect_environment(cfg: Dict[str, Any]) -> Dict[str, Any]:
    import datasets
    import huggingface_hub
    import transformers

    cuda_available = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    cuda_version = torch.version.cuda
    return {
        "timestamp_utc": utc_now(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "numpy": np.__version__,
        "cuda_available": cuda_available,
        "cuda_version": cuda_version,
        "gpu_name": gpu_name,
        "device_requested": cfg["device"],
        "dtype_requested": cfg["dtype"],
        "seed": int(cfg["seed"]),
        "hf_home": os.environ.get("HF_HOME"),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_branch": git_output(["git", "branch", "--show-current"]),
    }


def load_model_and_tokenizer(cfg: Dict[str, Any], root: Path, log) -> Tuple[Any, Any, Dict[str, Any]]:
    model_id = cfg["model_id"]
    revision = cfg.get("model_revision")
    log_line(log, f"loading model/tokenizer: {model_id}@{revision}")
    hf_info = model_info(model_id, revision=revision)
    dtype = dtype_from_config(cfg["dtype"])

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tokenizer_revision = model_info(model_id, revision=revision).sha
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=dtype,
        attn_implementation=cfg.get("attn_implementation", "eager"),
    )
    model.eval()
    model.config.use_cache = False
    model.to(torch.device(cfg["device"]))
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    config = model.config
    device = next(model.parameters()).device
    meta = {
        "timestamp_utc": utc_now(),
        "model_id": model_id,
        "revision": hf_info.sha,
        "tokenizer_revision": tokenizer_revision,
        "model_type": getattr(config, "model_type", None),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "num_hidden_layers": int(config.num_hidden_layers),
        "hidden_size": int(config.hidden_size),
        "intermediate_size": int(config.intermediate_size),
        "num_attention_heads": int(config.num_attention_heads),
        "num_key_value_heads": int(getattr(config, "num_key_value_heads", config.num_attention_heads)),
        "vocab_size": int(config.vocab_size),
        "max_position_embeddings": int(config.max_position_embeddings),
        "model_config_dtype": str(getattr(config, "torch_dtype", None)),
        "runtime_dtype": str(dtype),
        "device": str(device),
        "attn_implementation": getattr(config, "_attn_implementation", None),
        "eval_mode": not model.training,
        "use_cache": bool(getattr(model.config, "use_cache", False)),
    }
    write_json(root / "results" / "metrics" / "model_metadata.json", meta)
    return model, tokenizer, meta


def dtype_from_config(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def prepare_samples(cfg: Dict[str, Any], tokenizer: Any, root: Path, mode: str, log) -> Dict[str, List[Dict[str, Any]]]:
    log_line(log, "loading WikiText-2 validation/test splits")
    dataset = load_dataset(cfg["dataset_name"], cfg["dataset_config"], revision=cfg.get("dataset_revision"))
    seq_len = int(cfg["sequence_length"])
    if mode == "smoke":
        split_blocks = {
            "validation": int(cfg["smoke_validation_blocks"]),
            "test": int(cfg["smoke_test_blocks"]),
        }
    else:
        split_blocks = {
            "validation": int(cfg["validation_blocks"]),
            "test": int(cfg["test_blocks"]),
        }

    output: Dict[str, List[Dict[str, Any]]] = {}
    manifest: List[Dict[str, Any]] = []
    for split, n_blocks in split_blocks.items():
        texts = [row["text"] for row in dataset[split] if row["text"].strip()]
        joined = "\n\n".join(texts)
        ids = tokenizer(joined, add_special_tokens=False, truncation=False, verbose=False)["input_ids"]
        required = n_blocks * seq_len
        if len(ids) < required:
            raise RuntimeError(f"{split} has {len(ids)} tokens but {required} are required")
        samples = []
        for block_index in range(n_blocks):
            start = block_index * seq_len
            block_ids = torch.tensor(ids[start : start + seq_len], dtype=torch.long)
            loss_mask = torch.ones(seq_len, dtype=torch.bool)
            gate_valid_mask = torch.ones(seq_len, dtype=torch.bool)
            gate_valid_mask[0] = False
            sample_id = f"{split}_block_{block_index:04d}"
            sample = {
                "sample_id": sample_id,
                "split": split,
                "block_index": block_index,
                "token_start": start,
                "token_end": start + seq_len,
                "input_ids": block_ids,
                "loss_mask": loss_mask,
                "gate_valid_mask": gate_valid_mask,
                "prediction_tokens": int(loss_mask[1:].sum().item()),
                "valid_gate_tokens": int(gate_valid_mask.sum().item()),
            }
            samples.append(sample)
            manifest.append(
                {
                    "mode": mode,
                    "split": split,
                    "sample_id": sample_id,
                    "block_index": block_index,
                    "token_start": start,
                    "token_end": start + seq_len,
                    "sequence_length": seq_len,
                    "prediction_tokens": sample["prediction_tokens"],
                    "valid_gate_tokens": sample["valid_gate_tokens"],
                    "token0_excluded_from_gate_denominator": True,
                    "padding_tokens": 0,
                    "source": "contiguous WikiText-2 token blocks from split start, using the fixed evaluation protocol",
                }
            )
        output[split] = samples
    write_json(root / "data" / f"sample_manifest_{mode}.json", manifest)
    return output


def initial_states(model: Any, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
    hidden = model.model.embed_tokens(input_ids)
    seq_len = hidden.shape[1]
    device = hidden.device
    cache_position = torch.arange(seq_len, device=device)
    position_ids = cache_position.unsqueeze(0)
    position_embeddings = model.model.rotary_emb(hidden, position_ids)
    return {
        "hidden": hidden,
        "position_ids": position_ids,
        "cache_position": cache_position,
        "position_embeddings": position_embeddings,
        "causal_mask": make_causal_mask(seq_len, device, hidden.dtype),
        "local_mask": make_local_mask(seq_len, device, hidden.dtype),
    }


def make_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)
    future = torch.triu(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool), diagonal=1)
    mask = mask.masked_fill(future, MASK_NEGATIVE)
    return mask.view(1, 1, seq_len, seq_len)


def make_local_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), MASK_NEGATIVE, device=device, dtype=dtype)
    diag = torch.eye(seq_len, device=device, dtype=torch.bool)
    mask = mask.masked_fill(diag, 0.0)
    return mask.view(1, 1, seq_len, seq_len)


def apply_layer(model: Any, layer_index: int, hidden: torch.Tensor, states: Dict[str, torch.Tensor], local_only: bool) -> torch.Tensor:
    layer = model.model.layers[layer_index]
    output = layer(
        hidden,
        attention_mask=states["local_mask"] if local_only else states["causal_mask"],
        position_ids=states["position_ids"],
        cache_position=states["cache_position"],
        position_embeddings=states["position_embeddings"],
        use_cache=False,
    )
    return output[0] if isinstance(output, tuple) else output


def final_logits(model: Any, hidden: torch.Tensor) -> torch.Tensor:
    return model.lm_head(model.model.norm(hidden))


def nll_from_logits(logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor) -> Tuple[float, int]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = loss_mask[:, 1:].contiguous().view(-1)
    per_token = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    )
    masked = per_token[shift_mask]
    return float(masked.sum().item()), int(shift_mask.sum().item())


def ppl(nll: float, count: int) -> float:
    return float(math.exp(nll / max(count, 1)))


def cross_entropy(nll: float, count: int) -> float:
    return float(nll / max(count, 1))


def token_scores(candidate: torch.Tensor, local_current: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    scores = candidate.norm(dim=-1) / local_current.norm(dim=-1).clamp_min(EPS)
    return scores.squeeze(0).masked_fill(~valid_mask, -1.0)


def apply_token_gate(
    candidate: torch.Tensor,
    local_current: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = token_scores(candidate, local_current, valid_mask)
    active = (scores > threshold) & valid_mask
    active_f = active.to(candidate.dtype).view(1, -1, 1)
    valid_f = valid_mask.to(candidate.dtype).view(1, -1, 1)
    candidate = candidate * valid_f
    release = candidate * active_f
    latent_next = candidate - release
    return release, latent_next, active, scores


def valid_fro_norm(tensor: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    return (tensor * valid_mask.to(tensor.dtype).view(1, -1, 1)).norm()


def run_implementation_checks(
    model: Any,
    validation_samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
    mode: str,
    log,
) -> Dict[str, Any]:
    sample = validation_samples[0]
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    loss_mask = sample["loss_mask"].unsqueeze(0).to(device)
    valid_mask = sample["gate_valid_mask"].to(device)
    states = initial_states(model, input_ids)
    hidden0 = states["hidden"]

    with torch.no_grad():
        custom = hidden0
        for layer_index in range(model.config.num_hidden_layers):
            custom = apply_layer(model, layer_index, custom, states, local_only=False)
        custom_logits = final_logits(model, custom)
        hf_output = model(input_ids=input_ids, use_cache=False)
        hf_logits = hf_output.logits
        custom_nll, custom_count = nll_from_logits(custom_logits, input_ids, loss_mask)
        hf_nll, hf_count = nll_from_logits(hf_logits, input_ids, loss_mask)

        full0 = apply_layer(model, 0, hidden0, states, local_only=False)
        local0 = apply_layer(model, 0, hidden0, states, local_only=True)
        cross0 = full0 - local0
        decomposition_max_abs = float((full0 - (local0 + cross0)).abs().max().item())
        same_position_mask_check = bool(torch.all(states["local_mask"][0, 0].diag() == 0.0).item())
        mask0 = states["causal_mask"][0, 0]
        future_positions = torch.triu(torch.ones_like(mask0, dtype=torch.bool), diagonal=1)
        future_mask_check = bool((mask0[future_positions] <= MASK_NEGATIVE).all().item())

        release, latent_next, active, scores = apply_token_gate(
            full0 - local0,
            local0,
            valid_mask,
            threshold=float(torch.median(token_scores(full0 - local0, local0, valid_mask)[valid_mask]).item()),
        )
        mixed_layer = bool(active.any().item() and ((~active) & valid_mask).any().item())
        gate_identity_error = float((release + latent_next - (full0 - local0) * valid_mask.to(full0.dtype).view(1, -1, 1)).abs().max().item())

        tau = float(torch.quantile(scores[valid_mask].float(), 0.5).item())
        closure_sample = evaluate_selective_sample(
            model=model,
            sample=sample,
            cfg=cfg,
            threshold=tau,
            device=device,
            audit=True,
            reference_states=None,
        )

    logits_diff = custom_logits - hf_logits
    custom_hf_relative = float(logits_diff.norm().div(hf_logits.norm().clamp_min(EPS)).item())
    custom_hf_max_abs = float(logits_diff.abs().max().item())
    custom_hf_ce_abs = abs(cross_entropy(custom_nll, custom_count) - cross_entropy(hf_nll, hf_count))
    log_line(log, f"custom-vs-HF logits relative={custom_hf_relative:.3e} max_abs={custom_hf_max_abs:.3e}")

    return {
        "timestamp_utc": utc_now(),
        "sample_id": sample["sample_id"],
        "standard_hf_forward_matches_custom_layers": {
            "logits_relative_error": custom_hf_relative,
            "logits_max_abs_error": custom_hf_max_abs,
            "cross_entropy_absolute_error": custom_hf_ce_abs,
            "custom_cross_entropy": cross_entropy(custom_nll, custom_count),
            "hf_cross_entropy": cross_entropy(hf_nll, hf_count),
            "custom_perplexity": ppl(custom_nll, custom_count),
            "hf_perplexity": ppl(hf_nll, hf_count),
        },
        "full_local_decomposition": {
            "definition": "F_l(H)-L_l(H) with the same decoder layer, position ids, RoPE embeddings, residual order, RMSNorm, MLP, and masks",
            "layer0_max_abs_error": decomposition_max_abs,
            "local_layer_is_same_position_attention_mask": same_position_mask_check,
            "causal_future_mask_applied": future_mask_check,
        },
        "local_mapping": {
            "preserves_residual_stream": True,
            "preserves_rmsnorm": True,
            "preserves_mlp": True,
            "preserves_positionwise_nonlinearity": True,
            "preserves_same_position_attention_path": True,
            "removes_new_cross_token_attention_by_mask": True,
        },
        "token_gate": {
            "score_definition": cfg["score_normalization"],
            "gate_shape": list(active.shape),
            "candidate_shape": list(cross0.shape),
            "same_layer_active_and_inactive_tokens": mixed_layer,
            "binary_realization": True,
            "independent_by_token": True,
            "token0_excluded_from_primary_denominator": True,
            "valid_token_count": int(valid_mask.sum().item()),
            "active_count_in_check_layer": int(active.sum().item()),
            "inactive_count_in_check_layer": int(((~active) & valid_mask).sum().item()),
            "release_plus_latent_equals_candidate_max_abs": gate_identity_error,
            "score_min_valid": float(scores[valid_mask].min().item()),
            "score_max_valid": float(scores[valid_mask].max().item()),
        },
        "exact_transport_check": {
            "update": "h_l = F_l(H_l + xi_l) - L_l(H_l); H_{l+1}=L_l(H_l)+psi_l; xi_{l+1}=h_l-psi_l",
            "terminal_forced_release": False,
            "sample_max_relative_closure_error": max_or_none(closure_sample.closure_relative_by_layer),
            "sample_max_absolute_closure_error": max_or_none(closure_sample.closure_absolute_by_layer),
            "sample_max_latent_displacement_relative_error": max_or_none(
                closure_sample.latent_displacement_relative_by_layer
            ),
            "reconstructed_logit_relative_error": closure_sample.reconstructed_logit_relative_error,
            "reconstructed_logit_max_abs_error": closure_sample.reconstructed_logit_max_abs_error,
        },
        "deterministic_seed": int(cfg["seed"]),
        "mode": mode,
    }


def run_threshold_pilot(
    model: Any,
    validation_samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
    mode: str,
) -> Dict[str, Any]:
    pilot_count = min(len(validation_samples), int(cfg["threshold_pilot_blocks"]) if mode == "full" else 1)
    scores: List[float] = []
    final_latent_norms: List[float] = []
    for sample in tqdm(validation_samples[:pilot_count], desc="threshold pilot", leave=False):
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        valid_mask = sample["gate_valid_mask"].to(device)
        states = initial_states(model, input_ids)
        hidden = states["hidden"]
        latent = torch.zeros_like(hidden)
        with torch.no_grad():
            for layer_index in range(model.config.num_hidden_layers):
                z_state = hidden + latent
                full_z = apply_layer(model, layer_index, z_state, states, local_only=False)
                local_h = apply_layer(model, layer_index, hidden, states, local_only=True)
                candidate = full_z - local_h
                layer_scores = token_scores(candidate, local_h, valid_mask)
                scores.extend(layer_scores[valid_mask].detach().cpu().tolist())
                hidden = full_z
                latent = torch.zeros_like(hidden)
            final_latent_norms.append(float(valid_fro_norm(latent, valid_mask).item()))
    values = np.array(scores, dtype=np.float64)
    quantiles = {}
    if len(values):
        for q in cfg["threshold_quantile_targets"]:
            quantiles[str(q)] = float(np.quantile(values, float(q)))
    return {
        "timestamp_utc": utc_now(),
        "split": "validation",
        "pilot_sample_count": pilot_count,
        "score_count": int(len(scores)),
        "score_definition": cfg["score_normalization"],
        "score_min": float(np.min(values)) if len(values) else None,
        "score_median": float(np.median(values)) if len(values) else None,
        "score_max": float(np.max(values)) if len(values) else None,
        "score_quantiles": quantiles,
        "terminal_latent_norm_mean_for_all_active_probe": float(np.mean(final_latent_norms)) if final_latent_norms else 0.0,
    }


def formal_thresholds_from_pilot(pilot: Dict[str, Any], cfg: Dict[str, Any], mode: str) -> List[float]:
    thresholds = {0.0}
    for value in pilot.get("score_quantiles", {}).values():
        if value is not None and math.isfinite(float(value)):
            thresholds.add(float(value))
    score_min = pilot.get("score_min")
    score_max = pilot.get("score_max")
    score_median = pilot.get("score_median")
    if score_median is not None and score_median > 0:
        lo = max(float(score_median) * float(cfg["threshold_multiplier_min"]), 1e-12)
        hi = max(float(score_median) * float(cfg["threshold_multiplier_max"]), lo * 1.01)
    elif score_max is not None and score_max > 0:
        lo = max(float(score_max) * 1e-3, 1e-12)
        hi = float(score_max) * 2.0
    else:
        lo, hi = 1e-6, 1.0
    for value in np.geomspace(lo, hi, int(cfg["threshold_log_points"])):
        thresholds.add(float(value))
    if score_min is not None:
        thresholds.add(max(0.0, float(score_min) * 0.5))
    if score_max is not None:
        thresholds.add(float(score_max) * 1.25)

    max_count = int(cfg["smoke_max_formal_thresholds"] if mode == "smoke" else cfg["max_formal_thresholds"])
    sorted_thresholds = sorted(t for t in thresholds if math.isfinite(t) and t >= 0.0)
    if len(sorted_thresholds) <= max_count:
        return [round(t, 12) for t in sorted_thresholds]
    indices = np.linspace(0, len(sorted_thresholds) - 1, max_count).round().astype(int)
    reduced = {sorted_thresholds[int(i)] for i in indices}
    reduced.add(0.0)
    return [round(t, 12) for t in sorted(reduced)]


def threshold_config(cfg: Dict[str, Any], mode: str) -> Dict[str, Any]:
    return {
        "threshold_quantile_targets": cfg["threshold_quantile_targets"],
        "threshold_log_points": cfg["threshold_log_points"],
        "threshold_multiplier_min": cfg["threshold_multiplier_min"],
        "threshold_multiplier_max": cfg["threshold_multiplier_max"],
        "max_formal_thresholds": cfg["smoke_max_formal_thresholds"] if mode == "smoke" else cfg["max_formal_thresholds"],
        "perplexity_tolerance": cfg["perplexity_tolerance"],
        "fallback_perplexity_tolerance": cfg["fallback_perplexity_tolerance"],
    }


def evaluate_reference_split(
    model: Any,
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
    split: str,
    mode: str,
) -> Dict[str, Any]:
    sample_results: List[Dict[str, Any]] = []
    totals = {
        "continuous_nll": 0.0,
        "local_nll": 0.0,
        "continuous_count": 0,
        "local_count": 0,
    }
    top1_values = []
    t0 = time.time()
    for sample in tqdm(samples, desc=f"{split} reference", leave=False):
        result = evaluate_reference_sample(model, sample, device)
        row = asdict(result)
        sample_results.append(row)
        totals["continuous_nll"] += result.continuous_nll
        totals["local_nll"] += result.local_nll
        totals["continuous_count"] += result.continuous_count
        totals["local_count"] += result.local_count
        top1_values.append(result.top1_agreement_local_vs_continuous)

    continuous_ppl = ppl(totals["continuous_nll"], totals["continuous_count"])
    local_ppl = ppl(totals["local_nll"], totals["local_count"])
    return {
        "timestamp_utc": utc_now(),
        "split": split,
        "mode": mode,
        "sample_count": len(samples),
        "sequence_length": int(samples[0]["input_ids"].numel()) if samples else 0,
        "prediction_token_count": int(totals["continuous_count"]),
        "continuous_nll": totals["continuous_nll"],
        "continuous_count": totals["continuous_count"],
        "continuous_cross_entropy": cross_entropy(totals["continuous_nll"], totals["continuous_count"]),
        "continuous_perplexity": continuous_ppl,
        "local_only_nll": totals["local_nll"],
        "local_only_count": totals["local_count"],
        "local_only_cross_entropy": cross_entropy(totals["local_nll"], totals["local_count"]),
        "local_only_perplexity": local_ppl,
        "local_vs_continuous_pct": 100.0 * (local_ppl / continuous_ppl - 1.0),
        "top1_agreement_local_vs_continuous_mean": float(np.mean(top1_values)) if top1_values else None,
        "runtime_seconds": time.time() - t0,
        "trajectory_names": ["Continuous", "Local-only"],
        "samples": sample_results,
    }


def evaluate_reference_sample(model: Any, sample: Dict[str, Any], device: torch.device) -> ReferenceSampleResult:
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    loss_mask = sample["loss_mask"].unsqueeze(0).to(device)
    states = initial_states(model, input_ids)
    hidden0 = states["hidden"]

    with torch.no_grad():
        continuous = hidden0
        local = hidden0
        for layer_index in range(model.config.num_hidden_layers):
            continuous = apply_layer(model, layer_index, continuous, states, local_only=False)
            local = apply_layer(model, layer_index, local, states, local_only=True)
        continuous_logits = final_logits(model, continuous)
        local_logits = final_logits(model, local)
        continuous_nll, continuous_count = nll_from_logits(continuous_logits, input_ids, loss_mask)
        local_nll, local_count = nll_from_logits(local_logits, input_ids, loss_mask)
        top1 = top1_agreement(local_logits, continuous_logits, loss_mask)

    return ReferenceSampleResult(
        sample_id=sample["sample_id"],
        split=sample["split"],
        block_index=int(sample["block_index"]),
        n_tokens=int(input_ids.numel()),
        prediction_tokens=int(sample["prediction_tokens"]),
        continuous_nll=continuous_nll,
        local_nll=local_nll,
        continuous_count=continuous_count,
        local_count=local_count,
        top1_agreement_local_vs_continuous=top1,
    )


def evaluate_selective_split(
    model: Any,
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    threshold: float,
    device: torch.device,
    split: str,
    run_tag: str,
    root: Optional[Path],
    audit: bool,
    reference_metrics: Optional[Dict[str, Any]],
    representative_index: int = 0,
) -> Dict[str, Any]:
    sample_results: List[Dict[str, Any]] = []
    totals = {
        "selective_nll": 0.0,
        "selective_count": 0,
        "active_token_layers": 0,
        "valid_token_layers": 0,
    }
    layer_count = int(model.config.num_hidden_layers)
    seq_len = int(samples[0]["input_ids"].numel()) if samples else 0
    layer_active_counts = np.zeros(layer_count, dtype=np.float64)
    layer_valid_counts = np.zeros(layer_count, dtype=np.float64)
    position_active_counts = np.zeros(seq_len, dtype=np.float64)
    position_valid_counts = np.zeros(seq_len, dtype=np.float64)
    max_depth_values: List[int] = []
    terminal_latent_norms: List[float] = []
    logit_kl_values: List[float] = []
    top1_values: List[float] = []
    representative_matrix: Optional[List[List[int]]] = None
    representative_sample_id: Optional[str] = None
    t0 = time.time()

    for sample_index, sample in enumerate(
        tqdm(samples, desc=f"{split} selective tau={threshold:.5g}", leave=False)
    ):
        result = evaluate_selective_sample(
            model=model,
            sample=sample,
            cfg=cfg,
            threshold=threshold,
            device=device,
            audit=audit,
            reference_states=None,
        )
        row = asdict(result)
        sample_results.append(row)
        totals["selective_nll"] += result.selective_nll
        totals["selective_count"] += result.selective_count
        matrix = np.array(result.realization_matrix, dtype=np.int64)
        valid_tokens = sample["valid_gate_tokens"]
        totals["active_token_layers"] += int(matrix.sum())
        totals["valid_token_layers"] += int(matrix.shape[0] * valid_tokens)
        layer_active_counts += matrix.sum(axis=1)
        layer_valid_counts += valid_tokens
        position_active_counts += matrix.sum(axis=0)
        position_valid_counts += matrix.shape[0] * np.array(sample["gate_valid_mask"], dtype=np.float64)
        max_depth_values.append(result.max_consecutive_latent_depth)
        terminal_latent_norms.append(result.terminal_latent_norm)
        if result.main_logit_kl_vs_continuous is not None:
            logit_kl_values.append(result.main_logit_kl_vs_continuous)
        if result.main_top1_agreement_vs_continuous is not None:
            top1_values.append(result.main_top1_agreement_vs_continuous)
        if sample_index == representative_index:
            representative_matrix = result.realization_matrix
            representative_sample_id = result.sample_id
        if root is not None:
            write_json(
                root / "results" / "trajectories" / f"{run_tag}_{split}_selective_{sample['sample_id']}.json",
                row,
            )

    selective_ppl = ppl(totals["selective_nll"], totals["selective_count"])
    continuous_ppl = reference_metrics["continuous_perplexity"] if reference_metrics else None
    local_ppl = reference_metrics["local_only_perplexity"] if reference_metrics else None
    token_layer_ratio = totals["active_token_layers"] / max(totals["valid_token_layers"], 1)
    layer_fractions = np.divide(layer_active_counts, np.maximum(layer_valid_counts, 1.0))
    position_fractions = np.divide(position_active_counts, np.maximum(position_valid_counts, 1.0))
    active_layer_ratio = float(np.mean(layer_active_counts > 0.0)) if len(layer_active_counts) else 0.0

    metrics = {
        "timestamp_utc": utc_now(),
        "split": split,
        "threshold": float(threshold),
        "sample_count": len(samples),
        "sequence_length": seq_len,
        "prediction_token_count": int(totals["selective_count"]),
        "selective_nll": totals["selective_nll"],
        "selective_count": totals["selective_count"],
        "selective_cross_entropy": cross_entropy(totals["selective_nll"], totals["selective_count"]),
        "selective_perplexity": selective_ppl,
        "continuous_perplexity": continuous_ppl,
        "local_only_perplexity": local_ppl,
        "selective_vs_continuous_pct": 100.0 * (selective_ppl / continuous_ppl - 1.0)
        if continuous_ppl
        else None,
        "token_layer_realization_ratio": float(token_layer_ratio),
        "inactive_ratio": float(1.0 - token_layer_ratio),
        "active_layer_ratio": active_layer_ratio,
        "mean_active_token_fraction_per_layer": float(np.mean(layer_fractions)) if len(layer_fractions) else 0.0,
        "mean_active_layers_per_token": float(token_layer_ratio * layer_count),
        "mean_local_autonomy_layers_per_token": float((1.0 - token_layer_ratio) * layer_count),
        "max_consecutive_latent_depth": int(max(max_depth_values)) if max_depth_values else 0,
        "terminal_latent_norm_mean": float(np.mean(terminal_latent_norms)) if terminal_latent_norms else 0.0,
        "per_layer_active_token_fraction": layer_fractions.tolist(),
        "per_position_active_fraction": position_fractions.tolist(),
        "main_logit_kl_vs_continuous_mean": float(np.mean(logit_kl_values)) if logit_kl_values else None,
        "main_top1_agreement_vs_continuous_mean": float(np.mean(top1_values)) if top1_values else None,
        "representative_sample_id": representative_sample_id,
        "representative_realization_matrix": representative_matrix,
        "terminal_forced_release": False,
        "transport": "exact_finite_displacement",
        "update_equation": "h_l=F_l(H_l+xi_l)-L_l(H_l); H_{l+1}=L_l(H_l)+psi_l; xi_{l+1}=h_l-psi_l",
        "runtime_seconds": time.time() - t0,
        "samples": sample_results,
    }
    if root is not None:
        write_representative_matrix(root, metrics, run_tag, split)
        write_layer_position_tables(root, metrics)
    return metrics


def evaluate_selective_sample(
    model: Any,
    sample: Dict[str, Any],
    cfg: Dict[str, Any],
    threshold: float,
    device: torch.device,
    audit: bool,
    reference_states: Optional[List[torch.Tensor]],
) -> SelectiveSampleResult:
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    loss_mask = sample["loss_mask"].unsqueeze(0).to(device)
    valid_mask = sample["gate_valid_mask"].to(device)
    states = initial_states(model, input_ids)
    hidden0 = states["hidden"]
    model_dtype = hidden0.dtype
    layer_count = int(model.config.num_hidden_layers)
    seq_len = int(input_ids.numel())

    if audit or reference_states is not None:
        continuous_states = reference_states or compute_continuous_states(model, hidden0, states)
    else:
        continuous_states = None

    hidden = hidden0.to(torch.float64)
    latent = torch.zeros_like(hidden)
    realization_events: List[List[int]] = []
    active_fractions: List[float] = []
    latent_norms: List[float] = []
    closure_rel: List[float] = []
    closure_abs: List[float] = []
    displacement_rel: List[float] = []
    latent_run_lengths = torch.zeros(seq_len, dtype=torch.int64, device=device)
    latent_depth_values: List[int] = []

    with torch.no_grad():
        for layer_index in range(layer_count):
            z_state = (hidden + latent).to(model_dtype)
            hidden_input = hidden.to(model_dtype)
            full_z = apply_layer(model, layer_index, z_state, states, local_only=False)
            local_h = apply_layer(model, layer_index, hidden_input, states, local_only=True)
            full_z_acc = full_z.to(torch.float64)
            local_h_acc = local_h.to(torch.float64)
            candidate = full_z_acc - local_h_acc
            release, latent, active, _ = apply_token_gate(candidate, local_h, valid_mask, threshold)
            hidden = local_h_acc + release

            inactive_valid = (~active) & valid_mask
            latent_run_lengths = torch.where(
                inactive_valid,
                latent_run_lengths + 1,
                torch.zeros_like(latent_run_lengths),
            )
            latent_depth_values.extend(latent_run_lengths[valid_mask].detach().cpu().tolist())

            realization_events.append(active.to(torch.int64).detach().cpu().tolist())
            active_fractions.append(float(active.sum().item() / max(int(valid_mask.sum().item()), 1)))
            latent_norms.append(float(valid_fro_norm(latent, valid_mask).item()))

            if continuous_states is not None:
                z_next = hidden + latent
                continuous_next = continuous_states[layer_index + 1].to(torch.float64)
                masked_z_error = (z_next - continuous_next) * valid_mask.to(z_next.dtype).view(1, -1, 1)
                masked_full = continuous_next * valid_mask.to(z_next.dtype).view(1, -1, 1)
                closure_rel.append(float(masked_z_error.norm().div(masked_full.norm().clamp_min(EPS)).item()))
                closure_abs.append(float(masked_z_error.abs().max().item()))

                displacement_error = latent - (continuous_next - hidden)
                masked_disp = displacement_error * valid_mask.to(displacement_error.dtype).view(1, -1, 1)
                masked_reference = (continuous_next - hidden) * valid_mask.to(displacement_error.dtype).view(1, -1, 1)
                displacement_rel.append(float(masked_disp.norm().div(masked_reference.norm().clamp_min(EPS)).item()))

        selective_logits = final_logits(model, hidden.to(model_dtype))
        selective_nll, selective_count = nll_from_logits(selective_logits, input_ids, loss_mask)

        reconstructed_rel = None
        reconstructed_abs = None
        kl_value = None
        top1_value = None
        if continuous_states is not None:
            continuous_final = continuous_states[-1]
            continuous_logits = final_logits(model, continuous_final)
            reconstructed_logits = final_logits(model, (hidden + latent).to(model_dtype))
            diff = reconstructed_logits - continuous_logits
            reconstructed_rel = float(diff.norm().div(continuous_logits.norm().clamp_min(EPS)).item())
            reconstructed_abs = float(diff.abs().max().item())
            kl_value = logit_kl(selective_logits, continuous_logits, loss_mask)
            top1_value = top1_agreement(selective_logits, continuous_logits, loss_mask)

    active_matrix = np.array(realization_events, dtype=np.int64)
    active_counts_by_position = active_matrix.sum(axis=0)
    valid_positions = sample["gate_valid_mask"].numpy().astype(bool)
    token_layer_denominator = int(layer_count * sample["valid_gate_tokens"])
    active_total = int(active_matrix.sum())
    max_consecutive_latent_depth = int(max(latent_depth_values)) if latent_depth_values else 0

    return SelectiveSampleResult(
        sample_id=sample["sample_id"],
        split=sample["split"],
        block_index=int(sample["block_index"]),
        n_tokens=seq_len,
        prediction_tokens=int(sample["prediction_tokens"]),
        threshold=float(threshold),
        selective_nll=float(selective_nll),
        selective_count=int(selective_count),
        token_layer_realization_ratio=float(active_total / max(token_layer_denominator, 1)),
        inactive_ratio=float(1.0 - active_total / max(token_layer_denominator, 1)),
        active_layer_ratio=float(np.mean(active_matrix.sum(axis=1) > 0)) if active_matrix.size else 0.0,
        mean_active_token_fraction_per_layer=float(np.mean(active_fractions)) if active_fractions else 0.0,
        mean_active_layers_per_token=float(np.mean(active_counts_by_position[valid_positions])) if valid_positions.any() else 0.0,
        mean_local_autonomy_layers_per_token=float(
            layer_count - np.mean(active_counts_by_position[valid_positions])
        )
        if valid_positions.any()
        else float(layer_count),
        max_consecutive_latent_depth=max_consecutive_latent_depth,
        terminal_latent_norm=float(valid_fro_norm(latent, valid_mask).item()),
        per_layer_active_fraction=active_fractions,
        per_layer_latent_norm=latent_norms,
        per_position_active_fraction=(active_counts_by_position / max(layer_count, 1)).astype(float).tolist(),
        per_position_mean_active_layers=active_counts_by_position.astype(float).tolist(),
        latent_depth_distribution=[int(x) for x in latent_depth_values],
        realization_matrix=realization_events,
        closure_relative_by_layer=closure_rel if continuous_states is not None else None,
        closure_absolute_by_layer=closure_abs if continuous_states is not None else None,
        latent_displacement_relative_by_layer=displacement_rel if continuous_states is not None else None,
        reconstructed_logit_relative_error=reconstructed_rel,
        reconstructed_logit_max_abs_error=reconstructed_abs,
        main_logit_kl_vs_continuous=kl_value,
        main_top1_agreement_vs_continuous=top1_value,
    )


def compute_continuous_states(model: Any, hidden0: torch.Tensor, states: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    states_by_layer = [hidden0]
    continuous = hidden0
    with torch.no_grad():
        for layer_index in range(model.config.num_hidden_layers):
            continuous = apply_layer(model, layer_index, continuous, states, local_only=False)
            states_by_layer.append(continuous)
    return states_by_layer


def top1_agreement(logits_a: torch.Tensor, logits_b: torch.Tensor, loss_mask: torch.Tensor) -> float:
    pred_a = logits_a[:, :-1, :].argmax(dim=-1)
    pred_b = logits_b[:, :-1, :].argmax(dim=-1)
    mask = loss_mask[:, 1:].contiguous()
    if not bool(mask.any().item()):
        return 0.0
    return float((pred_a[mask] == pred_b[mask]).to(torch.float32).mean().item())


def logit_kl(logits_a: torch.Tensor, logits_b: torch.Tensor, loss_mask: torch.Tensor) -> float:
    mask = loss_mask[:, 1:].contiguous().view(-1)
    a = logits_a[:, :-1, :].contiguous().view(-1, logits_a.size(-1))[mask]
    b = logits_b[:, :-1, :].contiguous().view(-1, logits_b.size(-1))[mask]
    log_p = torch.nn.functional.log_softmax(a, dim=-1)
    q = torch.nn.functional.softmax(b, dim=-1)
    return float(torch.nn.functional.kl_div(log_p, q, reduction="batchmean").item())


def select_threshold(scan: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    tolerance = float(cfg["perplexity_tolerance"])
    fallback = float(cfg["fallback_perplexity_tolerance"])

    def nontrivial(row: Dict[str, Any]) -> bool:
        ratio = float(row["token_layer_realization_ratio"])
        return 0.0 < ratio < 0.999

    eligible = [
        row
        for row in scan
        if nontrivial(row)
        and row["continuous_perplexity"] is not None
        and row["selective_perplexity"] <= row["continuous_perplexity"] * (1.0 + tolerance)
    ]
    rule = "primary"
    if not eligible:
        eligible = [
            row
            for row in scan
            if nontrivial(row)
            and row["continuous_perplexity"] is not None
            and row["selective_perplexity"] <= row["continuous_perplexity"] * (1.0 + fallback)
        ]
        rule = "fallback"
    if eligible:
        selected = sorted(
            eligible,
            key=lambda r: (
                r["token_layer_realization_ratio"],
                r["selective_perplexity"],
                r["terminal_latent_norm_mean"],
            ),
        )[0]
        if rule == "primary":
            reason = "lowest token-layer realization ratio satisfying <= 1.06x validation Continuous perplexity"
        else:
            reason = "fallback: lowest token-layer realization ratio satisfying <= 1.10x validation Continuous perplexity"
    else:
        selected = sorted(scan, key=lambda r: (r["selective_vs_continuous_pct"], r["token_layer_realization_ratio"]))[0]
        reason = "no nontrivial threshold met the predefined 1.06x or 1.10x validation rule; selected lowest validation perplexity deviation"
        rule = "none_met"
    return {
        "timestamp_utc": utc_now(),
        "threshold": selected["threshold"],
        "reason": reason,
        "rule": rule,
        "validation_continuous_perplexity": selected["continuous_perplexity"],
        "validation_selective_perplexity": selected["selective_perplexity"],
        "validation_selective_vs_continuous_pct": selected["selective_vs_continuous_pct"],
        "validation_token_layer_realization_ratio": selected["token_layer_realization_ratio"],
        "validation_inactive_ratio": selected["inactive_ratio"],
        "validation_active_layer_ratio": selected["active_layer_ratio"],
        "validation_terminal_latent_norm_mean": selected["terminal_latent_norm_mean"],
        "test_not_used_for_selection": True,
    }


def choose_test_tradeoff_thresholds(thresholds: Sequence[float], selected: float, cfg: Dict[str, Any]) -> List[float]:
    sorted_thresholds = sorted(set(float(x) for x in thresholds))
    if selected not in sorted_thresholds:
        sorted_thresholds.append(float(selected))
        sorted_thresholds = sorted(sorted_thresholds)
    target_count = min(int(cfg["test_tradeoff_threshold_count"]), len(sorted_thresholds))
    indices = set(np.linspace(0, len(sorted_thresholds) - 1, target_count).round().astype(int).tolist())
    selected_index = min(range(len(sorted_thresholds)), key=lambda i: abs(sorted_thresholds[i] - selected))
    indices.add(selected_index)
    return [sorted_thresholds[i] for i in sorted(indices)]


def make_full_reference_payload(
    reference: Dict[str, Any],
    cfg: Dict[str, Any],
    model_meta: Dict[str, Any],
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return common_result_payload(
        cfg=cfg,
        model_meta=model_meta,
        split=reference["split"],
        metrics={
            "trajectory": "Continuous",
            "cross_entropy": reference["continuous_cross_entropy"],
            "perplexity": reference["continuous_perplexity"],
            "nll": reference["continuous_nll"],
            "prediction_token_count": reference["prediction_token_count"],
        },
        metric_definitions={
            "Continuous": "Original pretrained Transformer with full causal self-attention at every layer.",
        },
        sample_manifest=sample_manifest_for_json(samples),
    )


def make_local_only_payload(
    reference: Dict[str, Any],
    cfg: Dict[str, Any],
    model_meta: Dict[str, Any],
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return common_result_payload(
        cfg=cfg,
        model_meta=model_meta,
        split=reference["split"],
        metrics={
            "trajectory": "Local-only",
            "cross_entropy": reference["local_only_cross_entropy"],
            "perplexity": reference["local_only_perplexity"],
            "nll": reference["local_only_nll"],
            "prediction_token_count": reference["prediction_token_count"],
            "vs_continuous_pct": reference["local_vs_continuous_pct"],
        },
        metric_definitions={
            "Local-only": "Same decoder layers with a same-position-only attention mask; residual stream, RMSNorm, MLP, RoPE, and layer order are retained.",
        },
        sample_manifest=sample_manifest_for_json(samples),
    )


def make_main_test_payload(
    main_test: Dict[str, Any],
    reference: Dict[str, Any],
    selected: Dict[str, Any],
    cfg: Dict[str, Any],
    model_meta: Dict[str, Any],
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics = strip_samples(main_test)
    metrics.update(
        {
            "continuous_cross_entropy": reference["continuous_cross_entropy"],
            "continuous_perplexity": reference["continuous_perplexity"],
            "local_only_cross_entropy": reference["local_only_cross_entropy"],
            "local_only_perplexity": reference["local_only_perplexity"],
            "selected_threshold": selected,
        }
    )
    return common_result_payload(
        cfg=cfg,
        model_meta=model_meta,
        split=main_test["split"],
        metrics=metrics,
        metric_definitions=metric_definitions(),
        sample_manifest=sample_manifest_for_json(samples),
    )


def common_result_payload(
    cfg: Dict[str, Any],
    model_meta: Dict[str, Any],
    split: str,
    metrics: Dict[str, Any],
    metric_definitions: Dict[str, Any],
    sample_manifest: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    sample_manifest = sample_manifest or []
    return {
        "timestamp_utc": utc_now(),
        "config": serializable_config(cfg),
        "git_commit": git_output(["git", "rev-parse", "HEAD"]),
        "git_branch": git_output(["git", "branch", "--show-current"]),
        "model": model_meta,
        "data": {
            "dataset_name": cfg["dataset_name"],
            "dataset_config": cfg["dataset_config"],
            "dataset_revision": cfg.get("dataset_revision"),
            "split": split,
            "sequence_length": int(cfg["sequence_length"]),
            "sample_selection": "fixed contiguous token blocks from split start",
            "sample_count": len(sample_manifest),
            "samples": sample_manifest,
            "token_ranges": [
                {
                    "sample_id": row["sample_id"],
                    "token_start": row["token_start"],
                    "token_end": row["token_end"],
                }
                for row in sample_manifest
            ],
            "seed": int(cfg["seed"]),
        },
        "dtype": cfg["dtype"],
        "device": cfg["device"],
        "code_version": "exact_finite_transport",
        "metrics": metrics,
        "metric_definitions": metric_definitions,
    }


def serializable_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(cfg))


def metric_definitions() -> Dict[str, Any]:
    return {
        "selective_perplexity": "Language-modeling perplexity from realized state H_L only; no terminal forced release.",
        "token_layer_realization_ratio": "Active realized token-layer count divided by valid token-layer count; token 0 excluded.",
        "inactive_ratio": "One minus token-layer realization ratio.",
        "active_layer_ratio": "Fraction of layers with at least one realized valid token.",
        "mean_active_layers_per_token": "Average number of realized layers per valid token.",
        "mean_local_autonomy_layers_per_token": "Average number of non-realized local-autonomy layers per valid token.",
        "max_consecutive_latent_depth": "Maximum consecutive inactive depth over valid tokens.",
        "terminal_latent_norm_mean": "Mean Frobenius norm of xi_L over valid tokens.",
        "closure_error": "Layerwise norm of H_l + xi_l - H_l^F divided by norm of H_l^F.",
        "latent_displacement_error": "Layerwise norm of xi_l - (H_l^F-H_l) divided by norm of H_l^F-H_l.",
        "reconstructed_logit_error": "Error between logits from Z_L=H_L+xi_L and Continuous logits.",
        "fate_identity_error": "Error in phi_l = psi_l + (xi_{l+1}-xi_l) + [xi_l - T_l(xi_l)] on an audit subset.",
    }


def sample_manifest_for_json(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for sample in samples:
        rows.append(
            {
                "sample_id": sample["sample_id"],
                "split": sample["split"],
                "block_index": int(sample["block_index"]),
                "token_start": int(sample["token_start"]),
                "token_end": int(sample["token_end"]),
                "sequence_length": int(sample["input_ids"].numel()),
                "prediction_tokens": int(sample["prediction_tokens"]),
                "valid_gate_tokens": int(sample["valid_gate_tokens"]),
            }
        )
    return rows


def make_exact_closure_audit(
    main_test: Dict[str, Any],
    cfg: Dict[str, Any],
    model_meta: Dict[str, Any],
    samples: List[Dict[str, Any]],
) -> Dict[str, Any]:
    rel_values = flatten_optional(sample.get("closure_relative_by_layer") for sample in main_test["samples"])
    abs_values = flatten_optional(sample.get("closure_absolute_by_layer") for sample in main_test["samples"])
    disp_values = flatten_optional(sample.get("latent_displacement_relative_by_layer") for sample in main_test["samples"])
    logit_rel_values = [
        sample["reconstructed_logit_relative_error"]
        for sample in main_test["samples"]
        if sample.get("reconstructed_logit_relative_error") is not None
    ]
    logit_abs_values = [
        sample["reconstructed_logit_max_abs_error"]
        for sample in main_test["samples"]
        if sample.get("reconstructed_logit_max_abs_error") is not None
    ]
    max_rel = max(rel_values) if rel_values else None
    max_abs = max(abs_values) if abs_values else None
    passed = (
        max_rel is not None
        and max_abs is not None
        and max_rel <= float(cfg["closure_relative_tolerance"])
        and max_abs <= float(cfg["closure_absolute_tolerance"])
    )
    return common_result_payload(
        cfg=cfg,
        model_meta=model_meta,
        split=main_test["split"],
        metrics={
            "mean_relative_closure_error": float(np.mean(rel_values)) if rel_values else None,
            "max_relative_closure_error": max_rel,
            "mean_absolute_closure_error": float(np.mean(abs_values)) if abs_values else None,
            "max_absolute_closure_error": max_abs,
            "mean_relative_latent_displacement_error": float(np.mean(disp_values)) if disp_values else None,
            "max_relative_latent_displacement_error": max(disp_values) if disp_values else None,
            "mean_reconstructed_logit_relative_error": float(np.mean(logit_rel_values)) if logit_rel_values else None,
            "max_reconstructed_logit_relative_error": max(logit_rel_values) if logit_rel_values else None,
            "mean_reconstructed_logit_max_abs_error": float(np.mean(logit_abs_values)) if logit_abs_values else None,
            "max_reconstructed_logit_max_abs_error": max(logit_abs_values) if logit_abs_values else None,
            "relative_tolerance": float(cfg["closure_relative_tolerance"]),
            "absolute_tolerance": float(cfg["closure_absolute_tolerance"]),
            "passes_fp32_target": bool(passed),
            "dtype": cfg["dtype"],
            "audit_scope": "main test samples with independent Continuous hidden-state trajectory",
        },
        metric_definitions=metric_definitions(),
        sample_manifest=sample_manifest_for_json(samples),
    )


def run_fate_identity_audit(
    model: Any,
    samples: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    model_meta: Dict[str, Any],
    threshold: float,
    device: torch.device,
    mode: str,
) -> Dict[str, Any]:
    rel_errors: List[float] = []
    abs_errors: List[float] = []
    layer_rows: List[Dict[str, Any]] = []
    for sample in tqdm(samples, desc="fate identity audit", leave=False):
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        valid_mask = sample["gate_valid_mask"].to(device)
        states = initial_states(model, input_ids)
        hidden = states["hidden"]
        latent = torch.zeros_like(hidden)
        with torch.no_grad():
            for layer_index in range(model.config.num_hidden_layers):
                full_h = apply_layer(model, layer_index, hidden, states, local_only=False)
                local_h = apply_layer(model, layer_index, hidden, states, local_only=True)
                z_state = hidden + latent
                full_z = apply_layer(model, layer_index, z_state, states, local_only=False)
                transport = full_z - full_h
                phi = full_h - local_h
                candidate = full_z - local_h
                release, latent_next, active, _ = apply_token_gate(candidate, local_h, valid_mask, threshold)
                rhs = release + (latent_next - latent) + (latent - transport)
                masked_error = (phi - rhs) * valid_mask.to(phi.dtype).view(1, -1, 1)
                masked_phi = phi * valid_mask.to(phi.dtype).view(1, -1, 1)
                abs_error = float(masked_error.abs().max().item())
                rel_error = float(masked_error.norm().div(masked_phi.norm().clamp_min(EPS)).item())
                abs_errors.append(abs_error)
                rel_errors.append(rel_error)
                layer_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "layer": layer_index,
                        "relative_error": rel_error,
                        "max_absolute_error": abs_error,
                        "active_token_count": int(active.sum().item()),
                    }
                )
                hidden = local_h + release
                latent = latent_next
    return common_result_payload(
        cfg=cfg,
        model_meta=model_meta,
        split="validation",
        metrics={
            "mode": mode,
            "threshold": float(threshold),
            "identity": "phi_l = psi_l + (xi_{l+1}-xi_l) + [xi_l - T_l(xi_l)]",
            "transport_definition": "T_l(xi_l)=F_l(H_l+xi_l)-F_l(H_l)",
            "max_relative_error": max(rel_errors) if rel_errors else None,
            "mean_relative_error": float(np.mean(rel_errors)) if rel_errors else None,
            "max_absolute_error": max(abs_errors) if abs_errors else None,
            "mean_absolute_error": float(np.mean(abs_errors)) if abs_errors else None,
            "rows": layer_rows,
        },
        metric_definitions=metric_definitions(),
        sample_manifest=sample_manifest_for_json(samples),
    )


def flatten_optional(values: Any) -> List[float]:
    output: List[float] = []
    for item in values:
        if item is None:
            continue
        output.extend(float(x) for x in item)
    return output


def max_or_none(values: Optional[List[float]]) -> Optional[float]:
    return max(values) if values else None


def strip_samples(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in metrics.items() if k not in {"samples", "representative_realization_matrix"}}


def write_representative_matrix(root: Path, metrics: Dict[str, Any], run_tag: str, split: str) -> None:
    matrix = metrics.get("representative_realization_matrix") or []
    payload = {
        "sample_id": metrics.get("representative_sample_id"),
        "threshold": metrics["threshold"],
        "rows": "Transformer layers",
        "columns": "token positions; token 0 is present but excluded from primary denominator",
        "realization_matrix": matrix,
    }
    write_json(root / "results" / "metrics" / "representative_realization_matrix.json", payload)
    write_json(root / "results" / "trajectories" / f"{run_tag}_{split}_representative_realization_matrix.json", payload)
    csv_path = root / "results" / "tables" / "representative_realization_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["layer"] + [f"token_{i}" for i in range(len(matrix[0]) if matrix else 0)])
        for layer_index, row in enumerate(matrix):
            writer.writerow([layer_index] + row)


def write_layer_position_tables(root: Path, metrics: Dict[str, Any]) -> None:
    write_csv(
        root / "results" / "tables" / "per_layer_metrics.csv",
        [
            {
                "layer": idx,
                "active_token_fraction": value,
            }
            for idx, value in enumerate(metrics["per_layer_active_token_fraction"])
        ],
        ["layer", "active_token_fraction"],
    )
    write_csv(
        root / "results" / "tables" / "per_position_metrics.csv",
        [
            {
                "token_position": idx,
                "active_fraction": value,
            }
            for idx, value in enumerate(metrics["per_position_active_fraction"])
        ],
        ["token_position", "active_fraction"],
    )


def write_threshold_tradeoff(root: Path, scan: List[Dict[str, Any]], split: str) -> None:
    rows = []
    for row in scan:
        rows.append(
            {
                "split": split,
                "threshold": row["threshold"],
                "continuous_perplexity": row["continuous_perplexity"],
                "selective_perplexity": row["selective_perplexity"],
                "selective_vs_continuous_pct": row["selective_vs_continuous_pct"],
                "token_layer_realization_ratio": row["token_layer_realization_ratio"],
                "inactive_ratio": row["inactive_ratio"],
                "active_layer_ratio": row["active_layer_ratio"],
                "mean_active_layers_per_token": row["mean_active_layers_per_token"],
                "max_consecutive_latent_depth": row["max_consecutive_latent_depth"],
                "terminal_latent_norm_mean": row["terminal_latent_norm_mean"],
            }
        )
    filename = "threshold_tradeoff.csv" if split == "validation" else f"threshold_tradeoff_{split}.csv"
    write_csv(
        root / "results" / "tables" / filename,
        rows,
        [
            "split",
            "threshold",
            "continuous_perplexity",
            "selective_perplexity",
            "selective_vs_continuous_pct",
            "token_layer_realization_ratio",
            "inactive_ratio",
            "active_layer_ratio",
            "mean_active_layers_per_token",
            "max_consecutive_latent_depth",
            "terminal_latent_norm_mean",
        ],
    )


def write_summary_tables(
    root: Path,
    main_test: Dict[str, Any],
    reference: Dict[str, Any],
    validation_sweep: List[Dict[str, Any]],
    selected: Dict[str, Any],
) -> None:
    rows = [
        {
            "trajectory": "Continuous",
            "split": "test",
            "threshold": "",
            "cross_entropy": reference["continuous_cross_entropy"],
            "perplexity": reference["continuous_perplexity"],
            "vs_continuous_pct": 0.0,
            "token_layer_realization_ratio": 1.0,
            "terminal_latent_norm": 0.0,
        },
        {
            "trajectory": "Local-only",
            "split": "test",
            "threshold": "",
            "cross_entropy": reference["local_only_cross_entropy"],
            "perplexity": reference["local_only_perplexity"],
            "vs_continuous_pct": reference["local_vs_continuous_pct"],
            "token_layer_realization_ratio": 0.0,
            "terminal_latent_norm": "",
        },
        {
            "trajectory": "Selective",
            "split": "test",
            "threshold": main_test["threshold"],
            "cross_entropy": main_test["selective_cross_entropy"],
            "perplexity": main_test["selective_perplexity"],
            "vs_continuous_pct": main_test["selective_vs_continuous_pct"],
            "token_layer_realization_ratio": main_test["token_layer_realization_ratio"],
            "terminal_latent_norm": main_test["terminal_latent_norm_mean"],
        },
    ]
    write_csv(
        root / "results" / "tables" / "main_summary.csv",
        rows,
        [
            "trajectory",
            "split",
            "threshold",
            "cross_entropy",
            "perplexity",
            "vs_continuous_pct",
            "token_layer_realization_ratio",
            "terminal_latent_norm",
        ],
    )
    write_json(
        root / "results" / "metrics" / "main_summary.json",
        {
            "selected_threshold": selected,
            "validation_rule_source": "threshold_sweep_validation.json",
            "rows": rows,
        },
    )


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
