#!/usr/bin/env python3
"""Run the C. elegans head-network V4 exact-transport experiment.

Inputs are rebuilt from the official pinned source into an external cache.
Numerical kernels retain the frozen V4 definition.
The final method uses the exact finite displacement

    h_t = Phi_F(x_t + r_t) - Phi_L(x_t)

instead of a tangent/JVP approximation.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd


jax.config.update("jax_enable_x64", True)

MODULE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = MODULE_ROOT.parent
ROOT = Path(os.environ.get("CELEGANS_OUTPUT_DIR", str(MODULE_ROOT / "outputs"))).expanduser().resolve()
FROZEN_ROOT = Path(os.environ.get("CELEGANS_DATA_DIR", "~/.cache/causal-fate/celegans/prepared")).expanduser().resolve()

from celegans_dtpr_v4.dynamics import DynamicsModel, ModelParams, initial_state, normalize_connectome, params_from_config  # noqa: E402
from celegans_dtpr_v4.metrics import matrix_metrics  # noqa: E402


NONLINEAR_DEFECT_COLOR = "#d97904"


@dataclass(frozen=True)
class FrozenData:
    neuron_ids: np.ndarray
    chemical: np.ndarray
    gap: np.ndarray
    reversal: np.ndarray
    observed_mask: np.ndarray
    main_mask: np.ndarray
    q10_mask: np.ndarray
    all_valid_mask: np.ndarray
    valid_kernel_mask: np.ndarray
    empirical: np.ndarray
    empirical_raw: np.ndarray
    empirical_30: np.ndarray
    empirical_raw_30: np.ndarray
    empirical_q: np.ndarray
    occurrence: np.ndarray
    provenance: dict[str, Any]
    frozen_config: dict[str, Any]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SimResult:
    name: str
    transport_mode: str
    dt: float
    base_dt: float
    refinement: int
    decision_interval_s: float
    decision_steps: int
    lambda0: float
    block_threshold: float
    gate_mode: str
    stimulus_indices: np.ndarray
    states: np.ndarray
    propagation: np.ndarray
    raw_propagation: np.ndarray
    residuals: np.ndarray | None = None
    transported: np.ndarray | None = None
    fresh: np.ndarray | None = None
    realized: np.ndarray | None = None
    event_indicator: np.ndarray | None = None
    active_blocks: np.ndarray | None = None
    event_counts: np.ndarray | None = None
    closure_errors: np.ndarray | None = None
    closure_relative_errors: np.ndarray | None = None
    nonlinear_defect_norms: np.ndarray | None = None
    residual_norms: np.ndarray | None = None
    activity_norms: np.ndarray | None = None
    opportunity_gains: np.ndarray | None = None
    inter_event_intervals_s: np.ndarray | None = None

    @property
    def active_ratio(self) -> float:
        if self.event_indicator is None or self.event_indicator.size == 0:
            return float("nan")
        # V4 stores one event row per decision interval. Divide by internal
        # decision steps to report the same physical-time opportunity scale used
        # by the propagation model.
        return float(np.count_nonzero(self.event_indicator) / max(self.event_indicator.size * self.decision_steps, 1))

    @property
    def events_per_second(self) -> float:
        if self.event_indicator is None or self.event_indicator.size == 0:
            return float("nan")
        duration_s = self.decision_interval_s * max(self.event_indicator.shape[0], 1)
        return float(np.count_nonzero(self.event_indicator) / max(len(self.stimulus_indices) * duration_s, 1e-12))

    @property
    def mean_inter_event_interval_s(self) -> float:
        if self.inter_event_intervals_s is None or self.inter_event_intervals_s.size == 0:
            return float("inf")
        return float(np.mean(self.inter_event_intervals_s))

    @property
    def max_latent_norm(self) -> float:
        if self.residuals is None:
            return float("nan")
        return float(np.max(np.linalg.norm(self.residuals, axis=2)))

    @property
    def mean_closure_error(self) -> float:
        if self.closure_errors is None or self.closure_errors.size == 0:
            return float("nan")
        return float(np.mean(self.closure_errors))

    @property
    def max_closure_error(self) -> float:
        if self.closure_errors is None or self.closure_errors.size == 0:
            return float("nan")
        return float(np.max(self.closure_errors))

    @property
    def relative_closure_error(self) -> float:
        if self.closure_relative_errors is None or self.closure_relative_errors.size == 0:
            return float("nan")
        return float(np.mean(self.closure_relative_errors))


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, np.generic):
        return sanitize_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


def load_frozen_operating_object() -> FrozenData:
    operating = FROZEN_ROOT / "operating_data"
    frozen_config = load_json(FROZEN_ROOT / "frozen_config.json")
    manifest = load_json(FROZEN_ROOT / "provenance_manifest.json")
    return FrozenData(
        neuron_ids=np.loadtxt(operating / "neuron_ids.txt", dtype=str),
        chemical=np.load(operating / "chemical_adjacency.npy"),
        gap=np.load(operating / "gap_junction_adjacency.npy"),
        reversal=np.load(operating / "chemical_reversal.npy"),
        observed_mask=np.load(operating / "observed_pair_mask.npy"),
        main_mask=np.load(operating / "main_high_confidence_mask.npy"),
        q10_mask=np.load(operating / "q10_mask.npy"),
        all_valid_mask=np.load(operating / "all_observed_valid_kernel_mask.npy"),
        valid_kernel_mask=np.load(operating / "valid_kernel_mask.npy"),
        empirical=np.load(operating / "empirical_propagation_matrix.npy"),
        empirical_raw=np.load(operating / "empirical_raw_dff.npy"),
        empirical_30=np.load(operating / "empirical_30s_propagation_matrix.npy"),
        empirical_raw_30=np.load(operating / "empirical_30s_raw_dff.npy"),
        empirical_q=np.load(operating / "empirical_q.npy"),
        occurrence=np.load(operating / "occurrence_matrix.npy"),
        provenance=load_json(operating / "data_provenance.json"),
        frozen_config=frozen_config,
        manifest=manifest,
    )


def copy_frozen_operating_data(data: FrozenData) -> None:
    out = ROOT / "results" / "operating_data"
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "neuron_ids.txt", data.neuron_ids, fmt="%s")
    for name, array in {
        "chemical_adjacency.npy": data.chemical,
        "gap_junction_adjacency.npy": data.gap,
        "chemical_reversal.npy": data.reversal,
        "observed_pair_mask.npy": data.observed_mask,
        "main_high_confidence_mask.npy": data.main_mask,
        "q10_mask.npy": data.q10_mask,
        "all_observed_valid_kernel_mask.npy": data.all_valid_mask,
        "valid_kernel_mask.npy": data.valid_kernel_mask,
        "empirical_propagation_matrix.npy": data.empirical,
        "empirical_raw_dff.npy": data.empirical_raw,
        "empirical_30s_propagation_matrix.npy": data.empirical_30,
        "empirical_30s_raw_dff.npy": data.empirical_raw_30,
        "empirical_q.npy": data.empirical_q,
        "occurrence_matrix.npy": data.occurrence,
    }.items():
        np.save(out / name, array)
    provenance = dict(data.provenance)
    provenance["v4_source"] = rel(FROZEN_ROOT)
    provenance["v4_freeze_note"] = "V4 reads a self-contained frozen operating object and does not alter empirical windows, masks, observation operator, stimulus definitions, connectome signs, readout, or continuous parameters."
    provenance["v4_manifest"] = data.manifest
    write_json(out / "data_provenance.json", provenance)


def regression_metrics(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = mask & np.isfinite(predicted) & np.isfinite(target)
    y = predicted[valid].astype(float)
    x = target[valid].astype(float)
    out = matrix_metrics(predicted, target, mask)
    if x.size >= 2 and np.var(x) > 0.0:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope, intercept = float("nan"), float("nan")
    out.update(
        {
            "pair_count": int(x.size),
            "regression_slope": float(slope),
            "regression_intercept": float(intercept),
        }
    )
    return out


def frozen_object_audit(data: FrozenData) -> dict[str, Any]:
    observed_count = int(np.count_nonzero(data.observed_mask))
    valid_count = int(np.count_nonzero(data.all_valid_mask))
    main_count = int(np.count_nonzero(data.main_mask))
    stimulus_count = int(np.count_nonzero(data.observed_mask.any(axis=0)))
    response_count = int(np.count_nonzero(data.observed_mask.any(axis=1)))
    audit = {
        "snapshot_root": rel(FROZEN_ROOT),
        "schema_version": data.frozen_config.get("schema_version"),
        "public_sources": data.frozen_config.get("public_sources", {}),
        "manifest_file_count": len(data.manifest.get("array_files", [])),
        "observed_offdiagonal_pairs": observed_count,
        "valid_kernel_observed_pairs": valid_count,
        "high_confidence_q05_pairs": main_count,
        "formal_stimulus_neurons": stimulus_count,
        "formal_response_neurons": response_count,
        "T_star_s": float(data.frozen_config.get("frozen_reference_statistics", {}).get("selected_window_s", 10.0)),
        "observation_operator": "signed_mean",
        "readout": {
            key: value
            for key, value in data.frozen_config.get("readout", {}).items()
            if key != "coefficients"
        },
    }
    write_json(ROOT / "results" / "metrics" / "frozen_operating_object_audit.json", audit)
    md = f"""# Frozen Operating Object Audit

V4 reads its fixed empirical object from `{rel(FROZEN_ROOT)}`. The formal run
does not read any earlier experiment directory.

| Quantity | Value |
| --- | ---: |
| Stimulus neurons | {stimulus_count} |
| Response neurons | {response_count} |
| Observed off-diagonal pairs | {observed_count} |
| Valid-kernel observed pairs | {valid_count} |
| q<0.05 high-confidence pairs | {main_count} |
| T* | {audit['T_star_s']:.1f} s |

The manifest records one SHA-256 hash, shape and dtype record per frozen input
file. The readout manifest is saved separately in
`results/metrics/readout_manifest.json`.
"""
    (ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (ROOT / "reports" / "frozen_operating_object_audit.md").write_text(md, encoding="utf-8")
    return audit


def build_model(data: FrozenData, dt: float, decision_interval_s: float) -> DynamicsModel:
    config = data.frozen_config["config"]
    selected = data.frozen_config["selected_continuous_parameters"]
    cfg_row = {
        "g_chem": float(selected["g_chem"]),
        "g_gap": float(selected["g_gap"]),
        "g_leak": float(selected["g_leak"]),
        "stimulus_amplitude": float(selected["stimulus_amplitude"]),
        "v_half": float(selected["v_half"]),
        "k_slope": float(selected["k_slope"]),
    }
    params = params_from_config(config, cfg_row)
    chem, gap = normalize_connectome(data.chemical, data.gap)
    total_s = float(config["simulation"]["steps"]) * float(config["simulation"]["dt"])
    stimulus_s = float(config["simulation"]["stimulus_steps"]) * float(config["simulation"]["dt"])
    window_s = float(config["simulation"]["selected_observation_window_s"])
    return DynamicsModel(
        n=len(data.neuron_ids),
        chemical=chem,
        gap=gap,
        reversal=data.reversal,
        params=params,
        dt=float(dt),
        steps=int(round(total_s / dt)),
        stimulus_steps=int(round(stimulus_s / dt)),
        response_start_step=0,
        observation_window_steps=int(round(window_s / dt)),
        correction_decision_steps=max(1, int(round(decision_interval_s / dt))),
        readout_gain=np.asarray(data.frozen_config["readout"]["coefficients"], dtype=float),
        calibration_table=pd.DataFrame(),
    )


def state_weight(model: DynamicsModel, config: dict[str, Any]) -> np.ndarray:
    transport = config.get("causal_fate_transport", config.get("dtpr", {}))
    v_scale = float(transport["state_weight_voltage_scale_mV"])
    s_scale = float(transport["state_weight_synapse_scale"])
    return np.concatenate(
        [
            np.full(model.n, 1.0 / max(v_scale, 1e-12)),
            np.full(model.n, 1.0 / max(s_scale, 1e-12)),
        ]
    )


def stimulus_batch(model: DynamicsModel, step: int, stimulus_indices: np.ndarray) -> np.ndarray:
    u = np.zeros((len(stimulus_indices), model.n), dtype=float)
    if step < model.stimulus_steps:
        rows = np.arange(len(stimulus_indices))
        u[rows, stimulus_indices] = model.params.stimulus_amplitude
    return u


def stimulus_sequence(model: DynamicsModel, start_step: int, steps: int, stimulus_indices: np.ndarray) -> np.ndarray:
    return np.asarray([stimulus_batch(model, start_step + offset, stimulus_indices) for offset in range(steps)], dtype=float)


def _rhs_batch(
    x: jnp.ndarray,
    u: jnp.ndarray,
    chemical: jnp.ndarray,
    gap: jnp.ndarray,
    reversal: jnp.ndarray,
    p: ModelParams,
    full: bool,
) -> jnp.ndarray:
    n = chemical.shape[0]
    v = x[:, :n]
    s = x[:, n:]
    phi = 1.0 / (1.0 + jnp.exp(-(v - p.v_half) / p.k_slope))
    ds = p.synapse_rise_rate * phi * (1.0 - s) - p.synapse_decay_rate * s
    gap_current = jnp.zeros_like(v)
    chem_current = jnp.zeros_like(v)
    if full:
        gap_current = p.g_gap * (v * jnp.sum(gap, axis=1)[None, :] - v @ gap.T)
        weighted_s = s @ chemical.T
        weighted_reversal_s = s @ (chemical * reversal).T
        chem_current = p.g_chem * (v * weighted_s - weighted_reversal_s)
    dv = (-p.g_leak * (v - p.resting_potential) - gap_current - chem_current + u) / p.capacitance
    return jnp.concatenate([dv, ds], axis=1)


def make_integrator(model: DynamicsModel, method: str = "rk4"):
    chemical = jnp.asarray(model.chemical)
    gap = jnp.asarray(model.gap)
    reversal = jnp.asarray(model.reversal)
    dt = float(model.dt)

    def clip_state(y: jnp.ndarray) -> jnp.ndarray:
        v = y[:, : model.n]
        s = jnp.clip(y[:, model.n :], 0.0, 1.0)
        return jnp.concatenate([v, s], axis=1)

    @jax.jit
    def full_step(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        if method == "euler":
            y = x + dt * _rhs_batch(x, u, chemical, gap, reversal, model.params, True)
        else:
            k1 = _rhs_batch(x, u, chemical, gap, reversal, model.params, True)
            k2 = _rhs_batch(x + 0.5 * dt * k1, u, chemical, gap, reversal, model.params, True)
            k3 = _rhs_batch(x + 0.5 * dt * k2, u, chemical, gap, reversal, model.params, True)
            k4 = _rhs_batch(x + dt * k3, u, chemical, gap, reversal, model.params, True)
            y = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return clip_state(y)

    @jax.jit
    def local_step(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        if method == "euler":
            y = x + dt * _rhs_batch(x, u, chemical, gap, reversal, model.params, False)
        else:
            k1 = _rhs_batch(x, u, chemical, gap, reversal, model.params, False)
            k2 = _rhs_batch(x + 0.5 * dt * k1, u, chemical, gap, reversal, model.params, False)
            k3 = _rhs_batch(x + 0.5 * dt * k2, u, chemical, gap, reversal, model.params, False)
            k4 = _rhs_batch(x + dt * k3, u, chemical, gap, reversal, model.params, False)
            y = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return clip_state(y)

    return full_step, local_step


def make_block_integrators(model: DynamicsModel, method: str = "rk4"):
    chemical = jnp.asarray(model.chemical)
    gap = jnp.asarray(model.gap)
    reversal = jnp.asarray(model.reversal)
    dt = float(model.dt)

    def clip_state(y: jnp.ndarray) -> jnp.ndarray:
        v = y[:, : model.n]
        s = jnp.clip(y[:, model.n :], 0.0, 1.0)
        return jnp.concatenate([v, s], axis=1)

    def one_step(x: jnp.ndarray, u: jnp.ndarray, full: bool) -> jnp.ndarray:
        if method == "euler":
            y = x + dt * _rhs_batch(x, u, chemical, gap, reversal, model.params, full)
        else:
            k1 = _rhs_batch(x, u, chemical, gap, reversal, model.params, full)
            k2 = _rhs_batch(x + 0.5 * dt * k1, u, chemical, gap, reversal, model.params, full)
            k3 = _rhs_batch(x + 0.5 * dt * k2, u, chemical, gap, reversal, model.params, full)
            k4 = _rhs_batch(x + dt * k3, u, chemical, gap, reversal, model.params, full)
            y = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return clip_state(y)

    @jax.jit
    def full_block(x: jnp.ndarray, us: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def scan_step(carry: jnp.ndarray, u: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            y = one_step(carry, u, True)
            return y, y

        return jax.lax.scan(scan_step, x, us)

    @jax.jit
    def local_block(x: jnp.ndarray, us: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        def scan_step(carry: jnp.ndarray, u: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            y = one_step(carry, u, False)
            return y, y

        return jax.lax.scan(scan_step, x, us)

    return full_block, local_block


def apply_readout(raw_response: np.ndarray, readout_gain: np.ndarray) -> np.ndarray:
    return readout_gain[:, None] * raw_response


def readout_feature(states: np.ndarray, model: DynamicsModel) -> np.ndarray:
    end = min(states.shape[0], model.response_start_step + model.observation_window_steps + 1)
    v = states[model.response_start_step : end, :, : model.n]
    dv = v - model.params.resting_potential
    if v.shape[0] <= 1:
        response = dv[0]
    else:
        response = np.trapezoid(dv, dx=model.dt, axis=0) / max((v.shape[0] - 1) * model.dt, 1e-12)
    return response.T


def embed_propagation(
    raw_subset: np.ndarray,
    propagation_subset: np.ndarray,
    stimulus_indices: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.full((n, n), np.nan, dtype=float)
    prop = np.full((n, n), np.nan, dtype=float)
    raw[:, stimulus_indices] = raw_subset
    prop[:, stimulus_indices] = propagation_subset
    return raw, prop


def simulate_full_or_local(
    model: DynamicsModel,
    stimulus_indices: np.ndarray,
    full: bool,
    name: str,
    integrator: str = "rk4",
) -> SimResult:
    full_block, local_block = make_block_integrators(model, integrator)
    block_fn = full_block if full else local_block
    x0 = initial_state(model.n, model.params)
    x = np.tile(x0[None, :], (len(stimulus_indices), 1))
    states = [x.copy()]
    step = 0
    while step < model.steps:
        block_steps = min(model.correction_decision_steps, model.steps - step)
        us = stimulus_sequence(model, step, block_steps, stimulus_indices)
        x_jax, path_jax = block_fn(jnp.asarray(x), jnp.asarray(us))
        path = np.asarray(path_jax)
        states.extend([path[i].copy() for i in range(path.shape[0])])
        x = np.asarray(x_jax)
        step += block_steps
    states_arr = np.asarray(states, dtype=np.float64)
    raw_subset = readout_feature(states_arr, model)
    prop_subset = apply_readout(raw_subset, model.readout_gain)
    raw, prop = embed_propagation(raw_subset, prop_subset, stimulus_indices, model.n)
    return SimResult(
        name=name,
        transport_mode="full" if full else "local",
        dt=float(model.dt),
        base_dt=0.1,
        refinement=int(round(0.1 / model.dt)),
        decision_interval_s=model.correction_decision_steps * model.dt,
        decision_steps=model.correction_decision_steps,
        lambda0=float("nan"),
        block_threshold=0.0,
        gate_mode="none",
        stimulus_indices=stimulus_indices.copy(),
        states=states_arr,
        propagation=prop,
        raw_propagation=raw,
    )


def block_map(
    block_fn,
    x: np.ndarray,
    model: DynamicsModel,
    stimulus_indices: np.ndarray,
    start_step: int,
    steps: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    us = stimulus_sequence(model, start_step, steps, stimulus_indices)
    z, path = block_fn(jnp.asarray(x), jnp.asarray(us))
    path_np = np.asarray(path)
    return np.asarray(z), [path_np[i].copy() for i in range(path_np.shape[0])]


def block_jvp(
    full_block,
    model: DynamicsModel,
    stimulus_indices: np.ndarray,
    start_step: int,
    steps: int,
    x: np.ndarray,
    r: np.ndarray,
) -> np.ndarray:
    us = jnp.asarray(stimulus_sequence(model, start_step, steps, stimulus_indices))

    def phi(z: jnp.ndarray) -> jnp.ndarray:
        out, _ = full_block(z, us)
        return out

    _, tangent = jax.jvp(phi, (jnp.asarray(x),), (jnp.asarray(r),))
    return np.asarray(tangent)


def gate_batch(
    h: np.ndarray,
    weight: np.ndarray,
    model: DynamicsModel,
    lambda0: float,
    block_threshold: float = 0.0,
    gate_mode: str = "whole",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if gate_mode != "whole":
        hv = h[:, : model.n]
        hs = h[:, model.n :]
        wv = weight[: model.n]
        ws = weight[model.n :]
        norms = np.sqrt((wv[None, :] * hv) ** 2 + (ws[None, :] * hs) ** 2)
        shrink = np.maximum(1.0 - block_threshold / np.maximum(norms, 1e-12), 0.0)
        gains = 0.5 * np.maximum(norms - block_threshold, 0.0) ** 2
        event_gain = np.sum(gains, axis=1)
        event_active = event_gain > lambda0
        blocks = (shrink > 0.0) & event_active[:, None]
        q = np.zeros_like(h)
        q[:, : model.n] = shrink * h[:, : model.n] * event_active[:, None]
        q[:, model.n :] = shrink * h[:, model.n :] * event_active[:, None]
        return q, blocks, event_active, event_gain

    weighted_norm = np.linalg.norm(weight[None, :] * h, axis=1)
    event_gain = 0.5 * weighted_norm * weighted_norm
    event_active = event_gain > lambda0
    q = np.zeros_like(h)
    q[event_active] = h[event_active]
    blocks = np.repeat(event_active[:, None], model.n, axis=1)
    return q, blocks, event_active, event_gain


def event_intervals(events: np.ndarray, dt: float) -> np.ndarray:
    intervals: list[float] = []
    for stim in range(events.shape[1]):
        times = np.flatnonzero(events[:, stim])
        if len(times) > 1:
            intervals.extend((np.diff(times) * dt).astype(float).tolist())
    return np.asarray(intervals, dtype=float)


def simulate_selective(
    model: DynamicsModel,
    stimulus_indices: np.ndarray,
    transport_mode: str,
    weight: np.ndarray,
    lambda0: float,
    gate_mode: str = "whole",
    block_threshold: float = 0.0,
    integrator: str = "rk4",
    name: str = "selective",
    record_defect: bool = False,
) -> SimResult:
    if transport_mode not in {"finite_displacement", "tangent"}:
        raise ValueError(f"Unknown transport mode: {transport_mode}")
    full_block, local_block = make_block_integrators(model, integrator)
    x0 = initial_state(model.n, model.params)
    x = np.tile(x0[None, :], (len(stimulus_indices), 1))
    r = np.zeros_like(x)
    states = [x.copy()]
    residuals = [r.copy()]
    transported_rows: list[np.ndarray] = []
    fresh_rows: list[np.ndarray] = []
    realized_rows: list[np.ndarray] = []
    event_rows: list[np.ndarray] = []
    block_rows: list[np.ndarray] = []
    event_counts: list[int] = []
    closure_errors: list[float] = []
    closure_relative_errors: list[float] = []
    defect_norms: list[np.ndarray] = []
    residual_norms: list[np.ndarray] = []
    activity_norms: list[np.ndarray] = []
    opportunity_gains: list[np.ndarray] = []

    step = 0
    while step < model.steps:
        block_steps = min(model.correction_decision_steps, model.steps - step)
        local_end, local_path = block_map(local_block, x, model, stimulus_indices, step, block_steps)
        full_base, _ = block_map(full_block, x, model, stimulus_indices, step, block_steps)
        full_perturbed, _ = block_map(full_block, x + r, model, stimulus_indices, step, block_steps)
        fresh = full_base - local_end
        if transport_mode == "finite_displacement":
            transported = full_perturbed - full_base
        else:
            transported = block_jvp(full_block, model, stimulus_indices, step, block_steps, x, r)
            norm = np.linalg.norm(transported)
            if norm > 50.0:
                transported = transported * (50.0 / norm)
        h = full_perturbed - local_end if transport_mode == "finite_displacement" else fresh + transported
        q, block_active, event_active, event_gain = gate_batch(
            h,
            weight,
            model,
            lambda0=lambda0,
            block_threshold=block_threshold,
            gate_mode=gate_mode,
        )
        opportunity_gains.append(event_gain.copy())
        x_next = local_end + q
        r_next = h - q
        if transport_mode == "finite_displacement":
            closure = x_next + r_next - full_perturbed
            closure_norm = np.linalg.norm(closure, axis=1)
            closure_errors.extend(closure_norm.tolist())
            closure_relative_errors.extend((closure_norm / np.maximum(np.linalg.norm(full_perturbed, axis=1), 1e-12)).tolist())
        else:
            closure = x_next + r_next - full_perturbed
            closure_norm = np.linalg.norm(closure, axis=1)
            closure_errors.extend(closure_norm.tolist())
            closure_relative_errors.extend((closure_norm / np.maximum(np.linalg.norm(full_perturbed, axis=1), 1e-12)).tolist())
        if record_defect:
            jvp = block_jvp(full_block, model, stimulus_indices, step, block_steps, x, r)
            defect = full_perturbed - full_base - jvp
            defect_norms.append(np.linalg.norm(defect, axis=1))
            residual_norms.append(np.linalg.norm(r, axis=1))
            activity_norms.append(np.linalg.norm(x[:, : model.n] - model.params.resting_potential, axis=1))

        for offset, state in enumerate(local_path):
            if offset == block_steps - 1:
                states.append(x_next.copy())
            else:
                states.append(state.copy())
        residuals.append(r_next.copy())
        transported_rows.append(transported.copy())
        fresh_rows.append(fresh.copy())
        realized_rows.append(q.copy())
        event_rows.append(event_active.copy())
        block_rows.append(block_active.copy())
        event_counts.append(int(np.count_nonzero(block_active)))
        x, r = x_next, r_next
        step += block_steps

    states_arr = np.asarray(states, dtype=np.float64)
    raw_subset = readout_feature(states_arr, model)
    prop_subset = apply_readout(raw_subset, model.readout_gain)
    raw, prop = embed_propagation(raw_subset, prop_subset, stimulus_indices, model.n)
    events = np.asarray(event_rows, dtype=bool)
    blocks = np.asarray(block_rows, dtype=bool)
    return SimResult(
        name=name,
        transport_mode=transport_mode,
        dt=float(model.dt),
        base_dt=0.1,
        refinement=int(round(0.1 / model.dt)),
        decision_interval_s=model.correction_decision_steps * model.dt,
        decision_steps=model.correction_decision_steps,
        lambda0=float(lambda0),
        block_threshold=float(block_threshold),
        gate_mode=gate_mode,
        stimulus_indices=stimulus_indices.copy(),
        states=states_arr,
        propagation=prop,
        raw_propagation=raw,
        residuals=np.asarray(residuals, dtype=np.float32),
        transported=np.asarray(transported_rows, dtype=np.float32),
        fresh=np.asarray(fresh_rows, dtype=np.float32),
        realized=np.asarray(realized_rows, dtype=np.float32),
        event_indicator=events,
        active_blocks=blocks,
        event_counts=np.asarray(event_counts, dtype=int),
        closure_errors=np.asarray(closure_errors, dtype=float),
        closure_relative_errors=np.asarray(closure_relative_errors, dtype=float),
        nonlinear_defect_norms=np.concatenate(defect_norms) if defect_norms else None,
        residual_norms=np.concatenate(residual_norms) if residual_norms else None,
        activity_norms=np.concatenate(activity_norms) if activity_norms else None,
        opportunity_gains=np.concatenate(opportunity_gains) if opportunity_gains else None,
        inter_event_intervals_s=event_intervals(events, model.dt * model.correction_decision_steps),
    )


def simulate_selective_scheduled(
    model: DynamicsModel,
    stimulus_indices: np.ndarray,
    transport_mode: str,
    event_schedule: np.ndarray,
    weight: np.ndarray,
    integrator: str = "rk4",
    name: str = "scheduled_selective",
    record_defect: bool = False,
) -> SimResult:
    """Selective simulation using pre-fixed event decisions.

    `event_schedule` is indexed by decision interval and local stimulus batch
    position. When a scheduled event is true, the whole finite influence is
    released. This keeps the physical realization times fixed while comparing
    tangent and finite-displacement transport.
    """

    if transport_mode not in {"finite_displacement", "tangent"}:
        raise ValueError(f"Unknown transport mode: {transport_mode}")
    full_block, local_block = make_block_integrators(model, integrator)
    x0 = initial_state(model.n, model.params)
    x = np.tile(x0[None, :], (len(stimulus_indices), 1))
    r = np.zeros_like(x)
    states = [x.copy()]
    residuals = [r.copy()]
    transported_rows: list[np.ndarray] = []
    fresh_rows: list[np.ndarray] = []
    realized_rows: list[np.ndarray] = []
    event_rows: list[np.ndarray] = []
    block_rows: list[np.ndarray] = []
    event_counts: list[int] = []
    closure_errors: list[float] = []
    closure_relative_errors: list[float] = []
    defect_norms: list[np.ndarray] = []
    residual_norms: list[np.ndarray] = []
    activity_norms: list[np.ndarray] = []
    opportunity_gains: list[np.ndarray] = []
    step = 0
    decision_index = 0
    while step < model.steps:
        block_steps = min(model.correction_decision_steps, model.steps - step)
        local_end, local_path = block_map(local_block, x, model, stimulus_indices, step, block_steps)
        full_base, _ = block_map(full_block, x, model, stimulus_indices, step, block_steps)
        full_perturbed, _ = block_map(full_block, x + r, model, stimulus_indices, step, block_steps)
        fresh = full_base - local_end
        if transport_mode == "finite_displacement":
            transported = full_perturbed - full_base
        else:
            transported = block_jvp(full_block, model, stimulus_indices, step, block_steps, x, r)
            norm = np.linalg.norm(transported)
            if norm > 50.0:
                transported = transported * (50.0 / norm)
        h = full_perturbed - local_end if transport_mode == "finite_displacement" else fresh + transported
        if decision_index < event_schedule.shape[0]:
            event_active = np.asarray(event_schedule[decision_index], dtype=bool)
        else:
            event_active = np.zeros(len(stimulus_indices), dtype=bool)
        q = np.zeros_like(h)
        q[event_active] = h[event_active]
        block_active = np.repeat(event_active[:, None], model.n, axis=1)
        event_gain = 0.5 * np.linalg.norm(weight[None, :] * h, axis=1) ** 2
        opportunity_gains.append(event_gain.copy())
        x_next = local_end + q
        r_next = h - q
        closure = x_next + r_next - full_perturbed
        closure_norm = np.linalg.norm(closure, axis=1)
        closure_errors.extend(closure_norm.tolist())
        closure_relative_errors.extend((closure_norm / np.maximum(np.linalg.norm(full_perturbed, axis=1), 1e-12)).tolist())
        if record_defect:
            jvp = block_jvp(full_block, model, stimulus_indices, step, block_steps, x, r)
            defect = full_perturbed - full_base - jvp
            defect_norms.append(np.linalg.norm(defect, axis=1))
            residual_norms.append(np.linalg.norm(r, axis=1))
            activity_norms.append(np.linalg.norm(x[:, : model.n] - model.params.resting_potential, axis=1))
        for offset, state in enumerate(local_path):
            states.append(x_next.copy() if offset == block_steps - 1 else state.copy())
        residuals.append(r_next.copy())
        transported_rows.append(transported.copy())
        fresh_rows.append(fresh.copy())
        realized_rows.append(q.copy())
        event_rows.append(event_active.copy())
        block_rows.append(block_active.copy())
        event_counts.append(int(np.count_nonzero(block_active)))
        x, r = x_next, r_next
        step += block_steps
        decision_index += 1
    states_arr = np.asarray(states, dtype=np.float64)
    raw_subset = readout_feature(states_arr, model)
    prop_subset = apply_readout(raw_subset, model.readout_gain)
    raw, prop = embed_propagation(raw_subset, prop_subset, stimulus_indices, model.n)
    events = np.asarray(event_rows, dtype=bool)
    blocks = np.asarray(block_rows, dtype=bool)
    return SimResult(
        name=name,
        transport_mode=transport_mode,
        dt=float(model.dt),
        base_dt=0.1,
        refinement=int(round(0.1 / model.dt)),
        decision_interval_s=model.correction_decision_steps * model.dt,
        decision_steps=model.correction_decision_steps,
        lambda0=float("nan"),
        block_threshold=0.0,
        gate_mode="scheduled_whole",
        stimulus_indices=stimulus_indices.copy(),
        states=states_arr,
        propagation=prop,
        raw_propagation=raw,
        residuals=np.asarray(residuals, dtype=np.float32),
        transported=np.asarray(transported_rows, dtype=np.float32),
        fresh=np.asarray(fresh_rows, dtype=np.float32),
        realized=np.asarray(realized_rows, dtype=np.float32),
        event_indicator=events,
        active_blocks=blocks,
        event_counts=np.asarray(event_counts, dtype=int),
        closure_errors=np.asarray(closure_errors, dtype=float),
        closure_relative_errors=np.asarray(closure_relative_errors, dtype=float),
        nonlinear_defect_norms=np.concatenate(defect_norms) if defect_norms else None,
        residual_norms=np.concatenate(residual_norms) if residual_norms else None,
        activity_norms=np.concatenate(activity_norms) if activity_norms else None,
        opportunity_gains=np.concatenate(opportunity_gains) if opportunity_gains else None,
        inter_event_intervals_s=event_intervals(events, model.dt * model.correction_decision_steps),
    )


def threshold_candidates_from_gains(gains: np.ndarray, quantiles: list[float]) -> list[tuple[float, float]]:
    finite = gains[np.isfinite(gains)]
    if finite.size == 0:
        return [(0.0, 0.0)]
    candidates: list[tuple[float, float]] = []
    for q in quantiles:
        candidates.append((float(q), float(np.quantile(finite, q))))
    out: list[tuple[float, float]] = []
    seen: set[float] = set()
    for q, value in candidates:
        key = round(value, 15)
        if key not in seen:
            seen.add(key)
            out.append((q, value))
    return out


def fixed_split(data: FrozenData, seed: int = 188) -> tuple[np.ndarray, np.ndarray]:
    formal = np.flatnonzero(data.observed_mask.any(axis=0))
    rng = np.random.default_rng(seed)
    shuffled = formal.copy()
    rng.shuffle(shuffled)
    n_val = int(round(0.70 * len(shuffled)))
    val = np.sort(shuffled[:n_val])
    test = np.sort(shuffled[n_val:])
    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for path, arr in ((data_dir / "validation_stimuli.csv", val), (data_dir / "test_stimuli.csv", test)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "neuron_id"])
            for idx in arr:
                writer.writerow([int(idx), str(data.neuron_ids[idx])])
    return val, test


def read_stimulus_split(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing frozen stimulus split file: {path}")
    rows = pd.read_csv(path)
    return rows["index"].to_numpy(dtype=int)


def column_mask(base_mask: np.ndarray, stimulus_indices: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(base_mask, dtype=bool)
    mask[:, stimulus_indices] = base_mask[:, stimulus_indices]
    return mask


def metrics_row(
    result: SimResult,
    full: SimResult,
    mask: np.ndarray,
    label: str,
) -> dict[str, Any]:
    metrics = regression_metrics(result.propagation, full.propagation, mask)
    return {
        "label": label,
        "transport_mode": result.transport_mode,
        "dt": float(result.dt),
        "refinement": int(result.refinement),
        "decision_interval_s": float(result.decision_interval_s),
        "decision_steps": int(result.decision_steps),
        "lambda0": float(result.lambda0),
        "block_threshold": float(result.block_threshold),
        "gate_mode": result.gate_mode,
        "active_ratio": result.active_ratio,
        "event_count": int(np.count_nonzero(result.event_indicator)) if result.event_indicator is not None else 0,
        "events_per_second": result.events_per_second,
        "mean_inter_realization_interval_s": result.mean_inter_event_interval_s,
        "latent_state_max_norm": result.max_latent_norm,
        "closure_error_mean": result.mean_closure_error,
        "closure_error_max": result.max_closure_error,
        "closure_error_relative_mean": result.relative_closure_error,
        **metrics,
    }


def make_calibrated_full_local(data: FrozenData, model: DynamicsModel, stimulus_indices: np.ndarray) -> tuple[SimResult, SimResult]:
    full = simulate_full_or_local(model, stimulus_indices, True, "fully_realized", "rk4")
    local = simulate_full_or_local(model, stimulus_indices, False, "local_autonomy", "rk4")
    return full, local


def pilot_opportunity_gains(
    data: FrozenData,
    model: DynamicsModel,
    validation_indices: np.ndarray,
    decision_interval_s: float,
    weight: np.ndarray,
) -> np.ndarray:
    pilot_model = replace(
        model,
        correction_decision_steps=max(1, int(round(decision_interval_s / model.dt))),
    )
    pilot = simulate_selective(
        pilot_model,
        validation_indices,
        transport_mode="finite_displacement",
        weight=weight,
        lambda0=-1.0,
        gate_mode="whole",
        name="pilot_opportunities",
    )
    if pilot.opportunity_gains is not None:
        return pilot.opportunity_gains
    # With lambda0 < 0 all decisions activate, but opportunity gains are not
    # stored unless diagnostics are requested. Re-run a cheap diagnostic if
    # necessary.
    diagnostic = simulate_selective(
        pilot_model,
        validation_indices,
        transport_mode="finite_displacement",
        weight=weight,
        lambda0=-1.0,
        gate_mode="whole",
        name="pilot_opportunities",
        record_defect=True,
    )
    return np.asarray(diagnostic.opportunity_gains, dtype=float)


def run_validation_frontier(
    data: FrozenData,
    base_model: DynamicsModel,
    validation_indices: np.ndarray,
    full_validation: SimResult,
    weight: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    decision_intervals = [0.1, 0.5, 1.0]
    quantiles = [0.0, 0.5, 0.9, 0.99]
    rows: list[dict[str, Any]] = []
    results_by_key: dict[int, SimResult] = {}
    val_mask = column_mask(data.main_mask, validation_indices)
    grid_index = 0
    for interval in decision_intervals:
        model = replace(base_model, correction_decision_steps=max(1, int(round(interval / base_model.dt))))
        gains = pilot_opportunity_gains(data, model, validation_indices, interval, weight)
        for q, threshold in threshold_candidates_from_gains(gains, quantiles):
            result = simulate_selective(
                model,
                validation_indices,
                transport_mode="finite_displacement",
                weight=weight,
                lambda0=threshold,
                gate_mode="whole",
                name=f"validation_fd_{interval}_{q}",
            )
            row = metrics_row(result, full_validation, val_mask, f"grid_{grid_index}")
            row.update({"grid_index": grid_index, "threshold_quantile": q, "selected": False})
            rows.append(row)
            results_by_key[grid_index] = result
            grid_index += 1
    df = pd.DataFrame(rows)
    primary = df[
        (df["relative_frobenius_error"] <= 0.02)
        & (df["pearson"] >= 0.999)
        & (df["sign_agreement"] >= 0.99)
    ].copy()
    if not primary.empty:
        selected = primary.sort_values(["events_per_second", "relative_frobenius_error"]).iloc[0].to_dict()
        selected["selection_rule"] = "primary: E_prop<=0.02, rho>=0.999, sign>=0.99; lowest events/sec"
        selected["selection_status"] = "primary_threshold_passed"
    else:
        fallback = df[
            (df["relative_frobenius_error"] <= 0.05)
            & (df["pearson"] >= 0.999)
            & (df["sign_agreement"] >= 0.99)
        ].copy()
        if not fallback.empty:
            selected = fallback.sort_values(["events_per_second", "relative_frobenius_error"]).iloc[0].to_dict()
            selected["selection_rule"] = "fallback: E_prop<=0.05, rho>=0.999, sign>=0.99; lowest events/sec"
            selected["selection_status"] = "fallback_threshold_passed"
        else:
            selected = df.sort_values(["relative_frobenius_error", "events_per_second"]).iloc[0].to_dict()
            selected["selection_rule"] = "best feasible Pareto point; predefined 2% and 5% criteria not met"
            selected["selection_status"] = "best_feasible_thresholds_not_met"
    df.loc[df["grid_index"] == int(selected["grid_index"]), "selected"] = True
    scans = ROOT / "results" / "scans"
    scans.mkdir(parents=True, exist_ok=True)
    df.to_csv(scans / "quality_realization_frontier.csv", index=False)
    selected["selected_grid_index"] = int(selected["grid_index"])
    selected["selected"] = True
    selected["selection_stage"] = "validation_frontier"
    selected["validation_metrics"] = {
        "relative_frobenius_error": float(selected["relative_frobenius_error"]),
        "pearson": float(selected["pearson"]),
        "sign_agreement": float(selected["sign_agreement"]),
        "events_per_second": float(selected["events_per_second"]),
        "active_ratio": float(selected["active_ratio"]),
    }
    selected["threshold"] = float(selected["lambda0"])
    selected["decision_interval"] = float(selected["decision_interval_s"])
    selected["selected_before_test"] = True
    selected["test_used_for_selection"] = False
    return df, selected


def run_integration_convergence(
    data: FrozenData,
    selected: dict[str, Any],
    stimulus_indices: np.ndarray,
    mask: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    full_by_refinement: dict[int, SimResult] = {}
    selective_by_mode: dict[tuple[str, int], SimResult] = {}
    prev_full: SimResult | None = None
    prev_metrics: dict[str, Any] = {}
    converged_refinement = 8
    convergence_note = "finest baseline step/8 retained; rule not satisfied earlier"
    for refinement in [1, 2, 4, 8]:
        dt = 0.1 / refinement
        model = build_model(data, dt, float(selected["decision_interval_s"]))
        weight = state_weight(model, data.frozen_config["config"])
        full = simulate_full_or_local(model, stimulus_indices, True, f"full_h{refinement}", "rk4")
        local = simulate_full_or_local(model, stimulus_indices, False, f"local_h{refinement}", "rk4")
        full_by_refinement[refinement] = full
        full_change = float("nan")
        if prev_full is not None:
            m = column_mask(mask, stimulus_indices)
            valid = m & np.isfinite(full.propagation) & np.isfinite(prev_full.propagation)
            full_change = float(
                np.linalg.norm(full.propagation[valid] - prev_full.propagation[valid])
                / max(np.linalg.norm(full.propagation[valid]), 1e-12)
            )
        for mode in ["tangent", "finite_displacement"]:
            result = simulate_selective(
                model,
                stimulus_indices,
                transport_mode=mode,
                weight=weight,
                lambda0=float(selected["lambda0"]),
                gate_mode="whole",
                block_threshold=0.0,
                name=f"{mode}_h{refinement}",
                record_defect=(mode == "finite_displacement"),
            )
            selective_by_mode[(mode, refinement)] = result
            row = metrics_row(result, full, column_mask(mask, stimulus_indices), f"{mode}_h{refinement}")
            row.update(
                {
                    "full_propagation_relative_change_from_previous_h": full_change,
                    "local_relative_error": regression_metrics(local.propagation, full.propagation, column_mask(mask, stimulus_indices))[
                        "relative_frobenius_error"
                    ],
                }
            )
            if mode in prev_metrics and np.isfinite(full_change):
                row["selective_main_metric_change_from_previous_h"] = abs(
                    row["relative_frobenius_error"] - prev_metrics[mode]["relative_frobenius_error"]
                )
            else:
                row["selective_main_metric_change_from_previous_h"] = float("nan")
            rows.append(row)
            prev_metrics[mode] = row
        if refinement > 1:
            fd_rows = [r for r in rows if r["transport_mode"] == "finite_displacement" and r["refinement"] == refinement]
            if (
                fd_rows
                and np.isfinite(full_change)
                and full_change < 1e-3
                and fd_rows[0]["selective_main_metric_change_from_previous_h"] < 1e-3
            ):
                converged_refinement = refinement
                convergence_note = "first refinement satisfying full propagation and selective metric <1e-3 halving changes"
                break
        prev_full = full
    df = pd.DataFrame(rows)
    scans = ROOT / "results" / "scans"
    scans.mkdir(parents=True, exist_ok=True)
    df.to_csv(scans / "integration_convergence.csv", index=False)
    meta = {
        "convergence_rule": "next halving full propagation relative change <1e-3 and finite-displacement selective relative-error change <1e-3",
        "selected_refinement": int(converged_refinement),
        "selected_dt": float(0.1 / converged_refinement),
        "production_refinement": int(converged_refinement),
        "production_dt": float(0.1 / converged_refinement),
        "production_integrator": "rk4",
        "note": convergence_note,
    }
    write_json(ROOT / "results" / "metrics" / "integration_convergence.json", meta)
    return df, meta


def run_nonlinear_defect_audit(result: SimResult) -> dict[str, Any]:
    defects = np.asarray(result.nonlinear_defect_norms, dtype=float)
    residuals = np.asarray(result.residual_norms, dtype=float)
    activity = np.asarray(result.activity_norms, dtype=float)
    gains = np.asarray(result.opportunity_gains, dtype=float)

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        keep = np.isfinite(a) & np.isfinite(b)
        if np.count_nonzero(keep) < 3 or np.std(a[keep]) <= 0 or np.std(b[keep]) <= 0:
            return float("nan")
        return float(np.corrcoef(a[keep], b[keep])[0, 1])

    audit = {
        "definition": "|Phi_F(x+r)-Phi_F(x)-J Phi_F(x) r| over each decision interval",
        "count": int(defects.size),
        "mean_defect_norm": float(np.mean(defects)) if defects.size else float("nan"),
        "median_defect_norm": float(np.median(defects)) if defects.size else float("nan"),
        "p95_defect_norm": float(np.quantile(defects, 0.95)) if defects.size else float("nan"),
        "max_defect_norm": float(np.max(defects)) if defects.size else float("nan"),
        "mean_residual_norm": float(np.mean(residuals)) if residuals.size else float("nan"),
        "max_residual_norm": float(np.max(residuals)) if residuals.size else float("nan"),
        "defect_vs_residual_norm_corr": corr(defects, residuals),
        "defect_vs_activity_amplitude_corr": corr(defects, activity),
        "defect_vs_realization_gain_corr": corr(defects, gains),
        "decision_interval_s": float(result.decision_interval_s),
        "dt": float(result.dt),
    }
    write_json(ROOT / "results" / "metrics" / "nonlinear_transport_defect.json", audit)
    md = f"""# Nonlinear Transport Audit

V4 uses finite-displacement transport in the formal run. Tangent/JVP transport
is retained only as a diagnostic. The diagnostic
quantity is:

```text
|Phi_F(x+r) - Phi_F(x) - J Phi_F(x) r|
```

over the same decision-interval map used by finite-displacement transport.

- Mean defect norm: `{audit['mean_defect_norm']:.12g}`.
- 95th percentile defect norm: `{audit['p95_defect_norm']:.12g}`.
- Maximum defect norm: `{audit['max_defect_norm']:.12g}`.
- Correlation with |r|: `{audit['defect_vs_residual_norm_corr']:.6g}`.
- Correlation with activity amplitude: `{audit['defect_vs_activity_amplitude_corr']:.6g}`.
- Correlation with realization gain: `{audit['defect_vs_realization_gain_corr']:.6g}`.

This confirms that a numerically correct JVP can still be a first-order
approximation to finite nonlinear displacement.
"""
    (ROOT / "reports" / "nonlinear_transport_audit.md").write_text(md, encoding="utf-8")
    return audit




def alpha_star(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask & np.isfinite(predicted) & np.isfinite(target)
    x = predicted[valid].astype(float)
    y = target[valid].astype(float)
    denom = float(np.dot(x, x))
    alpha = float(np.dot(x, y) / denom) if denom > 0 else float("nan")
    rel_scaled = float(np.linalg.norm(alpha * x - y) / max(np.linalg.norm(y), 1e-12)) if np.isfinite(alpha) else float("nan")
    rel_unscaled = float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-12))
    return {"alpha_star": alpha, "relative_error_after_alpha": rel_scaled, "relative_error_before_alpha": rel_unscaled}


def save_outputs(
    data: FrozenData,
    full: SimResult,
    selective: SimResult,
    local: SimResult,
    validation: np.ndarray,
    test: np.ndarray,
    selected: dict[str, Any],
    final_metrics: dict[str, Any],
) -> None:
    matrices = ROOT / "results" / "matrices"
    trajectories = ROOT / "results" / "trajectories"
    events = ROOT / "results" / "events"
    for path in (matrices, trajectories, events):
        path.mkdir(parents=True, exist_ok=True)
    for name, array in {
        "observed_pair_mask.npy": data.observed_mask,
        "main_high_confidence_mask.npy": data.main_mask,
        "q10_mask.npy": data.q10_mask,
        "all_observed_valid_kernel_mask.npy": data.all_valid_mask,
        "continuous_propagation_matrix.npy": full.propagation,
        "continuous_raw_voltage_response.npy": full.raw_propagation,
        "v4_selective_propagation_matrix.npy": selective.propagation,
        "v4_selective_raw_voltage_response.npy": selective.raw_propagation,
        "local_only_propagation_matrix.npy": local.propagation,
        "local_only_raw_voltage_response.npy": local.raw_propagation,
    }.items():
        np.save(matrices / name, array)
    np.savez_compressed(trajectories / "continuous_states.npz", states=np.asarray(full.states, dtype=np.float32))
    np.savez_compressed(trajectories / "v4_selective_states.npz", states=np.asarray(selective.states, dtype=np.float32))
    np.savez_compressed(trajectories / "local_only_states.npz", states=np.asarray(local.states, dtype=np.float32))
    np.savez_compressed(trajectories / "v4_selective_latent_state.npz", residuals=np.asarray(selective.residuals, dtype=np.float32))
    np.savez_compressed(trajectories / "v4_selective_transported_latent_state.npz", transported=np.asarray(selective.transported, dtype=np.float32))
    np.savez_compressed(trajectories / "v4_selective_fresh_influence.npz", fresh=np.asarray(selective.fresh, dtype=np.float32))
    np.savez_compressed(trajectories / "v4_selective_realized_influence.npz", realized=np.asarray(selective.realized, dtype=np.float32))
    np.savez_compressed(
        events / "v4_selective_realization_events.npz",
        realization_indicator=selective.event_indicator,
        active_blocks=selective.active_blocks,
        event_counts=selective.event_counts,
        inter_realization_intervals_s=selective.inter_event_intervals_s,
    )
    pd.DataFrame(
        {
            "decision_index": np.arange(len(selective.event_counts)),
            "physical_time_s": (np.arange(len(selective.event_counts)) + 1) * selective.decision_interval_s,
            "realized_neuron_blocks": selective.event_counts,
            "active_stimulus_events": np.sum(selective.event_indicator, axis=1),
        }
    ).to_csv(events / "v4_selective_realization_counts_by_time.csv", index=False)
    summary = {
        "selected_working_point": selected,
        "validation_stimulus_count": int(len(validation)),
        "held_out_test_stimulus_count": int(len(test)),
        "final_metrics": final_metrics,
        "output_files": {
            "fully_realized_matrix": rel(matrices / "continuous_propagation_matrix.npy"),
            "v4_selective_matrix": rel(matrices / "v4_selective_propagation_matrix.npy"),
            "local_matrix": rel(matrices / "local_only_propagation_matrix.npy"),
            "events": rel(events / "v4_selective_realization_events.npz"),
        },
    }
    write_json(ROOT / "results" / "metrics" / "summary.json", summary)
    write_json(
        ROOT / "results" / "metrics" / "final_configs.json",
        {
            "selected_working_point": selected,
            "production_integrator": "rk4",
            "config": data.frozen_config["config"],
            "selected_continuous_parameters": data.frozen_config["selected_continuous_parameters"],
            "readout_manifest_file": rel(ROOT / "results" / "metrics" / "readout_manifest.json"),
            "frozen_operating_object": rel(FROZEN_ROOT),
        },
    )
    readout = dict(data.frozen_config["readout"])
    coeff = np.asarray(readout.pop("coefficients"), dtype=float)
    readout_manifest = {
        **readout,
        "coefficient_file": rel(ROOT / "results" / "metrics" / "readout_coefficients.csv"),
        "coefficient_count": int(coeff.size),
        "nonzero_coefficients": int(np.count_nonzero(coeff)),
        "min_coefficient": float(np.min(coeff)),
        "max_coefficient": float(np.max(coeff)),
        "mean_coefficient": float(np.mean(coeff)),
        "feature_dimension_per_response_neuron": 1,
        "response_neuron_count": int(len(data.neuron_ids)),
        "stimulus_index_dimension": int(len(data.neuron_ids)),
        "validation_stimulus_count": int(len(validation)),
        "held_out_test_stimulus_count": int(len(test)),
        "held_out_stimuli_used_for_fit": False,
        "application": "propagation = readout_gain[:, None] * raw_voltage_response",
    }
    write_json(ROOT / "results" / "metrics" / "readout_manifest.json", readout_manifest)
    pd.DataFrame({"neuron_id": data.neuron_ids, "readout_gain": coeff}).to_csv(
        ROOT / "results" / "metrics" / "readout_coefficients.csv",
        index=False,
    )






def run() -> dict[str, Any]:
    start = time.perf_counter()
    for path in [ROOT / "results" / "metrics", ROOT / "reports", ROOT / "logs"]:
        path.mkdir(parents=True, exist_ok=True)
    data = load_frozen_operating_object()
    copy_frozen_operating_data(data)
    frozen_audit = frozen_object_audit(data)
    validation, test = fixed_split(data, seed=188)

    convergence_seed = {
        "decision_interval_s": 0.5,
        "lambda0": float(data.frozen_config["selected_causal_fate_seed"]["lambda0"]),
    }
    convergence_df, convergence_meta = run_integration_convergence(
        data,
        convergence_seed,
        validation,
        data.main_mask,
    )

    base_model = build_model(data, float(convergence_meta.get("production_dt", convergence_meta["selected_dt"])), 0.5)
    full_validation = simulate_full_or_local(base_model, validation, True, "validation_full", "rk4")
    weight = state_weight(base_model, data.frozen_config["config"])

    frontier, selected = run_validation_frontier(data, base_model, validation, full_validation, weight)

    final_dt = float(convergence_meta.get("production_dt", convergence_meta["selected_dt"]))
    final_model = build_model(data, final_dt, float(selected["decision_interval_s"]))
    final_weight = state_weight(final_model, data.frozen_config["config"])
    all_formal = np.sort(np.concatenate([validation, test]))
    full = simulate_full_or_local(final_model, all_formal, True, "fully_realized_final", "rk4")
    local = simulate_full_or_local(final_model, all_formal, False, "local_autonomy_final", "rk4")
    selective = simulate_selective(
        final_model,
        all_formal,
        transport_mode="finite_displacement",
        weight=final_weight,
        lambda0=float(selected["lambda0"]),
        gate_mode="whole",
        block_threshold=0.0,
        name="v4_selective_final",
        record_defect=True,
    )
    defect = run_nonlinear_defect_audit(selective)
    test_mask = column_mask(data.main_mask, test)
    validation_mask = column_mask(data.main_mask, validation)
    final_metrics = {
        "held_out_test_selective_vs_full": regression_metrics(selective.propagation, full.propagation, test_mask),
        "held_out_test_local_vs_full": regression_metrics(local.propagation, full.propagation, test_mask),
        "validation_selective_vs_full": regression_metrics(selective.propagation, full.propagation, validation_mask),
        "all_formal_selective_vs_full": regression_metrics(selective.propagation, full.propagation, column_mask(data.main_mask, all_formal)),
        "all_formal_local_vs_full": regression_metrics(local.propagation, full.propagation, column_mask(data.main_mask, all_formal)),
        "event_active_ratio": selective.active_ratio,
        "events_per_second": selective.events_per_second,
        "mean_inter_realization_interval_s": selective.mean_inter_event_interval_s,
        "closure": {
            "mean": selective.mean_closure_error,
            "max": selective.max_closure_error,
            "relative_mean": selective.relative_closure_error,
        },
        "alpha_star": {
            "selective": alpha_star(selective.propagation, full.propagation, test_mask),
            "local": alpha_star(local.propagation, full.propagation, test_mask),
        },
    }
    write_json(ROOT / "results" / "metrics" / "held_out_test_metrics.json", final_metrics)
    save_outputs(data, full, selective, local, validation, test, selected, final_metrics)
    checks = {
        "runtime_s": float(time.perf_counter() - start),
        "frozen_operating_object": rel(FROZEN_ROOT),
        "output_root": str(ROOT),
    }
    write_json(ROOT / "results" / "metrics" / "run_metadata.json", checks)
    return {
        "frozen_audit": frozen_audit,
        "defect": defect,
        "convergence": convergence_meta,
        "selected": selected,
        "final_metrics": final_metrics,
        "runtime_s": checks["runtime_s"],
    }


def main() -> None:
    payload = run()
    print(json.dumps(payload, indent=2, default=json_default))


if __name__ == "__main__":
    main()
