"""Connectome-constrained graded-potential dynamics and sparse residual transport."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from .data import HeadData
from .metrics import matrix_metrics


jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ModelParams:
    g_chem: float
    g_gap: float
    g_leak: float
    stimulus_amplitude: float
    v_half: float
    k_slope: float
    capacitance: float
    resting_potential: float
    synapse_rise_rate: float
    synapse_decay_rate: float


@dataclass(frozen=True)
class DynamicsModel:
    n: int
    chemical: np.ndarray
    gap: np.ndarray
    reversal: np.ndarray
    params: ModelParams
    dt: float
    steps: int
    stimulus_steps: int
    response_start_step: int
    observation_window_steps: int
    correction_decision_steps: int
    readout_gain: np.ndarray
    calibration_table: pd.DataFrame


@dataclass(frozen=True)
class JaxOps:
    full_step: object
    local_step: object
    full_jvp: object


@dataclass(frozen=True)
class ForwardResult:
    states: np.ndarray
    propagation: np.ndarray
    raw_propagation: np.ndarray
    residuals: np.ndarray | None = None
    transported_residuals: np.ndarray | None = None
    fresh_corrections: np.ndarray | None = None
    released_corrections: np.ndarray | None = None
    event_indicator: np.ndarray | None = None
    active_blocks: np.ndarray | None = None
    active_ratio: float | None = None
    active_block_ratio: float | None = None
    active_block_ratio_conditional: float | None = None
    mean_active_neurons_per_event: float | None = None
    median_active_neurons_per_event: float | None = None
    active_neuron_fraction_per_event: float | None = None
    total_active_neuron_time_blocks: int | None = None
    event_counts: np.ndarray | None = None
    gate_mode: str | None = None
    lambda0: float | None = None
    block_threshold: float | None = None
    events_per_second: float | None = None
    mean_physical_inter_event_interval_s: float | None = None
    inter_event_intervals_s: np.ndarray | None = None


def normalize_connectome(chemical: np.ndarray, gap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chem = np.log1p(np.maximum(chemical, 0.0))
    gj = np.log1p(np.maximum(gap, 0.0))
    np.fill_diagonal(chem, 0.0)
    np.fill_diagonal(gj, 0.0)
    chem = chem / np.maximum(np.sum(chem, axis=1, keepdims=True), 1e-12)
    gj = gj / np.maximum(np.sum(gj, axis=1, keepdims=True), 1e-12)
    return np.nan_to_num(chem), np.nan_to_num(gj)


def params_from_config(config: dict[str, object], row: dict[str, float]) -> ModelParams:
    fixed = dict(config["biophysical_fixed"])
    return ModelParams(
        g_chem=float(row["g_chem"]),
        g_gap=float(row["g_gap"]),
        g_leak=float(row["g_leak"]),
        stimulus_amplitude=float(row["stimulus_amplitude"]),
        v_half=float(row["v_half"]),
        k_slope=float(row["k_slope"]),
        capacitance=float(fixed["capacitance"]),
        resting_potential=float(fixed["resting_potential_mV"]),
        synapse_rise_rate=float(fixed["synapse_rise_rate"]),
        synapse_decay_rate=float(fixed["synapse_decay_rate"]),
    )


def steady_synapse_gate(params: ModelParams) -> float:
    phi = 1.0 / (1.0 + np.exp(-(params.resting_potential - params.v_half) / params.k_slope))
    return float((params.synapse_rise_rate * phi) / (params.synapse_rise_rate * phi + params.synapse_decay_rate))


def initial_state(n: int, params: ModelParams) -> np.ndarray:
    v0 = np.full(n, params.resting_potential, dtype=float)
    s0 = np.full(n, steady_synapse_gate(params), dtype=float)
    return np.concatenate([v0, s0])


def stimulus_vector(n: int, stimulus_index: int, step: int, params: ModelParams, stimulus_steps: int) -> np.ndarray:
    u = np.zeros(n, dtype=float)
    if step < stimulus_steps:
        u[stimulus_index] = params.stimulus_amplitude
    return u


def _split(x: jnp.ndarray, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    return x[:n], x[n:]


def _rhs(
    x: jnp.ndarray,
    u: jnp.ndarray,
    chemical: jnp.ndarray,
    gap: jnp.ndarray,
    reversal: jnp.ndarray,
    p: ModelParams,
    full: bool,
) -> jnp.ndarray:
    n = chemical.shape[0]
    v, s = _split(x, n)
    phi = 1.0 / (1.0 + jnp.exp(-(v - p.v_half) / p.k_slope))
    ds = p.synapse_rise_rate * phi * (1.0 - s) - p.synapse_decay_rate * s
    gap_current = jnp.zeros_like(v)
    chem_current = jnp.zeros_like(v)
    if full:
        gap_current = p.g_gap * ((jnp.sum(gap, axis=1) * v) - gap @ v)
        weighted_s = chemical @ s
        weighted_reversal_s = (chemical * reversal) @ s
        chem_current = p.g_chem * (v * weighted_s - weighted_reversal_s)
    dv = (-p.g_leak * (v - p.resting_potential) - gap_current - chem_current + u) / p.capacitance
    return jnp.concatenate([dv, ds])


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


def step_state(
    x: jnp.ndarray,
    u: jnp.ndarray,
    chemical: jnp.ndarray,
    gap: jnp.ndarray,
    reversal: jnp.ndarray,
    p: ModelParams,
    dt: float,
    full: bool,
) -> jnp.ndarray:
    rhs = _rhs(x, u, chemical, gap, reversal, p, full)
    y = x + dt * rhs
    n = chemical.shape[0]
    v, s = _split(y, n)
    s = jnp.clip(s, 0.0, 1.0)
    return jnp.concatenate([v, s])


def step_state_batch(
    x: jnp.ndarray,
    u: jnp.ndarray,
    chemical: jnp.ndarray,
    gap: jnp.ndarray,
    reversal: jnp.ndarray,
    p: ModelParams,
    dt: float,
    full: bool,
) -> jnp.ndarray:
    rhs = _rhs_batch(x, u, chemical, gap, reversal, p, full)
    y = x + dt * rhs
    n = chemical.shape[0]
    v = y[:, :n]
    s = jnp.clip(y[:, n:], 0.0, 1.0)
    return jnp.concatenate([v, s], axis=1)


def make_jax_ops(model: DynamicsModel, clip_norm: float = 50.0) -> JaxOps:
    chemical = jnp.asarray(model.chemical)
    gap = jnp.asarray(model.gap)
    reversal = jnp.asarray(model.reversal)

    @jax.jit
    def full_step(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return step_state_batch(x, u, chemical, gap, reversal, model.params, model.dt, True)

    @jax.jit
    def local_step(x: jnp.ndarray, u: jnp.ndarray) -> jnp.ndarray:
        return step_state_batch(x, u, chemical, gap, reversal, model.params, model.dt, False)

    @jax.jit
    def full_jvp(x: jnp.ndarray, u: jnp.ndarray, r: jnp.ndarray) -> jnp.ndarray:
        def one_step(z: jnp.ndarray) -> jnp.ndarray:
            return step_state_batch(z, u, chemical, gap, reversal, model.params, model.dt, True)

        _, tangent = jax.jvp(one_step, (x,), (r,))
        norm = jnp.linalg.norm(tangent)
        scale = jnp.minimum(1.0, clip_norm / jnp.maximum(norm, 1e-12))
        return tangent * scale

    return JaxOps(full_step=full_step, local_step=local_step, full_jvp=full_jvp)


def stimulus_batch(model: DynamicsModel, step: int) -> np.ndarray:
    u = np.zeros((model.n, model.n), dtype=float)
    if step < model.stimulus_steps:
        np.fill_diagonal(u, model.params.stimulus_amplitude)
    return u


def _simulate_all(model: DynamicsModel, full: bool) -> np.ndarray:
    ops = make_jax_ops(model)
    x0 = initial_state(model.n, model.params)
    x = np.tile(x0[None, :], (model.n, 1))
    states = [x.copy()]
    step_fn = ops.full_step if full else ops.local_step
    for step in range(model.steps):
        x = np.asarray(step_fn(jnp.asarray(x), jnp.asarray(stimulus_batch(model, step))))
        states.append(x.copy())
    return np.stack(states, axis=0)


def readout_feature(states: np.ndarray, model: DynamicsModel) -> np.ndarray:
    end = min(states.shape[0], model.response_start_step + model.observation_window_steps + 1)
    v = states[model.response_start_step : end, :, : model.n]
    dv = v - model.params.resting_potential
    if v.shape[0] <= 1:
        response = dv[0]
    else:
        response = np.trapezoid(dv, dx=model.dt, axis=0) / max((v.shape[0] - 1) * model.dt, 1e-12)
    return response.T


def apply_readout(raw_response: np.ndarray, readout_gain: np.ndarray) -> np.ndarray:
    return readout_gain[:, None] * raw_response


def simulate_continuous(model: DynamicsModel) -> ForwardResult:
    states = _simulate_all(model, full=True)
    raw = readout_feature(states, model)
    return ForwardResult(states=states, propagation=apply_readout(raw, model.readout_gain), raw_propagation=raw)


def simulate_local(model: DynamicsModel) -> ForwardResult:
    states = _simulate_all(model, full=False)
    raw = readout_feature(states, model)
    return ForwardResult(states=states, propagation=apply_readout(raw, model.readout_gain), raw_propagation=raw)


def fit_node_readout_gain(
    raw_response: np.ndarray,
    empirical: np.ndarray,
    mask: np.ndarray,
    ridge: float,
) -> np.ndarray:
    gains = np.zeros(raw_response.shape[0], dtype=float)
    for i in range(raw_response.shape[0]):
        row_mask = mask[i] & np.isfinite(raw_response[i]) & np.isfinite(empirical[i])
        if not np.any(row_mask):
            continue
        x = raw_response[i, row_mask]
        y = empirical[i, row_mask]
        denom = float(np.sum(x * x) + ridge * max(1, x.size) * (np.var(x) + 1e-12))
        if denom <= 0.0:
            continue
        gains[i] = max(0.0, float(np.sum(x * y) / denom))
    return gains


def _calibration_score(metrics: dict[str, float], config: dict[str, object]) -> float:
    weights = dict(config["calibration"]["objective_weights"])
    rel = metrics["relative_frobenius_error"]
    pearson = metrics["pearson"] if np.isfinite(metrics["pearson"]) else 0.0
    sign = metrics["sign_agreement"] if np.isfinite(metrics["sign_agreement"]) else 0.0
    mode = metrics["leading_mode_similarity"] if np.isfinite(metrics["leading_mode_similarity"]) else 0.0
    return float(
        weights["relative_error"] * rel
        + weights["pearson_loss"] * (1.0 - pearson)
        + weights["sign_loss"] * (1.0 - sign)
        + weights["leading_mode_loss"] * (1.0 - mode)
    )


def calibrate_continuous_model(data: HeadData, config: dict[str, object]) -> DynamicsModel:
    chem, gap = normalize_connectome(data.chemical, data.gap)
    grid = dict(config["continuous_grid"])
    rows: list[dict[str, float]] = []
    best: tuple[float, dict[str, float], np.ndarray] | None = None
    keys = ["g_chem", "g_gap", "g_leak", "stimulus_amplitude", "v_half", "k_slope"]
    mask = data.main_mask
    for values in product(*(grid[key] for key in keys)):
        row = {key: float(value) for key, value in zip(keys, values)}
        params = params_from_config(config, row)
        model = DynamicsModel(
            n=len(data.neuron_ids),
            chemical=chem,
            gap=gap,
            reversal=data.chemical_reversal,
            params=params,
            dt=float(config["simulation"]["dt"]),
            steps=int(config["simulation"]["steps"]),
            stimulus_steps=int(config["simulation"]["stimulus_steps"]),
            response_start_step=int(config["simulation"]["response_start_step"]),
            observation_window_steps=int(config["simulation"]["observation_window_steps"]),
            correction_decision_steps=int(config["simulation"]["correction_decision_steps"]),
            readout_gain=np.ones(len(data.neuron_ids), dtype=float),
            calibration_table=pd.DataFrame(),
        )
        continuous = simulate_continuous(model)
        gain = fit_node_readout_gain(
            continuous.raw_propagation,
            data.empirical,
            mask,
            float(config["calibration"]["readout_ridge"]),
        )
        pred = apply_readout(continuous.raw_propagation, gain)
        metrics = matrix_metrics(pred, data.empirical, mask)
        score = _calibration_score(metrics, config)
        rows.append(
            {
                **row,
                **{f"continuous_vs_empirical_{key}": value for key, value in metrics.items()},
                "node_local_readout_parameters": int(np.count_nonzero(gain)),
                "fitted_parameter_count": int(np.count_nonzero(gain) + len(keys)),
                "calibration_score": score,
            }
        )
        if best is None or score < best[0]:
            best = (score, row, gain)
    if best is None:
        raise RuntimeError("Continuous model calibration grid produced no candidates.")
    table = pd.DataFrame(rows).sort_values("calibration_score", ascending=True).reset_index(drop=True)
    params = params_from_config(config, best[1])
    return DynamicsModel(
        n=len(data.neuron_ids),
        chemical=chem,
        gap=gap,
        reversal=data.chemical_reversal,
        params=params,
        dt=float(config["simulation"]["dt"]),
        steps=int(config["simulation"]["steps"]),
        stimulus_steps=int(config["simulation"]["stimulus_steps"]),
        response_start_step=int(config["simulation"]["response_start_step"]),
        observation_window_steps=int(config["simulation"]["observation_window_steps"]),
        correction_decision_steps=int(config["simulation"]["correction_decision_steps"]),
        readout_gain=best[2],
        calibration_table=table,
    )


def full_step_np(model: DynamicsModel, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.asarray(
        step_state(
            jnp.asarray(x),
            jnp.asarray(u),
            jnp.asarray(model.chemical),
            jnp.asarray(model.gap),
            jnp.asarray(model.reversal),
            model.params,
            model.dt,
            True,
        )
    )


def local_step_np(model: DynamicsModel, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.asarray(
        step_state(
            jnp.asarray(x),
            jnp.asarray(u),
            jnp.asarray(model.chemical),
            jnp.asarray(model.gap),
            jnp.asarray(model.reversal),
            model.params,
            model.dt,
            False,
        )
    )


def full_jvp_np(model: DynamicsModel, x: np.ndarray, u: np.ndarray, r: np.ndarray) -> np.ndarray:
    chemical = jnp.asarray(model.chemical)
    gap = jnp.asarray(model.gap)
    reversal = jnp.asarray(model.reversal)

    def one_step(z: jnp.ndarray) -> jnp.ndarray:
        return step_state(z, jnp.asarray(u), chemical, gap, reversal, model.params, model.dt, True)

    _, tangent = jax.jvp(one_step, (jnp.asarray(x),), (jnp.asarray(r),))
    out = np.asarray(tangent)
    norm = float(np.linalg.norm(out))
    if norm > 0.0 and norm > 50.0:
        out = out * (50.0 / norm)
    return out


def finite_difference_jvp_check(model: DynamicsModel, seed: int = 188) -> dict[str, float | bool]:
    rng = np.random.default_rng(seed)
    x = initial_state(model.n, model.params)
    x += rng.normal(0.0, 0.01, size=x.shape)
    u = stimulus_vector(model.n, int(rng.integers(0, model.n)), 0, model.params, model.stimulus_steps)
    r = rng.normal(0.0, 0.01, size=x.shape)
    jvp = full_jvp_np(model, x, u, r)
    eps = 1e-5
    fd = (full_step_np(model, x + eps * r, u) - full_step_np(model, x - eps * r, u)) / (2.0 * eps)
    rel = float(np.linalg.norm(jvp - fd) / max(np.linalg.norm(fd), 1e-12))
    return {
        "full_jacobian_jvp_relative_difference": rel,
        "full_jacobian_jvp_check_passed": bool(rel < 1e-5),
    }


def weighted_block_norms(h: np.ndarray, model: DynamicsModel, weight: np.ndarray) -> np.ndarray:
    hv = h[..., : model.n]
    hs = h[..., model.n :]
    wv = weight[: model.n]
    ws = weight[model.n :]
    return np.sqrt((wv * hv) ** 2 + (ws * hs) ** 2)


def gate_neuron_blocks(
    h: np.ndarray,
    model: DynamicsModel,
    weight: np.ndarray,
    lambda0: float,
    block_threshold: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, bool, float]:
    q = np.zeros_like(h)
    if mode == "whole":
        weighted_norm = float(np.linalg.norm(weight * h))
        gain = 0.5 * weighted_norm * weighted_norm
        active = gain > lambda0
        if active:
            q = h.copy()
        return q, np.full(model.n, active, dtype=bool), bool(active), gain

    norms = weighted_block_norms(h, model, weight)
    shrink = np.maximum(1.0 - block_threshold / np.maximum(norms, 1e-12), 0.0)
    gains = 0.5 * np.maximum(norms - block_threshold, 0.0) ** 2
    event_gain = float(np.sum(gains))
    event_active = event_gain > lambda0
    block_active = event_active & (shrink > 0.0)
    if event_active:
        q[: model.n] = shrink * h[: model.n]
        q[model.n :] = shrink * h[model.n :]
    return q, block_active, bool(event_active), event_gain


def gate_neuron_blocks_batch(
    h: np.ndarray,
    model: DynamicsModel,
    weight: np.ndarray,
    lambda0: float,
    block_threshold: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.zeros_like(h)
    if mode == "whole":
        weighted_norm = np.linalg.norm(weight[None, :] * h, axis=1)
        event_gain = 0.5 * weighted_norm * weighted_norm
        event_active = event_gain > lambda0
        q[event_active] = h[event_active]
        blocks = np.repeat(event_active[:, None], model.n, axis=1)
        return q, blocks, event_active

    norms = weighted_block_norms(h, model, weight)
    shrink = np.maximum(1.0 - block_threshold / np.maximum(norms, 1e-12), 0.0)
    gains = 0.5 * np.maximum(norms - block_threshold, 0.0) ** 2
    event_active = np.sum(gains, axis=1) > lambda0
    blocks = (shrink > 0.0) & event_active[:, None]
    q[:, : model.n] = shrink * h[:, : model.n] * event_active[:, None]
    q[:, model.n :] = shrink * h[:, model.n :] * event_active[:, None]
    return q, blocks, event_active


def simulate_sparse(
    model: DynamicsModel,
    method: str,
    weight: np.ndarray,
    lambda0: float,
    block_threshold: float,
    gate_mode: str,
    ops: JaxOps | None = None,
) -> ForwardResult:
    if method not in {"dtpr", "static"}:
        raise ValueError(f"Unknown sparse method: {method}")
    if ops is None:
        ops = make_jax_ops(model)
    x0 = initial_state(model.n, model.params)
    x = np.tile(x0[None, :], (model.n, 1))
    r = np.zeros_like(x)
    states = [x.copy()]
    residuals = [r.copy()]
    transported_rows = []
    fresh_rows = []
    released_rows = []
    event_rows = []
    block_rows = []
    for step in range(model.steps):
        u = stimulus_batch(model, step)
        local_next = np.asarray(ops.local_step(jnp.asarray(x), jnp.asarray(u)))
        full_next = np.asarray(ops.full_step(jnp.asarray(x), jnp.asarray(u)))
        fresh = full_next - local_next
        if method == "dtpr":
            transported = np.asarray(ops.full_jvp(jnp.asarray(x), jnp.asarray(u), jnp.asarray(r)))
        else:
            transported = r.copy()
        h = transported + fresh
        decision_step = ((step + 1) % model.correction_decision_steps) == 0
        if decision_step:
            q, block_active, event_active = gate_neuron_blocks_batch(
                h,
                model,
                weight,
                lambda0=lambda0,
                block_threshold=block_threshold,
                mode=gate_mode,
            )
        else:
            q = np.zeros_like(h)
            block_active = np.zeros((model.n, model.n), dtype=bool)
            event_active = np.zeros(model.n, dtype=bool)
        x = local_next + q
        r = h - q
        states.append(x.copy())
        residuals.append(r.copy())
        transported_rows.append(transported.copy())
        fresh_rows.append(fresh.copy())
        released_rows.append(q.copy())
        event_rows.append(event_active.copy())
        block_rows.append(block_active.copy())

    states_arr = np.asarray(states)
    raw = readout_feature(states_arr, model)
    events = np.asarray(event_rows, dtype=bool)
    blocks = np.asarray(block_rows, dtype=bool)
    event_counts = np.sum(blocks, axis=(1, 2))
    event_indicator = events
    active_event_ratio = float(np.count_nonzero(event_indicator) / event_indicator.size)
    active_block_ratio = float(np.count_nonzero(blocks) / blocks.size)
    active_event_block_counts = event_counts[event_indicator.any(axis=1)]
    # The event-level fixed cost is shared within each stimulus and time. For
    # reporting, count active stimulus-time events and neuron blocks separately.
    per_event_counts = blocks.reshape(model.steps * model.n, model.n).sum(axis=1)
    active_per_event_counts = per_event_counts[per_event_counts > 0]
    if active_per_event_counts.size:
        mean_blocks = float(np.mean(active_per_event_counts))
        median_blocks = float(np.median(active_per_event_counts))
        conditional = float(np.sum(active_per_event_counts) / (active_per_event_counts.size * model.n))
    else:
        mean_blocks = 0.0
        median_blocks = 0.0
        conditional = 0.0
    active_times = np.argwhere(events)
    intervals: list[float] = []
    for stim in range(model.n):
        times = np.flatnonzero(events[:, stim])
        if len(times) > 1:
            intervals.extend(np.diff(times).astype(float).tolist())
    intervals_s = np.asarray(intervals, dtype=float) * model.dt
    duration_s = model.steps * model.dt
    events_per_second = float(np.count_nonzero(events) / max(model.n * duration_s, 1e-12))
    mean_interval = float(np.mean(intervals_s)) if intervals_s.size else float("inf")
    return ForwardResult(
        states=states_arr,
        propagation=apply_readout(raw, model.readout_gain),
        raw_propagation=raw,
        residuals=np.asarray(residuals),
        transported_residuals=np.asarray(transported_rows),
        fresh_corrections=np.asarray(fresh_rows),
        released_corrections=np.asarray(released_rows),
        event_indicator=event_indicator,
        active_blocks=blocks,
        active_ratio=active_event_ratio,
        active_block_ratio=active_block_ratio,
        active_block_ratio_conditional=conditional,
        mean_active_neurons_per_event=mean_blocks,
        median_active_neurons_per_event=median_blocks,
        active_neuron_fraction_per_event=conditional,
        total_active_neuron_time_blocks=int(np.count_nonzero(blocks)),
        event_counts=event_counts,
        gate_mode=gate_mode,
        lambda0=float(lambda0),
        block_threshold=float(block_threshold),
        events_per_second=events_per_second,
        mean_physical_inter_event_interval_s=mean_interval,
        inter_event_intervals_s=intervals_s,
    )


def simulate_periodic(
    model: DynamicsModel,
    method: str,
    target_active_ratio: float,
    gate_mode: str,
) -> ForwardResult:
    decision_steps = np.arange(model.correction_decision_steps - 1, model.steps, model.correction_decision_steps)
    decision_ratio = len(decision_steps) / max(model.steps, 1)
    period_decisions = max(1, int(round(decision_ratio / max(target_active_ratio, 1e-12))))
    ops = make_jax_ops(model)
    x0 = initial_state(model.n, model.params)
    x = np.tile(x0[None, :], (model.n, 1))
    r = np.zeros_like(x)
    states = [x.copy()]
    residuals = [r.copy()]
    transported_rows = []
    fresh_rows = []
    released_rows = []
    event_rows = []
    block_rows = []
    decision_count = 0
    for step in range(model.steps):
        u = stimulus_batch(model, step)
        local_next = np.asarray(ops.local_step(jnp.asarray(x), jnp.asarray(u)))
        full_next = np.asarray(ops.full_step(jnp.asarray(x), jnp.asarray(u)))
        fresh = full_next - local_next
        if method == "dtpr":
            transported = np.asarray(ops.full_jvp(jnp.asarray(x), jnp.asarray(u), jnp.asarray(r)))
        else:
            transported = r.copy()
        h = transported + fresh
        decision_step = ((step + 1) % model.correction_decision_steps) == 0
        release = False
        if decision_step:
            release = (decision_count % period_decisions) == 0
            decision_count += 1
        if release:
            q = h.copy()
            event_active = np.ones(model.n, dtype=bool)
            block_active = np.ones((model.n, model.n), dtype=bool) if gate_mode == "whole" else np.ones((model.n, model.n), dtype=bool)
        else:
            q = np.zeros_like(h)
            event_active = np.zeros(model.n, dtype=bool)
            block_active = np.zeros((model.n, model.n), dtype=bool)
        x = local_next + q
        r = h - q
        states.append(x.copy())
        residuals.append(r.copy())
        transported_rows.append(transported.copy())
        fresh_rows.append(fresh.copy())
        released_rows.append(q.copy())
        event_rows.append(event_active.copy())
        block_rows.append(block_active.copy())
    states_arr = np.asarray(states)
    raw = readout_feature(states_arr, model)
    events = np.asarray(event_rows, dtype=bool)
    blocks = np.asarray(block_rows, dtype=bool)
    event_counts = np.sum(blocks, axis=(1, 2))
    per_event_counts = blocks.reshape(model.steps * model.n, model.n).sum(axis=1)
    active_per_event_counts = per_event_counts[per_event_counts > 0]
    conditional = float(np.sum(active_per_event_counts) / (active_per_event_counts.size * model.n)) if active_per_event_counts.size else 0.0
    intervals: list[float] = []
    for stim in range(model.n):
        times = np.flatnonzero(events[:, stim])
        if len(times) > 1:
            intervals.extend(np.diff(times).astype(float).tolist())
    intervals_s = np.asarray(intervals, dtype=float) * model.dt
    duration_s = model.steps * model.dt
    return ForwardResult(
        states=states_arr,
        propagation=apply_readout(raw, model.readout_gain),
        raw_propagation=raw,
        residuals=np.asarray(residuals),
        transported_residuals=np.asarray(transported_rows),
        fresh_corrections=np.asarray(fresh_rows),
        released_corrections=np.asarray(released_rows),
        event_indicator=events,
        active_blocks=blocks,
        active_ratio=float(np.count_nonzero(events) / events.size),
        active_block_ratio=float(np.count_nonzero(blocks) / blocks.size),
        active_block_ratio_conditional=conditional,
        mean_active_neurons_per_event=float(np.mean(active_per_event_counts)) if active_per_event_counts.size else 0.0,
        median_active_neurons_per_event=float(np.median(active_per_event_counts)) if active_per_event_counts.size else 0.0,
        active_neuron_fraction_per_event=conditional,
        total_active_neuron_time_blocks=int(np.count_nonzero(blocks)),
        event_counts=event_counts,
        gate_mode=gate_mode,
        lambda0=float("nan"),
        block_threshold=0.0,
        events_per_second=float(np.count_nonzero(events) / max(model.n * duration_s, 1e-12)),
        mean_physical_inter_event_interval_s=float(np.mean(intervals_s)) if intervals_s.size else float("inf"),
        inter_event_intervals_s=intervals_s,
    )


def estimate_gate_thresholds(
    model: DynamicsModel,
    weight: np.ndarray,
    config: dict[str, object],
    method: str = "dtpr",
) -> dict[str, np.ndarray]:
    if method not in {"dtpr", "static"}:
        raise ValueError(f"Unknown threshold-estimation method: {method}")
    ops = make_jax_ops(model)
    norms: list[float] = []
    whole_gains: list[float] = []
    block_gains: list[float] = []
    x0 = initial_state(model.n, model.params)
    x = np.tile(x0[None, :], (model.n, 1))
    r = np.zeros_like(x)
    for step in range(model.steps):
        u = stimulus_batch(model, step)
        full_next = np.asarray(ops.full_step(jnp.asarray(x), jnp.asarray(u)))
        local_next = np.asarray(ops.local_step(jnp.asarray(x), jnp.asarray(u)))
        fresh = full_next - local_next
        if method == "dtpr":
            transported = np.asarray(ops.full_jvp(jnp.asarray(x), jnp.asarray(u), jnp.asarray(r)))
        else:
            transported = r.copy()
        h = transported + fresh
        if ((step + 1) % model.correction_decision_steps) == 0:
            block_norm = weighted_block_norms(h, model, weight)
            norms.extend(block_norm.reshape(-1).tolist())
            block_gains.extend(np.sum(0.5 * block_norm**2, axis=1).tolist())
            whole = np.linalg.norm(weight[None, :] * h, axis=1)
            whole_gains.extend((0.5 * whole * whole).tolist())
        x = local_next
        r = h
    norms_arr = np.asarray(norms, dtype=float)
    block_gain_arr = np.asarray(block_gains, dtype=float)
    whole_gain_arr = np.asarray(whole_gains, dtype=float)
    dtpr_cfg = dict(config["dtpr"])
    def quantiles(values: np.ndarray, qs: list[float]) -> np.ndarray:
        valid = values[np.isfinite(values) & (values > 0.0)]
        if valid.size == 0:
            return np.asarray([0.0], dtype=float)
        return np.unique(np.quantile(valid, qs))

    return {
        "block_thresholds": quantiles(norms_arr, dtpr_cfg["block_threshold_quantiles"]),
        "event_thresholds": quantiles(block_gain_arr, dtpr_cfg["event_threshold_quantiles"]),
        "whole_event_thresholds": quantiles(whole_gain_arr, dtpr_cfg["whole_event_threshold_quantiles"]),
    }
