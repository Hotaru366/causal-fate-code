"""Kernel audit and empirical window selection for C. elegans V4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from .data import AWC_TO_FUNATLAS, _head_ids, _indices, ensure_official_repo


@dataclass(frozen=True)
class WindowTarget:
    time: np.ndarray
    selected_window_s: float
    empirical_raw: np.ndarray
    empirical_robust: np.ndarray
    empirical_raw_30: np.ndarray
    empirical_robust_30: np.ndarray
    valid_kernel_mask: np.ndarray
    high_confidence_mask: np.ndarray
    q10_mask: np.ndarray
    all_observed_valid_mask: np.ndarray
    transform_scale: float
    transform_scale_30: float
    audit: dict[str, object]
    q05_pair_stats: list[dict[str, object]]


def _eval_kernel_array(arr: np.ndarray, time: np.ndarray) -> np.ndarray | None:
    if len(arr) == 0 or len(arr) % 4 != 0:
        return None
    terms = np.asarray(arr, dtype=float).reshape(len(arr) // 4, 4)
    y = np.zeros_like(time, dtype=float)
    for g, factor, power_t, _branch in terms:
        mult = 1.0 if power_t == 0.0 else np.power(time, power_t)
        y += factor * mult * np.exp(-g * time)
    if not np.all(np.isfinite(y)):
        return None
    return y


def _window_stat(time: np.ndarray, y: np.ndarray, window_s: float, operator: str) -> float:
    keep = time <= window_s + 1e-12
    if np.count_nonzero(keep) < 2:
        return float("nan")
    if operator == "signed_mean":
        return float(np.trapezoid(y[keep], time[keep]) / max(window_s, 1e-12))
    if operator == "signed_integral":
        return float(np.trapezoid(y[keep], time[keep]))
    if operator == "signed_extremum":
        yy = y[keep]
        return float(yy[np.argmax(np.abs(yy))])
    raise ValueError(f"Unknown observation operator: {operator}")


def _energy(time: np.ndarray, y: np.ndarray, window_s: float) -> float:
    keep = time <= window_s + 1e-12
    if np.count_nonzero(keep) < 2:
        return 0.0
    return float(np.trapezoid(np.abs(y[keep]), time[keep]))


def _robust_transform(raw: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    vals = raw[mask & np.isfinite(raw)]
    scale = float(np.nanpercentile(np.abs(vals), 95)) if vals.size else 1.0
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    out = np.full_like(raw, np.nan, dtype=float)
    valid = np.isfinite(raw)
    out[valid] = np.tanh(raw[valid] / scale)
    return out, scale


def build_kernel_window_target(root: Path, config: dict[str, object]) -> WindowTarget:
    repo = ensure_official_repo(root, str(config["official_repo_url"]), str(config["official_repo_commit"]))
    data_dir = repo / "wormneuroatlas" / "data"
    nominal_ids = _head_ids(data_dir)
    funatlas_ids = np.asarray([AWC_TO_FUNATLAS.get(str(name), str(name)) for name in nominal_ids])
    kernel_cfg = dict(config["kernel_window"])
    dt = float(kernel_cfg["time_step_s"])
    max_time = float(kernel_cfg["max_time_s"])
    time = np.arange(0.0, max_time + 0.5 * dt, dt)
    candidates = [float(x) for x in kernel_cfg["candidate_windows_s"]]
    operator = str(kernel_cfg["observation_operator"])
    max_abs = float(kernel_cfg["max_abs_kernel_value"])

    n = len(nominal_ids)
    valid_kernel = np.zeros((n, n), dtype=bool)
    abnormal_kernel = np.zeros((n, n), dtype=bool)
    raw_by_window = {window: np.full((n, n), np.nan, dtype=float) for window in candidates}
    energy_by_window = {window: np.zeros((n, n), dtype=float) for window in candidates}
    peak_time = np.full((n, n), np.nan, dtype=float)
    signed_peak_time = np.full((n, n), np.nan, dtype=float)
    onset_time = np.full((n, n), np.nan, dtype=float)
    peak_value = np.full((n, n), np.nan, dtype=float)

    with h5py.File(data_dir / "funatlas.h5", "r") as fun_h5:
        source_fun_ids = np.asarray([name.decode("utf-8") for name in fun_h5["neuron_ids"][:]])
        fun_idx = _indices(source_fun_ids, funatlas_ids)
        strain = str(config["strain"])
        empirical_q = fun_h5[strain]["q"][:][fun_idx][:, fun_idx].astype(float)
        occurrence = fun_h5[strain]["occ1"][:][fun_idx][:, fun_idx].astype(int)
        official_dff = fun_h5[strain]["dFF"][:][fun_idx][:, fun_idx].astype(float)
        kernels = fun_h5[strain]["kernels"]
        kernel_key_order = fun_h5.attrs["kernels_keys"].decode("utf-8")
        funatlas_time_compiled = fun_h5.attrs["time_compiled"].decode("utf-8")
        for ai, src_i in enumerate(fun_idx):
            for aj, src_j in enumerate(fun_idx):
                if ai == aj:
                    continue
                arr = kernels[src_i, src_j]
                y = _eval_kernel_array(arr, time)
                if y is None:
                    continue
                if np.nanmax(np.abs(y)) > max_abs:
                    abnormal_kernel[ai, aj] = True
                    continue
                valid_kernel[ai, aj] = True
                abs_i = int(np.nanargmax(np.abs(y)))
                sign_i = int(np.nanargmax(y)) if abs(float(np.nanmax(y))) >= abs(float(np.nanmin(y))) else int(np.nanargmin(y))
                peak_time[ai, aj] = float(time[abs_i])
                signed_peak_time[ai, aj] = float(time[sign_i])
                peak_value[ai, aj] = float(y[abs_i])
                threshold = 0.1 * max(abs(float(y[abs_i])), 1e-12)
                above = np.flatnonzero(np.abs(y) >= threshold)
                onset_time[ai, aj] = float(time[above[0]]) if above.size else float("nan")
                for window in candidates:
                    raw_by_window[window][ai, aj] = _window_stat(time, y, window, operator)
                    energy_by_window[window][ai, aj] = _energy(time, y, window)

    observed = np.isfinite(official_dff) & ~np.eye(n, dtype=bool)
    q05 = observed & valid_kernel & np.isfinite(empirical_q) & (empirical_q < 0.05)
    q10 = observed & valid_kernel & np.isfinite(empirical_q) & (empirical_q < 0.10)
    all_valid = observed & valid_kernel
    e30 = energy_by_window[30.0]
    total_energy_30 = float(np.sum(e30[q05]))
    candidate_rows = []
    for window in candidates:
        row = {
            "window_s": window,
            "available_pair_count": int(np.count_nonzero(q05 & np.isfinite(raw_by_window[window]))),
            "peak_time_coverage": float(np.mean(peak_time[q05] <= window)) if np.any(q05) else float("nan"),
            "energy_fraction_vs_30s": float(np.sum(energy_by_window[window][q05]) / max(total_energy_30, 1e-12)),
        }
        row["passes_80_80_rule"] = bool(row["peak_time_coverage"] >= 0.8 and row["energy_fraction_vs_30s"] >= 0.8)
        candidate_rows.append(row)
    passing = [row for row in candidate_rows if row["passes_80_80_rule"]]
    if passing:
        selected = min(passing, key=lambda row: row["window_s"])
        selection_rule = "shortest candidate with >=80% q<0.05 absolute peak coverage and >=80% 30s absolute-response energy"
    else:
        raise ValueError("No candidate window satisfies both fixed 80% criteria.")
    selected_window = float(selected["window_s"])
    raw_main = raw_by_window[selected_window]
    raw_30 = raw_by_window[30.0]
    robust_main, scale_main = _robust_transform(raw_main, q05)
    robust_30, scale_30 = _robust_transform(raw_30, q05)

    q05_peaks = peak_time[q05]
    q05_signed_peaks = signed_peak_time[q05]
    q05_onsets = onset_time[q05]
    q05_pair_stats: list[dict[str, object]] = []
    for response_i, stimulus_j in np.argwhere(q05):
        energy30 = float(e30[response_i, stimulus_j])
        row: dict[str, object] = {
            "response_neuron": str(nominal_ids[response_i]),
            "stimulus_neuron": str(nominal_ids[stimulus_j]),
            "response_index": int(response_i),
            "stimulus_index": int(stimulus_j),
            "q_value": float(empirical_q[response_i, stimulus_j]),
            "occurrence_count": int(occurrence[response_i, stimulus_j]),
            "official_dff_30s": float(official_dff[response_i, stimulus_j]),
            "window_response": float(raw_main[response_i, stimulus_j]) if "raw_main" in locals() else float("nan"),
            "absolute_peak_time_s": float(peak_time[response_i, stimulus_j]),
            "signed_peak_time_s": float(signed_peak_time[response_i, stimulus_j]),
            "response_onset_s": float(onset_time[response_i, stimulus_j]),
            "absolute_peak_value": float(peak_value[response_i, stimulus_j]),
            "dominant_response_sign": int(np.sign(peak_value[response_i, stimulus_j])),
            "cumulative_abs_energy_30s": energy30,
        }
        for window in candidates:
            energy = float(energy_by_window[window][response_i, stimulus_j])
            row[f"energy_fraction_{window:g}s_vs_30s"] = energy / max(energy30, 1e-12)
            row[f"{operator}_{window:g}s"] = float(raw_by_window[window][response_i, stimulus_j])
        q05_pair_stats.append(row)
    audit = {
        "source": "official wormneuroatlas funatlas.h5 pairwise exponential-convolution kernels",
        "funatlas_time_compiled": funatlas_time_compiled,
        "kernel_key_order": kernel_key_order,
        "official_code_evidence": {
            "NeuroAtlas.py": "load_signal_propagation_atlas stores dFF as delta F/F averaged over a time window and trials; get_kernel returns an ExponentialConvolution_min kernel for pair i<-j.",
            "scripts/signalpropagation_kernels.py": "official example evaluates a returned kernel object on a positive time axis.",
            "ExponentialConvolution_min.py": "kernel terms are evaluated in the time domain as factor * time**power_t * exp(-g*time).",
        },
        "time_axis_start_s": float(time[0]),
        "time_axis_end_s": float(time[-1]),
        "time_step_s": dt,
        "observation_operator": operator,
        "kernel_sign_and_amplitude_definition": "Kernel amplitude is signed fitted delta-F/F response; positive and negative lobes are preserved. Absolute peak times use max |k_ij(t)|.",
        "kernel_contains_calcium_indicator_response": "The official signal-propagation kernels are fitted dF/F response kernels; they are evaluated as calcium-response kernels, not simultaneous whole-brain traces.",
        "pairwise_not_simultaneous_recording": True,
        "abnormal_kernel_policy": f"Pairs with non-finite evaluations or max |k(t)| > {max_abs:g} on the audit grid are excluded from valid-kernel masks.",
        "nominal_head_neuron_count": int(n),
        "observed_offdiagonal_pair_count": int(np.count_nonzero(observed)),
        "valid_kernel_pair_count": int(np.count_nonzero(valid_kernel & observed)),
        "valid_kernel_missing_rate_observed": float(1.0 - np.count_nonzero(valid_kernel & observed) / max(np.count_nonzero(observed), 1)),
        "abnormal_kernel_pair_count": int(np.count_nonzero(abnormal_kernel & observed)),
        "q05_observed_pair_count": int(np.count_nonzero(observed & np.isfinite(empirical_q) & (empirical_q < 0.05))),
        "q05_valid_kernel_pair_count": int(np.count_nonzero(q05)),
        "q10_valid_kernel_pair_count": int(np.count_nonzero(q10)),
        "peak_time_80pct_s": float(np.nanpercentile(q05_peaks, 80)) if q05_peaks.size else float("nan"),
        "peak_time_90pct_s": float(np.nanpercentile(q05_peaks, 90)) if q05_peaks.size else float("nan"),
        "signed_peak_time_80pct_s": float(np.nanpercentile(q05_signed_peaks, 80)) if q05_signed_peaks.size else float("nan"),
        "onset_time_80pct_s": float(np.nanpercentile(q05_onsets, 80)) if q05_onsets.size else float("nan"),
        "q05_absolute_peak_times_s": [float(x) for x in q05_peaks[np.isfinite(q05_peaks)]],
        "candidate_windows": candidate_rows,
        "selected_window_s": selected_window,
        "selection_rule": selection_rule,
        "selection_used_mechanism_results": False,
        "main_transform": "tanh_95pct_abs_main_q05",
        "main_transform_scale": scale_main,
        "official_dff_window_note": "NeuroAtlas documents dFF as averaged over a time window and trials; the frozen V4 object uses kernel signed mean so source and model use the same operator.",
    }
    return WindowTarget(
        time=time,
        selected_window_s=selected_window,
        empirical_raw=raw_main,
        empirical_robust=robust_main,
        empirical_raw_30=raw_30,
        empirical_robust_30=robust_30,
        valid_kernel_mask=valid_kernel,
        high_confidence_mask=q05,
        q10_mask=q10,
        all_observed_valid_mask=all_valid,
        transform_scale=scale_main,
        transform_scale_30=scale_30,
        audit=audit,
        q05_pair_stats=q05_pair_stats,
    )
