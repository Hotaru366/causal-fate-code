"""Official Worm Neuro Atlas data access for the V4 head-network experiment."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


AWC_TO_FUNATLAS = {"AWCL": "AWCOF", "AWCR": "AWCON"}
AWC_TO_ANATOMY = {"AWCL": "AWCOFF", "AWCR": "AWCON"}


@dataclass(frozen=True)
class HeadData:
    neuron_ids: np.ndarray
    funatlas_ids: np.ndarray
    anatomy_ids: np.ndarray
    chemical: np.ndarray
    gap: np.ndarray
    chemical_polarity: np.ndarray
    chemical_reversal: np.ndarray
    empirical_raw: np.ndarray
    empirical: np.ndarray
    empirical_raw_30: np.ndarray
    empirical_30: np.ndarray
    empirical_q: np.ndarray
    occurrence: np.ndarray
    observed_mask: np.ndarray
    main_mask: np.ndarray
    q10_mask: np.ndarray
    all_observed_valid_mask: np.ndarray
    valid_kernel_mask: np.ndarray
    response_neuron_mask: np.ndarray
    stimulus_neuron_mask: np.ndarray
    transform_scale: float
    transform_scale_30: float
    selected_window_s: float
    provenance: dict[str, object]


def ensure_official_repo(root: Path, repo_url: str, commit: str) -> Path:
    """Fetch the pinned official repository into an external raw-data cache."""

    raw_dir = Path(os.environ.get("CELEGANS_RAW_DIR", "~/.cache/causal-fate/celegans/raw")).expanduser().resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    repo = raw_dir / "wormneuroatlas"
    if not repo.exists():
        subprocess.run(["git", "clone", repo_url, str(repo)], check=True)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
        raise RuntimeError(f"Upstream cache has local changes: {repo}")

    has_commit = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=repo,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not has_commit:
        subprocess.run(["git", "fetch", "--tags", "--all"], cwd=repo, check=True)
    subprocess.run(["git", "checkout", commit], cwd=repo, check=True)
    return repo


def official_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def _head_ids(data_dir: Path) -> np.ndarray:
    with (data_dir / "aconnectome_ids_ganglia.json").open("r", encoding="utf-8") as fh:
        ganglia = json.load(fh)
    head: list[str] = []
    for ganglion_name in ganglia["head"]:
        head.extend(ganglia[ganglion_name])
    return np.unique(np.asarray(head, dtype=str))


def _indices(ids: np.ndarray, selected: np.ndarray) -> np.ndarray:
    index = {str(name): i for i, name in enumerate(ids)}
    missing = [str(name) for name in selected if str(name) not in index]
    if missing:
        raise KeyError(f"IDs missing from source matrix: {missing}")
    return np.asarray([index[str(name)] for name in selected], dtype=int)


def _transform_empirical(empirical_raw: np.ndarray, observed_mask: np.ndarray) -> tuple[np.ndarray, float]:
    scale = float(np.nanpercentile(np.abs(empirical_raw[observed_mask]), 95))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("Cannot compute a positive empirical transform scale.")
    empirical = np.full_like(empirical_raw, np.nan, dtype=float)
    empirical[observed_mask] = np.tanh(empirical_raw[observed_mask] / scale)
    diag = np.diag_indices_from(empirical_raw)
    diag_observed = np.isfinite(empirical_raw[diag])
    empirical[diag[0][diag_observed], diag[1][diag_observed]] = np.tanh(
        empirical_raw[diag[0][diag_observed], diag[1][diag_observed]] / scale
    )
    return empirical, scale


def _normalize_sign_id(name: object) -> str:
    out = str(name)
    if len(out) > 2 and out[-2:].isdigit() and int(out[-2:]) < 10:
        out = out[:-2] + str(int(out[-2:]))
    if out == "AWCL":
        return "AWCOFF"
    if out == "AWCR":
        return "AWCON"
    return out


def _load_synapse_polarity(
    data_dir: Path,
    anatomy_ids: np.ndarray,
    chemical: np.ndarray,
    fixed_cfg: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Map Fenyves-style sign predictions to the head chemical adjacency."""

    sign_path = data_dir / "journal.pcbi.1007974.s003.xlsx"
    polarity = np.full_like(chemical, np.nan, dtype=float)
    if sign_path.exists():
        sign_df = pd.read_excel(sign_path, sheet_name="5. Sign prediction", header=1)
        index = {str(name): i for i, name in enumerate(anatomy_ids)}
        for _, row in sign_df.iterrows():
            src = _normalize_sign_id(row.get("Neuron"))
            dst = _normalize_sign_id(row.get("Neuron.1"))
            if src not in index or dst not in index:
                continue
            dst_i = index[dst]
            src_j = index[src]
            pred = str(row.get("Unnamed: 16"))
            if pred == "+":
                polarity[dst_i, src_j] = 1.0
            elif pred == "-":
                polarity[dst_i, src_j] = -1.0
            elif pred == "complex":
                polarity[dst_i, src_j] = 0.0

    nt_fallback = np.ones(len(anatomy_ids), dtype=float)
    if sign_path.exists():
        nt_df = pd.read_excel(sign_path, sheet_name="1. NT expr")
        nt_by_neuron = {
            _normalize_sign_id(row["Class member"]): str(row["Dominant NT"])
            for _, row in nt_df.iterrows()
        }
        nt_fallback = np.asarray(
            [-1.0 if nt_by_neuron.get(str(name)) == "GABA" else 1.0 for name in anatomy_ids],
            dtype=float,
        )

    chemical_edges = chemical > 0.0
    missing = chemical_edges & ~np.isfinite(polarity)
    fallback_matrix = np.broadcast_to(nt_fallback[None, :], chemical.shape)
    polarity[missing] = fallback_matrix[missing]
    polarity[~chemical_edges] = 0.0

    neutral = float(fixed_cfg["neutral_reversal_mV"])
    excit = float(fixed_cfg["exc_reversal_mV"])
    inhib = float(fixed_cfg["inh_reversal_mV"])
    reversal = np.full_like(chemical, neutral, dtype=float)
    reversal[polarity > 0.0] = excit
    reversal[polarity < 0.0] = inhib
    reversal[polarity == 0.0] = neutral
    reversal[~chemical_edges] = neutral

    metadata = {
        "synapse_sign_source": "Fenyves et al. 2020 predictions distributed as wormneuroatlas journal.pcbi.1007974.s003.xlsx",
        "predicted_excitatory_head_edges": int(np.count_nonzero(chemical_edges & (polarity > 0.0))),
        "predicted_inhibitory_head_edges": int(np.count_nonzero(chemical_edges & (polarity < 0.0))),
        "predicted_complex_or_neutral_head_edges": int(np.count_nonzero(chemical_edges & (polarity == 0.0))),
        "polarity_fallback_edges": int(np.count_nonzero(missing)),
        "polarity_fallback_rule": "unknown chemical signs use dominant neurotransmitter if available; GABA inhibitory, otherwise excitatory",
        "chemical_matrix_direction": "chemical[i,j] is synapse count from presynaptic neuron j to postsynaptic neuron i",
        "gap_matrix_direction": "gap is symmetrized and treated as electrical coupling between i and j",
    }
    return polarity, reversal, metadata


def load_head_data(root: Path, config: dict[str, object], window_target=None) -> HeadData:
    repo = ensure_official_repo(
        root,
        str(config["official_repo_url"]),
        str(config["official_repo_commit"]),
    )
    data_dir = repo / "wormneuroatlas" / "data"
    nominal_ids = _head_ids(data_dir)
    funatlas_ids = np.asarray([AWC_TO_FUNATLAS.get(str(name), str(name)) for name in nominal_ids])
    anatomy_ids = np.asarray([AWC_TO_ANATOMY.get(str(name), str(name)) for name in nominal_ids])

    with h5py.File(data_dir / "funatlas.h5", "r") as fun_h5:
        source_fun_ids = np.asarray([name.decode("utf-8") for name in fun_h5["neuron_ids"][:]])
        fun_idx = _indices(source_fun_ids, funatlas_ids)
        strain = str(config["strain"])
        official_empirical_raw = fun_h5[strain]["dFF"][:][fun_idx][:, fun_idx].astype(float)
        empirical_q = fun_h5[strain]["q"][:][fun_idx][:, fun_idx].astype(float)
        occurrence = fun_h5[strain]["occ1"][:][fun_idx][:, fun_idx].astype(int)
        funatlas_attrs = {
            key: (val.decode("utf-8") if isinstance(val, bytes) else str(val))
            for key, val in fun_h5.attrs.items()
        }

    source_anatomy_ids = np.loadtxt(data_dir / "neuron_ids.txt", dtype=str)[:, 1]
    anatomy_idx = _indices(source_anatomy_ids, anatomy_ids)
    with h5py.File(data_dir / "aconnectome_default.h5", "r") as aconn_h5:
        chemical = aconn_h5["chem"][:][anatomy_idx][:, anatomy_idx].astype(float)
        gap = aconn_h5["gap"][:][anatomy_idx][:, anatomy_idx].astype(float)
    gap = 0.5 * (gap + gap.T)
    np.fill_diagonal(chemical, 0.0)
    np.fill_diagonal(gap, 0.0)

    observed_mask = np.isfinite(official_empirical_raw)
    observed_mask &= ~np.eye(len(nominal_ids), dtype=bool)
    official_empirical, official_transform_scale = _transform_empirical(official_empirical_raw, observed_mask)
    if window_target is None:
        empirical_raw = official_empirical_raw
        empirical = official_empirical
        main_mask = observed_mask
        q10_mask = observed_mask & np.isfinite(empirical_q) & (empirical_q < 0.10)
        all_valid = observed_mask
        valid_kernel_mask = observed_mask
        transform_scale = official_transform_scale
        transform_scale_30 = official_transform_scale
        selected_window_s = 30.0
    else:
        empirical_raw = window_target.empirical_raw
        empirical = window_target.empirical_robust
        main_mask = window_target.high_confidence_mask
        q10_mask = window_target.q10_mask
        all_valid = window_target.all_observed_valid_mask
        valid_kernel_mask = window_target.valid_kernel_mask
        transform_scale = float(window_target.transform_scale)
        transform_scale_30 = official_transform_scale
        selected_window_s = float(window_target.selected_window_s)

    polarity, reversal, sign_metadata = _load_synapse_polarity(
        data_dir,
        anatomy_ids,
        chemical,
        dict(config["biophysical_fixed"]),
    )
    response_neuron_mask = np.any(observed_mask, axis=1)
    stimulus_neuron_mask = np.any(observed_mask, axis=0)
    provenance = {
        "source": "francescorandi/wormneuroatlas",
        "paper": "Randi et al., Neural signal propagation atlas of Caenorhabditis elegans, Nature, 2023",
        "repo_url": config["official_repo_url"],
        "repo_commit": official_commit(repo),
        "funatlas_time_compiled": funatlas_attrs.get("time_compiled"),
        "funatlas_kernel_keys": funatlas_attrs.get("kernels_keys"),
        "nominal_head_neuron_count": int(len(nominal_ids)),
        "used_head_neuron_count": int(len(funatlas_ids)),
        "observed_offdiagonal_pair_count": int(np.count_nonzero(observed_mask)),
        "response_neuron_count_with_observations": int(np.count_nonzero(response_neuron_mask)),
        "stimulus_neuron_count_with_observations": int(np.count_nonzero(stimulus_neuron_mask)),
        "missing_response_neurons": nominal_ids[~response_neuron_mask].tolist(),
        "missing_stimulus_neurons": nominal_ids[~stimulus_neuron_mask].tolist(),
        "awc_mapping_for_funatlas": AWC_TO_FUNATLAS,
        "awc_mapping_for_anatomy": AWC_TO_ANATOMY,
        "empirical_transform": config["empirical_transform"],
        "empirical_transform_scale": transform_scale,
        "empirical_30s_transform_scale": transform_scale_30,
        "selected_window_s": selected_window_s,
        "valid_kernel_pair_count": int(np.count_nonzero(valid_kernel_mask & observed_mask)),
        "main_q05_window_pair_count": int(np.count_nonzero(main_mask)),
        "q10_window_pair_count": int(np.count_nonzero(q10_mask)),
        "all_observed_valid_kernel_pair_count": int(np.count_nonzero(all_valid)),
        **sign_metadata,
    }
    return HeadData(
        neuron_ids=nominal_ids,
        funatlas_ids=funatlas_ids,
        anatomy_ids=anatomy_ids,
        chemical=chemical,
        gap=gap,
        chemical_polarity=polarity,
        chemical_reversal=reversal,
        empirical_raw=empirical_raw,
        empirical=empirical,
        empirical_raw_30=official_empirical_raw,
        empirical_30=official_empirical,
        empirical_q=empirical_q,
        occurrence=occurrence,
        observed_mask=observed_mask,
        main_mask=main_mask,
        q10_mask=q10_mask,
        all_observed_valid_mask=all_valid,
        valid_kernel_mask=valid_kernel_mask,
        response_neuron_mask=response_neuron_mask,
        stimulus_neuron_mask=stimulus_neuron_mask,
        transform_scale=transform_scale,
        transform_scale_30=transform_scale_30,
        selected_window_s=selected_window_s,
        provenance=provenance,
    )


def save_operating_data(root: Path, data: HeadData) -> None:
    out = root / "results" / "operating_data"
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "neuron_ids.txt", data.neuron_ids, fmt="%s")
    np.savetxt(out / "funatlas_ids.txt", data.funatlas_ids, fmt="%s")
    np.savetxt(out / "anatomy_ids.txt", data.anatomy_ids, fmt="%s")
    np.save(out / "chemical_adjacency.npy", data.chemical)
    np.save(out / "gap_junction_adjacency.npy", data.gap)
    np.save(out / "chemical_polarity.npy", data.chemical_polarity)
    np.save(out / "chemical_reversal.npy", data.chemical_reversal)
    np.save(out / "empirical_raw_dff.npy", data.empirical_raw)
    np.save(out / "empirical_propagation_matrix.npy", data.empirical)
    np.save(out / "empirical_30s_raw_dff.npy", data.empirical_raw_30)
    np.save(out / "empirical_30s_propagation_matrix.npy", data.empirical_30)
    np.save(out / "empirical_q.npy", data.empirical_q)
    np.save(out / "occurrence_matrix.npy", data.occurrence)
    np.save(out / "observed_pair_mask.npy", data.observed_mask)
    np.save(out / "main_high_confidence_mask.npy", data.main_mask)
    np.save(out / "q10_mask.npy", data.q10_mask)
    np.save(out / "all_observed_valid_kernel_mask.npy", data.all_observed_valid_mask)
    np.save(out / "valid_kernel_mask.npy", data.valid_kernel_mask)
    (out / "data_provenance.json").write_text(
        json.dumps(data.provenance, indent=2, sort_keys=True),
        encoding="utf-8",
    )
