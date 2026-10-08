#!/usr/bin/env python3
"""Rebuild numerical inputs from the provider, without bundled observations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from celegans_dtpr_v4 import experiment as ex
from celegans_dtpr_v4.data import load_head_data, save_operating_data
from celegans_dtpr_v4.dynamics import fit_node_readout_gain, simulate_continuous
from celegans_dtpr_v4.kernel_window import build_kernel_window_target


def main() -> None:
    definition = json.loads((ROOT / "config/default.json").read_text())
    config = definition["config"]
    # The data-access module retains the historical spelling of this key.
    input_config = dict(config, empirical_transform=config["source_response_transform"])
    target = build_kernel_window_target(ROOT, input_config)
    data = load_head_data(ROOT, input_config, target)
    if target.selected_window_s != config["simulation"]["selected_observation_window_s"]:
        raise ValueError("Source-derived observation window differs from the fixed definition")
    prepared = ex.FROZEN_ROOT
    prepared.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=prepared.parent) as work:
        work = Path(work)
        save_operating_data(work, data)
        shutil.copytree(work / "results/operating_data", prepared / "operating_data", dirs_exist_ok=True)
    definition["readout"]["coefficients"] = np.ones(len(data.neuron_ids)).tolist()
    ex.write_json(prepared / "frozen_config.json", definition)
    ex.write_json(prepared / "provenance_manifest.json", {"array_files": []})
    frozen = ex.load_frozen_operating_object()
    # Recreate the historical substrate readout at dt=0.1 with its Euler map.
    # The final finite-transport evaluation below uses refined RK4 throughout.
    model = ex.build_model(frozen, float(config["simulation"]["dt"]), 0.5)
    reference = simulate_continuous(model)
    gain = fit_node_readout_gain(reference.raw_propagation, data.empirical,
                                 data.main_mask, config["calibration"]["readout_ridge"])
    definition["readout"]["coefficients"] = gain.tolist()
    definition["readout"]["nonzero_coefficients"] = int(np.count_nonzero(gain))
    ex.write_json(prepared / "frozen_config.json", definition)
    expected = json.loads((ROOT / "config/expected_input_hashes.json").read_text())
    records = []
    for path in sorted((prepared / "operating_data").glob("*.npy")):
        array = np.load(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"file": path.name, "sha256": digest, "shape": list(array.shape),
                        "dtype": str(array.dtype), "matches_reference_hash": expected.get(path.name) == digest})
    ex.write_json(prepared / "provenance_manifest.json", {
        "array_files": records,
        "source_commit": config["official_repo_commit"],
        "preparation": "source kernels, anatomy and signs; fixed substrate parameters; regenerated readout",
        "input_hashes_all_match": all(r["matches_reference_hash"] for r in records if r["file"] in expected),
    })
    ex.write_json(prepared / "kernel_window_audit.json", target.audit)
    print(json.dumps({"prepared": str(prepared), "neurons": len(data.neuron_ids),
                      "window_s": target.selected_window_s,
                      "observed_pairs": int(data.observed_mask.sum()),
                      "high_confidence_pairs": int(data.main_mask.sum()),
                      "readout_nonzero": int(np.count_nonzero(gain)),
                      "reference_array_hash_matches": sum(r["matches_reference_hash"] for r in records)}, indent=2))


if __name__ == "__main__":
    main()
