#!/usr/bin/env python
"""Run pretrained FlyVis on Pilot v2 native stimuli."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flyvis-root", type=Path, default=ROOT / "flyvis_data")
    parser.add_argument("--model", default="flow/0000/000")
    parser.add_argument("--stim-dir", type=Path, default=ROOT / "outputs" / "flyvis_pilot_v2" / "stimuli")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "flyvis_pilot_v2" / "responses")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ.setdefault("FLYVIS_ROOT_DIR", str(args.flyvis_root.resolve()))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "" if args.device == "cpu" else os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_flyvis")

    import torch
    import flyvis
    from flyvis import results_dir
    from flyvis.network import NetworkView

    stim_dir = args.stim_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stimuli = np.load(stim_dir / "stimuli.npy", mmap_mode="r")
    metadata = pd.read_csv(stim_dir / "metadata.csv")
    dt = float(metadata["dt"].iloc[0])

    network_view = NetworkView(results_dir / args.model)
    network = network_view.init_network()
    central_idx = network_view.connectome.central_cells_index[:]
    all_cell_types = network_view.connectome.nodes.type[:].astype(str)
    all_u = network_view.connectome.nodes.u[:]
    all_v = network_view.connectome.nodes.v[:]
    cell_meta = pd.DataFrame(
        {
            "neuron": np.arange(len(central_idx)),
            "node_index": central_idx,
            "cell_type": all_cell_types[central_idx],
            "u": all_u[central_idx],
            "v": all_v[central_idx],
        }
    )
    cell_meta.to_csv(out_dir / "cell_metadata.csv", index=False)

    response_path = out_dir / "flyvis_central_cell_responses.npy"
    responses = np.lib.format.open_memmap(
        response_path, mode="w+", dtype=np.float32, shape=(stimuli.shape[0], stimuli.shape[1], len(central_idx))
    )
    initial_states = {}
    for start in range(0, stimuli.shape[0], args.batch_size):
        stop = min(start + args.batch_size, stimuli.shape[0])
        print(f"Running FlyVis responses {start}:{stop} / {stimuli.shape[0]}", flush=True)
        batch = torch.as_tensor(np.asarray(stimuli[start:stop]).copy(), dtype=torch.float32)
        batch_n = int(batch.shape[0])
        if batch_n not in initial_states:
            print(f"Computing cached grey steady state for batch size {batch_n}", flush=True)
            # Match the 0.5 grey background used by the stimulus generator.
            # Explicitly match the 0.5 grey background used by the generator.
            initial_states[batch_n] = network.steady_state(
                t_pre=1.0,
                dt=dt,
                batch_size=batch_n,
                value=0.5,
            )
        with torch.no_grad():
            resp = network.simulate(batch, dt=dt, initial_state=initial_states[batch_n])
            central = resp[:, :, central_idx].detach().cpu().numpy().astype(np.float32)
        responses[start:stop] = central
        responses.flush()
        print(f"Wrote FlyVis responses {start}:{stop} / {stimuli.shape[0]}", flush=True)

    # Type-level responses are one central cell per type for this FlyVis central
    # output, but save an explicit long table for downstream analysis.
    time = np.arange(stimuli.shape[1], dtype=np.float32) * dt - float(metadata["t_pre"].iloc[0])
    type_rows = []
    for ni, row in cell_meta.iterrows():
        trace = responses[:, :, ni]
        baseline_frames = max(1, int(round(float(metadata["t_pre"].iloc[0]) / dt)))
        base = trace[:, :baseline_frames].mean(axis=1, keepdims=True)
        delta = trace - base
        peak = delta.max(axis=1)
        mean = delta.mean(axis=1)
        latency = time[np.argmax(delta, axis=1)]
        type_rows.append(
            pd.DataFrame(
                {
                    "sample": metadata["sample"].to_numpy(),
                    "cell_type": row.cell_type,
                    "mean_delta_response": mean,
                    "peak_delta_response": peak,
                    "latency_to_peak": latency,
                }
            )
        )
    type_summary = pd.concat(type_rows, ignore_index=True).merge(metadata, on="sample", how="left")
    type_summary.to_csv(out_dir / "type_response_summary.csv", index=False)
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    write_json(
        {
            "model": args.model,
            "flyvis_version": getattr(flyvis, "__version__", "unknown"),
            "stimulus_shape": list(stimuli.shape),
            "central_response_shape": list(responses.shape),
            "response_path": str(response_path),
            "n_trials": int(stimuli.shape[0]),
            "dt": dt,
            "steady_state_seconds": 1.0,
            "steady_state_grey_value": 0.5,
            "note": "Saved central-cell/type responses; all-node responses would be prohibitively large.",
        },
        out_dir / "run_manifest.json",
    )
    print(f"Wrote FlyVis central responses to {response_path} with shape {responses.shape}")


if __name__ == "__main__":
    main()
