#!/usr/bin/env python
"""Direct FlyVis circuit perturbations on unique stimulus conditions.

This script edits the loaded FlyVis network before simulation. It evaluates:

* the intact checkpoint;
* complete T4/T5 state silencing via FlyVis's documented state-hook API;
* spatial synapse-count kernel shuffling within each source/target type pair;
* learned type-pair synaptic-strength shuffling within sign classes.

No perturbed network is retrained.  Results are therefore acute model
interventions, not a claim about how an alternative architecture would perform
after matched optical-flow training.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_FAMILIES = ["moving_bar", "small_translating_target"]
ALL_DIRECTION_FAMILIES = ["moving_edge", *PRIMARY_FAMILIES]


def parse_list(text: str) -> list[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def temporal_bins(resp: np.ndarray, pre_frames: int, bins: int = 5) -> np.ndarray:
    base = resp[:, :pre_frames].mean(axis=1, keepdims=True)
    delta = resp - base
    return np.concatenate([chunk.mean(axis=1) for chunk in np.array_split(delta, bins, axis=1)], axis=1).astype(np.float32)


def classifier(seed: int):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
    )


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(center - radius), float(center + radius)


def apply_perturbation(network, variant: str, seed: int) -> dict[str, object]:
    import torch

    info: dict[str, object] = {"variant": variant, "seed": seed}
    rng = np.random.default_rng(seed)
    if variant == "full":
        info["n_modified_parameter_groups"] = 0
        return info
    if variant == "t4t5_state_silencing":
        node_types = network.connectome.nodes.type[:].astype(str)
        silence_idx = np.flatnonzero(np.char.startswith(node_types, "T4") | np.char.startswith(node_types, "T5"))
        idx_tensor = torch.tensor(silence_idx, dtype=torch.long)

        def silence_hook(state, indices):
            activity = state.nodes.activity.clone()
            activity[..., indices] = 0.0
            state.nodes.activity = activity
            return state

        network.register_state_hook(silence_hook, indices=idx_tensor)
        info["n_silenced_nodes"] = int(len(silence_idx))
        info["silenced_cell_types"] = sorted(set(node_types[silence_idx].tolist()))
        return info
    if variant == "spatial_kernel_shuffle":
        param = network.edge_params.syn_count
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, key in enumerate(param.keys):
            groups[(str(key[0]), str(key[1]))].append(index)
        source = param.raw_values.detach().clone()
        modified = 0
        with torch.no_grad():
            for indices in groups.values():
                if len(indices) < 2:
                    continue
                permutation = rng.permutation(indices)
                param.raw_values[indices] = source[permutation]
                modified += len(indices)
        info["n_modified_parameter_groups"] = int(modified)
        info["preserved"] = "source/target cell-type pair, sign, and multiset of spatial-kernel synapse counts"
        return info
    if variant == "type_pair_strength_shuffle":
        strength = network.edge_params.syn_strength
        sign = network.edge_params.sign
        if list(strength.keys) != list(sign.keys):
            raise RuntimeError("Synaptic strength and sign group keys are not aligned.")
        source = strength.raw_values.detach().clone()
        signs = sign.semantic_values.detach().cpu().numpy()
        modified = 0
        with torch.no_grad():
            for sign_value in np.unique(signs):
                indices = np.flatnonzero(signs == sign_value)
                permutation = rng.permutation(indices)
                strength.raw_values[indices] = source[permutation]
                modified += len(indices)
        info["n_modified_parameter_groups"] = int(modified)
        info["preserved"] = "connectome topology, edge signs, and global strength distribution within sign class"
        return info
    raise ValueError(f"Unknown variant: {variant}")


def simulate_variant(
    checkpoint: str,
    variant: str,
    stimuli: np.ndarray,
    dt: float,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    import torch
    from flyvis import results_dir
    from flyvis.network import NetworkView

    network_view = NetworkView(results_dir / f"flow/0000/{checkpoint}")
    network = network_view.init_network()
    perturbation = apply_perturbation(network, variant, seed)
    central_idx = network_view.connectome.central_cells_index[:]
    all_types = network_view.connectome.nodes.type[:].astype(str)
    cell_meta = pd.DataFrame(
        {
            "central_feature_index": np.arange(len(central_idx)),
            "node_index": central_idx,
            "cell_type": all_types[central_idx],
        }
    )
    output = np.empty((len(stimuli), stimuli.shape[1], len(central_idx)), dtype=np.float32)
    initial_states = {}
    for start in range(0, len(stimuli), batch_size):
        stop = min(start + batch_size, len(stimuli))
        n_batch = stop - start
        if n_batch not in initial_states:
            with torch.no_grad():
                initial_states[n_batch] = network.steady_state(
                    t_pre=1.0,
                    dt=dt,
                    batch_size=n_batch,
                    value=0.5,
                )
        batch = torch.tensor(np.asarray(stimuli[start:stop]).copy(), dtype=torch.float32)
        with torch.no_grad():
            response = network.simulate(batch, dt=dt, initial_state=initial_states[n_batch])
        output[start:stop] = response[:, :, central_idx].detach().cpu().numpy().astype(np.float32)
        print(f"{checkpoint}/{variant}: {stop}/{len(stimuli)}", flush=True)
    return output, cell_meta, perturbation


def predictions_refit(x: np.ndarray, meta: pd.DataFrame, subset: np.ndarray, seed: int, label_encoder: LabelEncoder) -> pd.DataFrame:
    y = label_encoder.transform(meta["direction"].astype(str))
    rows = []
    for heldout in sorted(meta.loc[subset, "scale"].unique()):
        train = subset & (meta["scale"].to_numpy() != heldout)
        test = subset & (meta["scale"].to_numpy() == heldout)
        clf = classifier(seed)
        clf.fit(x[train], y[train])
        pred = clf.predict(x[test])
        for idx, yt, yp in zip(np.flatnonzero(test), y[test], pred):
            rows.append(
                {
                    "condition_index": int(idx),
                    "heldout_scale": float(heldout),
                    "true_label": str(label_encoder.inverse_transform([yt])[0]),
                    "pred_label": str(label_encoder.inverse_transform([yp])[0]),
                    "correct": bool(yt == yp),
                }
            )
    return pd.DataFrame(rows)


def predictions_fixed_full_decoder(
    full_x: np.ndarray,
    variant_x: np.ndarray,
    meta: pd.DataFrame,
    subset: np.ndarray,
    seed: int,
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    y = label_encoder.transform(meta["direction"].astype(str))
    rows = []
    for heldout in sorted(meta.loc[subset, "scale"].unique()):
        train = subset & (meta["scale"].to_numpy() != heldout)
        test = subset & (meta["scale"].to_numpy() == heldout)
        clf = classifier(seed)
        clf.fit(full_x[train], y[train])
        pred = clf.predict(variant_x[test])
        for idx, yt, yp in zip(np.flatnonzero(test), y[test], pred):
            rows.append(
                {
                    "condition_index": int(idx),
                    "heldout_scale": float(heldout),
                    "true_label": str(label_encoder.inverse_transform([yt])[0]),
                    "pred_label": str(label_encoder.inverse_transform([yp])[0]),
                    "correct": bool(yt == yp),
                }
            )
    return pd.DataFrame(rows)


def summarize_predictions(pred: pd.DataFrame, checkpoint: str, variant: str, decoder: str, analysis_set: str) -> list[dict[str, object]]:
    scales = sorted(pred["heldout_scale"].unique())
    lower, upper = scales[0], scales[-1]
    regimes = {
        "all_scales": pred,
        "interior_interpolation": pred[pred["heldout_scale"].isin(scales[1:-1])],
        "lower_boundary": pred[pred["heldout_scale"] == lower],
        "upper_boundary": pred[pred["heldout_scale"] == upper],
    }
    rows = []
    for regime, data in regimes.items():
        vals = data["correct"].astype(bool).to_numpy()
        lo, hi = wilson(int(vals.sum()), len(vals))
        rows.append(
            {
                "checkpoint": checkpoint,
                "variant": variant,
                "decoder": decoder,
                "analysis_set": analysis_set,
                "regime": regime,
                "accuracy": float(vals.mean()),
                "wilson_95_low": lo,
                "wilson_95_high": hi,
                "n_unique_conditions": len(vals),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "direct_flyvis_perturbations")
    parser.add_argument("--flyvis-root", type=Path, default=ROOT / "flyvis_data")
    parser.add_argument("--checkpoints", default="000")
    parser.add_argument(
        "--variants",
        default="full,t4t5_state_silencing,spatial_kernel_shuffle,type_pair_strength_shuffle",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument(
        "--full-response-path",
        type=Path,
        default=None,
        help="Optional intact-response .npy to reuse when --variants omits full.",
    )
    parser.add_argument(
        "--reuse-existing-responses",
        action="store_true",
        help="Reuse response .npy files already present in --out-dir (useful for evaluation-only recovery).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("FLYVIS_ROOT_DIR", str(args.flyvis_root.resolve()))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_flyvis")
    t0 = time.time()

    all_stim = np.load(args.outputs_dir / "flyvis_pilot_v2" / "stimuli" / "stimuli.npy", mmap_mode="r")
    all_meta = pd.read_csv(args.outputs_dir / "flyvis_pilot_v2" / "responses" / "metadata.csv")
    select = (all_meta["repeat"] == 0) & all_meta["feature_family"].isin(ALL_DIRECTION_FAMILIES)
    source_samples = np.flatnonzero(select.to_numpy())
    stimuli = all_stim[source_samples]
    meta = all_meta.iloc[source_samples].reset_index(drop=True)
    dt = float(meta["dt"].iloc[0])
    pre_frames = int(round(float(meta["t_pre"].iloc[0]) / dt))
    label_encoder = LabelEncoder().fit(meta["direction"].astype(str))
    subsets = {
        "complete_mixture_including_scale_neutral_edge": np.ones(len(meta), dtype=bool),
        "scale_varying_moving_bar_and_target": meta["feature_family"].isin(PRIMARY_FAMILIES).to_numpy(),
    }

    feature_sets: dict[tuple[str, str], np.ndarray] = {}
    perturbation_info = []
    stability_rows = []
    cell_meta_saved = False
    checkpoints = parse_list(args.checkpoints)
    variants = parse_list(args.variants)
    if "full" not in variants:
        if args.full_response_path is None:
            raise ValueError("--full-response-path is required when --variants omits full")
        full_response = np.load(args.full_response_path)
        if full_response.shape[:2] != (len(meta), stimuli.shape[1]):
            raise ValueError(
                f"Reusable intact response has shape {full_response.shape}, expected first dimensions {(len(meta), stimuli.shape[1])}"
            )
        for checkpoint in checkpoints:
            feature_sets[(checkpoint, "full")] = temporal_bins(full_response, pre_frames, bins=5)
    for checkpoint in checkpoints:
        for variant in variants:
            response_path = args.out_dir / f"responses_checkpoint{checkpoint}_{variant}.npy"
            if args.reuse_existing_responses and response_path.exists():
                response = np.load(response_path)
                info = {"variant": variant, "seed": args.seed, "reused_existing_response": True}
            else:
                response, cell_meta, info = simulate_variant(
                    checkpoint,
                    variant,
                    stimuli,
                    dt,
                    args.batch_size,
                    args.seed,
                )
                np.save(response_path, response)
                if not cell_meta_saved:
                    cell_meta.to_csv(args.out_dir / "central_cell_metadata.csv", index=False)
                    cell_meta_saved = True
            n_nonfinite = int(response.size - np.isfinite(response).sum())
            stability_rows.append(
                {
                    "checkpoint": checkpoint,
                    "variant": variant,
                    "seed": args.seed,
                    "finite_response": n_nonfinite == 0,
                    "n_nonfinite_values": n_nonfinite,
                    "n_response_values": int(response.size),
                }
            )
            if n_nonfinite == 0:
                feature_sets[(checkpoint, variant)] = temporal_bins(response, pre_frames, bins=5)
            perturbation_info.append({"checkpoint": checkpoint, **info})

    prediction_tables = []
    summary_rows = []
    for checkpoint in checkpoints:
        full_x = feature_sets[(checkpoint, "full")]
        for variant in variants:
            if (checkpoint, variant) not in feature_sets:
                continue
            x = feature_sets[(checkpoint, variant)]
            for analysis_set, subset in subsets.items():
                refit = predictions_refit(x, meta, subset, args.seed, label_encoder)
                refit["checkpoint"] = checkpoint
                refit["variant"] = variant
                refit["decoder"] = "refit_within_variant"
                refit["analysis_set"] = analysis_set
                prediction_tables.append(refit)
                summary_rows.extend(summarize_predictions(refit, checkpoint, variant, "refit_within_variant", analysis_set))

                fixed = predictions_fixed_full_decoder(full_x, x, meta, subset, args.seed, label_encoder)
                fixed["checkpoint"] = checkpoint
                fixed["variant"] = variant
                fixed["decoder"] = "fixed_intact_decoder"
                fixed["analysis_set"] = analysis_set
                prediction_tables.append(fixed)
                summary_rows.extend(summarize_predictions(fixed, checkpoint, variant, "fixed_intact_decoder", analysis_set))

    predictions = pd.concat(prediction_tables, ignore_index=True)
    predictions = predictions.merge(
        meta.reset_index().rename(columns={"index": "condition_index"})[
            ["condition_index", "sample", "feature_family", "scale", "contrast", "repeat"]
        ],
        on="condition_index",
        how="left",
        validate="many_to_one",
    )
    predictions.to_csv(args.out_dir / "predictions_direct_perturbations.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "table_direct_perturbation_summary.csv", index=False)
    stability = pd.DataFrame(stability_rows)
    stability.to_csv(args.out_dir / "response_stability.csv", index=False)
    pd.DataFrame(perturbation_info).to_json(args.out_dir / "perturbation_details.json", orient="records", indent=2)
    try:
        summary.to_markdown(args.out_dir / "table_direct_perturbation_summary.md", index=False)
    except Exception:
        pass

    run_info = {
        "status": "completed" if stability["finite_response"].all() else "completed_with_nonfinite_intervention",
        "python": platform.python_version(),
        "runtime_seconds": time.time() - t0,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_unique_conditions": len(meta),
        "source_samples": source_samples.tolist(),
        "pre_frames": pre_frames,
        "direct_network_intervention": True,
        "networks_retrained_after_intervention": False,
    }
    (args.out_dir / "run_info.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
