#!/usr/bin/env python
"""Audit LOSO results at the independent-condition level.

This script checks three data-quality and evaluation issues:

1. The legacy generator recorded ten repeat seeds but did not use them, so
   repeats are exact stimulus duplicates.  Uncertainty is therefore computed
   over unique rendered conditions, not over the duplicated rows.
2. Moving edges are unchanged by the ``scale`` argument (an infinite edge is
   scale-neutral). Results are reported both for the complete mixture and for
   the genuinely scale-varying moving-bar/target families.
3. Interpolation and the lower/upper boundary extrapolations are reported
   separately, including the worst held-out scale.

The script also audits 24x24 and 32x32 coordinate projections. The full
TemporalResNet run used 32x32, where all 721 hex samples occupy distinct
grid cells; the lightweight STN run used 24x24, where collisions occur.
It also re-scores both unmatched baselines on the same de-duplicated primary
condition set used for the unique-condition FlyVis statistic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    radius = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return float(center - radius), float(center + radius)


def stratified_condition_bootstrap(
    df: pd.DataFrame,
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[float, float]:
    """Bootstrap unique conditions within held-out-scale x class strata."""

    groups = [g["correct"].astype(float).to_numpy() for _, g in df.groupby(["heldout_scale", "true_label"], sort=False)]
    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled = [vals[rng.integers(0, len(vals), len(vals))] for vals in groups]
        boots[b] = np.concatenate(sampled).mean()
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def summarize_subset(
    df: pd.DataFrame,
    analysis_set: str,
    regime: str,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    vals = df["correct"].astype(bool).to_numpy()
    k = int(vals.sum())
    n = int(len(vals))
    wlo, whi = wilson_interval(k, n)
    blo, bhi = stratified_condition_bootstrap(df, np.random.default_rng(seed), n_boot)
    return {
        "analysis_set": analysis_set,
        "regime": regime,
        "accuracy": float(vals.mean()),
        "n_unique_conditions": n,
        "n_correct": k,
        "wilson_95_low": wlo,
        "wilson_95_high": whi,
        "condition_bootstrap_95_low": blo,
        "condition_bootstrap_95_high": bhi,
    }


def projection_audit(coords: pd.DataFrame, grid_size: int) -> dict[str, object]:
    x = coords["x"].to_numpy(dtype=float)
    y = coords["y"].to_numpy(dtype=float)
    gx = np.round((x - x.min()) / max(x.max() - x.min(), 1e-8) * (grid_size - 1)).astype(int)
    gy = np.round((y - y.min()) / max(y.max() - y.min(), 1e-8) * (grid_size - 1)).astype(int)
    occupied = len(set(zip(gx.tolist(), gy.tolist())))
    return {
        "grid_size": grid_size,
        "n_hex_samples": len(coords),
        "n_occupied_grid_cells": occupied,
        "n_sample_collisions": len(coords) - occupied,
        "occupancy_fraction": occupied / float(grid_size * grid_size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "unique_condition_audit")
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stim_path = args.outputs_dir / "flyvis_pilot_v2" / "stimuli" / "stimuli.npy"
    meta_path = args.outputs_dir / "flyvis_pilot_v2" / "responses" / "metadata.csv"
    pred_path = args.outputs_dir / "flyvis_pilot_v3" / "tables" / "direction_loso_predictions_all_models.csv"
    coord_path = args.outputs_dir / "flyvis_pilot_v2" / "stimuli" / "hex_coordinates.csv"

    meta = pd.read_csv(meta_path)
    pred = pd.read_csv(pred_path)
    pred = pred[pred["model"] == "flyvis"].copy()
    pred = pred.merge(
        meta[["sample", "feature_family", "scale", "angle", "contrast", "repeat", "seed"]],
        on="sample",
        how="left",
        validate="one_to_one",
    )

    cluster_cols = ["feature_family", "scale", "angle", "contrast"]
    stimuli = np.load(stim_path, mmap_mode="r")
    duplicate_rows = []
    dynamic_meta = meta[meta["feature_family"].isin(["moving_edge", "moving_bar", "small_translating_target"])]
    for key, group in dynamic_meta.groupby(cluster_cols, sort=False):
        ids = group.sort_values("repeat")["sample"].astype(int).to_numpy()
        stimulus_identical = all(np.array_equal(stimuli[ids[0]], stimuli[j]) for j in ids[1:])
        pgroup = pred[pred["sample"].isin(ids)].sort_values("repeat")
        prediction_identical = pgroup["pred_label"].nunique(dropna=False) == 1 and pgroup["correct"].nunique(dropna=False) == 1
        duplicate_rows.append(
            {
                **dict(zip(cluster_cols, key)),
                "n_recorded_repeats": len(ids),
                "stimuli_pixel_identical": bool(stimulus_identical),
                "predictions_identical": bool(prediction_identical),
            }
        )
    duplicate_audit = pd.DataFrame(duplicate_rows)
    duplicate_audit.to_csv(args.out_dir / "table_repeat_independence_audit.csv", index=False)

    # One row per actual rendered condition.  Using repeat zero is exact because
    # the audit above verifies equality across every legacy repeat cluster.
    unique = pred[pred["repeat"] == 0].copy()
    scales = sorted(unique["heldout_scale"].unique())
    lower, upper = scales[0], scales[-1]
    interior = scales[1:-1]
    analysis_sets = {
        "complete_mixture_including_scale_neutral_edge": unique,
        "scale_varying_moving_bar_and_target": unique[unique["feature_family"].isin(["moving_bar", "small_translating_target"])],
    }

    summary_rows: list[dict[str, object]] = []
    by_scale_rows: list[dict[str, object]] = []
    for set_index, (name, data) in enumerate(analysis_sets.items()):
        regimes = {
            "all_scales": data,
            "interpolation_interior_scales": data[data["heldout_scale"].isin(interior)],
            "extrapolation_both_boundaries": data[data["heldout_scale"].isin([lower, upper])],
            "lower_boundary_extrapolation": data[data["heldout_scale"] == lower],
            "upper_boundary_extrapolation": data[data["heldout_scale"] == upper],
        }
        scale_acc = data.groupby("heldout_scale")["correct"].mean()
        worst_scale = float(scale_acc.idxmin())
        regimes[f"worst_scale_s={worst_scale:g}"] = data[data["heldout_scale"] == worst_scale]
        for regime_index, (regime, subset) in enumerate(regimes.items()):
            summary_rows.append(
                summarize_subset(
                    subset,
                    name,
                    regime,
                    args.n_bootstrap,
                    args.seed + 100 * set_index + regime_index,
                )
            )
        for scale_index, scale in enumerate(scales):
            subset = data[data["heldout_scale"] == scale]
            row = summarize_subset(
                subset,
                name,
                f"heldout_scale_{scale:g}",
                args.n_bootstrap,
                args.seed + 1000 + 100 * set_index + scale_index,
            )
            row["heldout_scale"] = float(scale)
            by_scale_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    by_scale = pd.DataFrame(by_scale_rows)
    summary.to_csv(args.out_dir / "table_interpolation_extrapolation_unique_conditions.csv", index=False)
    by_scale.to_csv(args.out_dir / "table_loso_by_scale_unique_conditions.csv", index=False)
    try:
        summary.to_markdown(args.out_dir / "table_interpolation_extrapolation_unique_conditions.md", index=False)
        by_scale.to_markdown(args.out_dir / "table_loso_by_scale_unique_conditions.md", index=False)
    except Exception:
        pass

    family_scale = (
        unique.groupby(["feature_family", "heldout_scale"], as_index=False)
        .agg(accuracy=("correct", "mean"), n_unique_conditions=("correct", "size"))
    )
    family_scale.to_csv(args.out_dir / "table_family_by_scale_unique_conditions.csv", index=False)

    # Re-score the two existing square-grid baselines on exactly the same
    # unique bar/target sample identifiers. Rows remain grouped by training
    # seed; variability is across seeds rather than duplicated stimuli.
    primary_sample_ids = set(
        meta.loc[
            (meta["repeat"] == 0)
            & meta["feature_family"].isin(["moving_bar", "small_translating_target"]),
            "sample",
        ].astype(int)
    )
    baseline_sources = {
        "TemporalResNet18Small_32x32_unmatched": sorted(
            (args.outputs_dir / "serious_cnn_baseline" / "predictions").glob("*.csv")
        ),
        "STN_CNN_24x24_unmatched": [args.outputs_dir / "stn_cnn_baseline" / "predictions_stn_cnn.csv"],
    }
    baseline_seed_rows = []
    for model_name, paths in baseline_sources.items():
        tables = [pd.read_csv(path) for path in paths if path.exists()]
        if not tables:
            continue
        table = pd.concat(tables, ignore_index=True)
        table = table[table["sample"].astype(int).isin(primary_sample_ids)]
        for seed_value, seed_table in table.groupby("seed"):
            baseline_seed_rows.append(
                {
                    "model": model_name,
                    "seed": int(seed_value),
                    "accuracy": float(seed_table["correct"].astype(bool).mean()),
                    "n_unique_conditions": int(seed_table["sample"].nunique()),
                }
            )
    baseline_by_seed = pd.DataFrame(baseline_seed_rows)
    baseline_summary = (
        baseline_by_seed.groupby("model", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            sd_across_seeds=("accuracy", "std"),
            min_seed_accuracy=("accuracy", "min"),
            max_seed_accuracy=("accuracy", "max"),
            n_seeds=("seed", "nunique"),
            n_unique_conditions_per_seed=("n_unique_conditions", "min"),
        )
    )
    baseline_by_seed.to_csv(args.out_dir / "table_unmatched_baselines_unique_conditions_by_seed.csv", index=False)
    baseline_summary.to_csv(args.out_dir / "table_unmatched_baselines_unique_conditions_summary.csv", index=False)

    coords = pd.read_csv(coord_path)
    mapping = pd.DataFrame([projection_audit(coords, 24), projection_audit(coords, 32)])
    mapping.to_csv(args.out_dir / "table_hex_to_grid_resolution_audit.csv", index=False)

    edge = meta[meta["feature_family"] == "moving_edge"]
    edge_reference = int(edge.iloc[0]["sample"])
    edge_scale_identical = True
    ref = edge.iloc[0]
    for scale in sorted(edge["scale"].unique()):
        row = edge[
            (edge["scale"] == scale)
            & (edge["angle"] == ref["angle"])
            & (edge["contrast"] == ref["contrast"])
            & (edge["repeat"] == ref["repeat"])
        ].iloc[0]
        edge_scale_identical &= bool(np.array_equal(stimuli[edge_reference], stimuli[int(row["sample"])]))

    report = {
        "all_dynamic_repeat_clusters": int(len(duplicate_audit)),
        "all_repeat_clusters_pixel_identical": bool(duplicate_audit["stimuli_pixel_identical"].all()),
        "all_repeat_cluster_predictions_identical": bool(duplicate_audit["predictions_identical"].all()),
        "recorded_dynamic_rows": int(len(dynamic_meta)),
        "unique_dynamic_rendered_conditions": int(len(unique)),
        "moving_edge_identical_across_scale_for_matched_conditions": bool(edge_scale_identical),
        "primary_scale_varying_family_set": ["moving_bar", "small_translating_target"],
        "interior_scales": list(map(float, interior)),
        "lower_boundary_scale": float(lower),
        "upper_boundary_scale": float(upper),
        "uncertainty_unit": "unique rendered condition; stratified by held-out scale and direction",
        "original_prediction_source": str(pred_path),
        "unmatched_baseline_table": str(args.out_dir / "table_unmatched_baselines_unique_conditions_summary.csv"),
    }
    (args.out_dir / "audit_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out_dir / "REPORT.md").write_text(
        "\n".join(
            [
                "# Unique-condition independence and evaluation-regime audit",
                "",
                f"- The legacy dynamic set contains {len(dynamic_meta)} rows but only {len(unique)} unique rendered conditions.",
                f"- Every one of the {len(duplicate_audit)} repeat clusters is pixel-identical: {bool(duplicate_audit['stimuli_pixel_identical'].all())}.",
                f"- Moving-edge movies are identical across the nominal scale labels for matched direction/contrast/repeat: {bool(edge_scale_identical)}.",
                "- The primary scale-varying analysis therefore uses moving bars and translating targets and resamples unique conditions.",
                "- Interior-scale interpolation, both boundary extrapolations, the upper boundary, and the worst scale are reported separately.",
                "- The full TemporalResNet run used 32x32, not 24x24; the 32x32 assignment has no sample collisions. The STN run used 24x24 and does have collisions.",
                "",
                "These corrections narrow the claim from full-range scale stability to strong interpolation with failed upper-bound extrapolation.",
            ]
        ),
        encoding="utf-8",
    )
    print(summary.round(4).to_string(index=False))
    print(mapping.to_string(index=False))


if __name__ == "__main__":
    main()
