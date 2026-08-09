#!/usr/bin/env python
"""Create the main analysis figures from saved result tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(base.with_suffix(f".{ext}"), dpi=260 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def fig2(audit_dir: Path, out: Path) -> None:
    df = pd.read_csv(audit_dir / "table_loso_by_scale_unique_conditions.csv")
    labels = {
        "scale_varying_moving_bar_and_target": "Scale-varying bars + targets (primary)",
        "complete_mixture_including_scale_neutral_edge": "Complete mixture (+ scale-neutral edge)",
    }
    colors = {
        "scale_varying_moving_bar_and_target": "#1f77b4",
        "complete_mixture_including_scale_neutral_edge": "#9aa0a6",
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for name in ["complete_mixture_including_scale_neutral_edge", "scale_varying_moving_bar_and_target"]:
        sub = df[df["analysis_set"] == name].sort_values("heldout_scale")
        yerr = np.vstack(
            [
                sub["accuracy"] - sub["wilson_95_low"],
                sub["wilson_95_high"] - sub["accuracy"],
            ]
        )
        ax.errorbar(
            sub["heldout_scale"],
            sub["accuracy"],
            yerr=yerr,
            marker="o",
            capsize=2.5,
            lw=1.7,
            color=colors[name],
            label=labels[name],
        )
    ax.axhline(1 / 6, color="#333333", linestyle="--", lw=1, label="Chance (1/6)")
    ax.axvspan(3, 16, color="#1f77b4", alpha=0.06, label="Interior interpolation range")
    ax.set_xticks([2, 3, 4, 6, 8, 12, 16, 24])
    ax.set_ylim(0, 1.06)
    ax.set_xlabel("Held-out apparent scale")
    ax.set_ylabel("Supervised LOSO direction accuracy")
    ax.set_title("Strong interior transfer, chance performance at the upper boundary")
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.20, 1, 1])
    save(fig, out / "fig2_scale_matrix")


def fig3(outputs: Path, out: Path) -> None:
    match = pd.read_csv(outputs / "matched_geometry_controls" / "table_matched_geometry_summary.csv")
    unmatched = pd.read_csv(
        outputs / "unique_condition_audit" / "table_unmatched_baselines_unique_conditions_summary.csv"
    ).set_index("model")
    name_map = {
        "hex_six_neighbor": "Matched hex\nneighbours",
        "square32_six_neighbor": "Matched square32\nneighbours",
        "self_only_no_neighbor_message": "Matched self-only\n(no message)",
    }
    match["label"] = match["variant"].map(name_map)
    fixed = pd.DataFrame(
        [
            {"label": "FlyVis linear\ntransfer", "value": 170 / 192, "err": 0.0, "kind": "FlyVis"},
            {
                "label": "TemporalResNet\n(unmatched)",
                "value": unmatched.loc["TemporalResNet18Small_32x32_unmatched", "mean_accuracy"],
                "err": 0.0,
                "kind": "Unmatched",
            },
            {
                "label": "STN-CNN 24x24\n(unmatched)",
                "value": unmatched.loc["STN_CNN_24x24_unmatched", "mean_accuracy"],
                "err": 0.0,
                "kind": "Unmatched",
            },
        ]
    )
    matched = pd.DataFrame(
        {
            "label": match["label"],
            "value": match["mean_accuracy"],
            "err": match["sd_across_seeds"],
            "kind": "Matched",
        }
    )
    data = pd.concat([matched, fixed], ignore_index=True)
    colors = {"Matched": "#2ca02c", "FlyVis": "#1f77b4", "Unmatched": "#b0b0b0"}
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    xpos = np.arange(len(data))
    ax.bar(xpos, data["value"], color=[colors[k] for k in data["kind"]], yerr=data["err"], capsize=3)
    ax.axhline(1 / 6, color="#333333", linestyle="--", lw=1)
    ax.set_xticks(xpos, data["label"], rotation=20, ha="right")
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("LOSO accuracy on unique bars + targets")
    ax.set_title("Matched temporal classifiers solve the task across neighbourhood operators")
    handles = [mpl.patches.Patch(color=colors[k], label=k) for k in ["Matched", "FlyVis", "Unmatched"]]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    save(fig, out / "fig3_controls")


def fig4(audit_dir: Path, out: Path) -> None:
    df = pd.read_csv(audit_dir / "table_family_by_scale_unique_conditions.csv")
    labels = {
        "moving_edge": "Moving edge (scale-neutral control)",
        "moving_bar": "Moving bar",
        "small_translating_target": "Translating target",
    }
    colors = {"moving_edge": "#9aa0a6", "moving_bar": "#1f77b4", "small_translating_target": "#ff7f0e"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw={"width_ratios": [1.7, 1.0]})
    for family in labels:
        sub = df[df["feature_family"] == family].sort_values("heldout_scale")
        marker = "o" if family != "small_translating_target" else "x"
        marker_size = 8 if family == "moving_bar" else 6
        line_style = "--" if family == "small_translating_target" else "-"
        axes[0].plot(
            sub["heldout_scale"],
            sub["accuracy"],
            marker=marker,
            markersize=marker_size,
            linestyle=line_style,
            lw=1.7,
            color=colors[family],
            label=labels[family],
        )
    axes[0].axhline(1 / 6, color="#333333", linestyle="--", lw=1)
    axes[0].set_xticks([2, 3, 4, 6, 8, 12, 16, 24])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlabel("Held-out apparent scale")
    axes[0].set_ylabel("LOSO accuracy")
    axes[0].set_title("A  Scale-varying family profiles", loc="left", fontweight="bold")
    family_handles, family_labels = axes[0].get_legend_handles_labels()

    aggregate = df.groupby("feature_family", as_index=False).apply(
        lambda x: pd.Series({"accuracy": np.average(x["accuracy"], weights=x["n_unique_conditions"])})
    ).reset_index(drop=True)
    order = list(labels)
    values = [float(aggregate.loc[aggregate["feature_family"] == family, "accuracy"].iloc[0]) for family in order]
    axes[1].bar(np.arange(len(order)), values, color=[colors[v] for v in order])
    axes[1].axhline(1 / 6, color="#333333", linestyle="--", lw=1)
    axes[1].set_xticks(np.arange(len(order)), ["Edge\ncontrol", "Moving\nbar", "Translating\ntarget"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("B  Unique-condition aggregate", loc="left", fontweight="bold")
    axes[1].set_ylabel("LOSO accuracy")
    fig.suptitle("Both genuinely scale-varying direction families share the upper-scale boundary")
    fig.legend(
        family_handles,
        family_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=7.2,
    )
    fig.tight_layout(rect=[0, 0.16, 1, 0.93])
    save(fig, out / "fig4_feature_family")


def fig5(outputs: Path, out: Path, tables: Path) -> None:
    source = pd.read_csv(outputs / "supplementary" / "supplementary_tables" / "tableS6_temporal_lesion_results.csv")
    keep = ["early_0_20pct", "middle_33_66pct", "late_66_100pct", "full_time_mean", "temporal_bins_5"]
    labels = {
        "early_0_20pct": "Early\n0--20%",
        "middle_33_66pct": "Middle\n33--66%",
        "late_66_100pct": "Late\n66--100%",
        "full_time_mean": "Full-time\nmean",
        "temporal_bins_5": "Five temporal\nbins",
    }
    source = source.set_index("feature_variant").loc[keep].reset_index()
    source["n_unique_conditions"] = 288
    source["n_correct"] = np.rint(source["accuracy"] * source["n_unique_conditions"]).astype(int)

    def interval(row):
        k, n = int(row["n_correct"]), int(row["n_unique_conditions"])
        z = 1.959963984540054
        p = k / n
        den = 1 + z * z / n
        center = (p + z * z / (2 * n)) / den
        radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
        return pd.Series({"wilson_95_low": center - radius, "wilson_95_high": center + radius})

    source = pd.concat([source, source.apply(interval, axis=1)], axis=1)
    tables.mkdir(parents=True, exist_ok=True)
    source[
        ["feature_variant", "accuracy", "n_correct", "n_unique_conditions", "wilson_95_low", "wilson_95_high"]
    ].to_csv(tables / "table_temporal_windows_unique_conditions.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    x = np.arange(len(source))
    yerr = np.vstack([source["accuracy"] - source["wilson_95_low"], source["wilson_95_high"] - source["accuracy"]])
    ax.bar(x, source["accuracy"], color=["#8c6bb1", "#2c7fb8", "#41b6c4", "#7fcdbb", "#225ea8"], yerr=yerr, capsize=3)
    ax.axhline(1 / 6, color="#333333", linestyle="--", lw=1, label="Chance")
    ax.set_xticks(x, [labels[v] for v in source["feature_variant"]])
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("LOSO direction accuracy")
    ax.set_title("Direction information becomes linearly accessible after the early response")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, out / "fig5_temporal")


def supplementary_celltype(outputs: Path, out: Path, tables: Path) -> None:
    source = pd.read_csv(outputs / "supplementary" / "supplementary_tables" / "tableS7_cell_group_ablation_results.csv")
    source = source[["ablation", "n_removed_cell_types", "accuracy", "full_accuracy", "drop_accuracy"]].copy()
    source.to_csv(tables / "table_response_feature_cellgroup_ablation_point_estimates.csv", index=False)
    plot = source.sort_values("drop_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(6.6, 3.7))
    ax.barh(plot["ablation"], plot["drop_accuracy"], color="#9467bd")
    ax.set_xlabel("Change in LOSO accuracy after response-feature removal")
    ax.set_title("Cell-group response-feature ablation: distributed linear information")
    ax.text(
        0.99,
        0.03,
        "Point estimates on the complete mixture;\nnot a direct circuit intervention",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    fig.tight_layout()
    save(fig, out / "figS1_cellgroup_response_feature_ablation")


def load_direct_runs(outputs: Path) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    runs = []
    for directory in [
        outputs / "direct_flyvis_perturbations",
        outputs / "direct_flyvis_perturbations_seed42",
        outputs / "direct_flyvis_perturbations_seed84",
    ]:
        table = directory / "table_direct_perturbation_summary.csv"
        info = directory / "run_info.json"
        if not table.exists() or not info.exists():
            continue
        run = json.loads(info.read_text(encoding="utf-8"))
        seed = int(run["args"]["seed"])
        frame = pd.read_csv(table)
        frame["shuffle_seed"] = seed
        frames.append(frame)
        stability_path = directory / "response_stability.csv"
        stability = pd.read_csv(stability_path).to_dict(orient="records") if stability_path.exists() else []
        runs.append({"directory": str(directory), "seed": seed, "stability": stability})
    if not frames:
        raise FileNotFoundError("No direct intervention summaries were found")
    return pd.concat(frames, ignore_index=True), runs


def fig6(outputs: Path, out: Path, tables: Path) -> None:
    df, runs = load_direct_runs(outputs)
    df = df[
        (df["analysis_set"] == "scale_varying_moving_bar_and_target")
        & (df["regime"] == "all_scales")
    ].copy()
    variant_order = ["full", "t4t5_state_silencing", "spatial_kernel_shuffle", "type_pair_strength_shuffle"]
    decoder_order = ["refit_within_variant", "fixed_intact_decoder"]
    labels = {
        "full": "Intact",
        "t4t5_state_silencing": "T4/T5 state\nsilencing",
        "spatial_kernel_shuffle": "Spatial-kernel\nshuffle",
        "type_pair_strength_shuffle": "Type-pair strength\nshuffle",
    }
    colors = {"refit_within_variant": "#1f77b4", "fixed_intact_decoder": "#d62728"}
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    width = 0.34
    base = np.arange(len(variant_order))
    for di, decoder in enumerate(decoder_order):
        means = []
        sds = []
        point_groups = []
        for variant in variant_order:
            vals = df[(df["variant"] == variant) & (df["decoder"] == decoder)][["shuffle_seed", "accuracy"]].drop_duplicates()
            means.append(vals["accuracy"].mean())
            sds.append(vals["accuracy"].std(ddof=1) if len(vals) > 1 else 0.0)
            point_groups.append(vals["accuracy"].to_numpy())
        x = base + (di - 0.5) * width
        ax.bar(x, means, width=width, color=colors[decoder], alpha=0.78, yerr=sds, capsize=3, label="Refit decoder" if di == 0 else "Fixed intact decoder")
        for xi, vals in zip(x, point_groups):
            if len(vals):
                jitter = np.linspace(-0.06, 0.06, len(vals))
                ax.scatter(np.full(len(vals), xi) + jitter, vals, s=18, color="black", zorder=4)
    ax.axhline(1 / 6, color="#333333", linestyle="--", lw=1, label="Chance")
    ax.set_xticks(base, [labels[v] for v in variant_order])
    ax.set_ylim(0, 1.06)
    ax.set_ylabel("LOSO accuracy on unique bars + targets")
    ax.set_title("Acute FlyVis interventions: available information vs intact-code preservation")
    unstable_strength = sum(
        1
        for run in runs
        for row in run["stability"]
        if row["variant"] == "type_pair_strength_shuffle" and not bool(row["finite_response"])
    )
    if unstable_strength:
        ax.text(
            0.99,
            0.97,
            f"Learned-strength shuffle: {unstable_strength}/3 run non-finite",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#7f0000",
        )
    handles, legend_labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.15, 1, 1])
    save(fig, out / "fig6_ablation")

    finite_summary = (
        df.groupby(["variant", "decoder"], as_index=False)
        .agg(
            n_finite_runs=("accuracy", "count"),
            mean_accuracy=("accuracy", "mean"),
            sd_accuracy=("accuracy", "std"),
            min_accuracy=("accuracy", "min"),
            max_accuracy=("accuracy", "max"),
        )
    )
    stability_rows = [
        {**row, "run_seed": run["seed"]}
        for run in runs
        for row in run["stability"]
    ]
    tables.mkdir(parents=True, exist_ok=True)
    finite_summary.to_csv(tables / "table_direct_intervention_finite_runs.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(tables / "table_direct_intervention_stability.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=ROOT / "figures",
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=ROOT / "outputs" / "analysis_tables",
    )
    parser.add_argument(
        "--supplementary-figures-dir",
        type=Path,
        default=ROOT / "figures" / "supplementary",
    )
    args = parser.parse_args()
    style()
    fig2(args.outputs_dir / "unique_condition_audit", args.figures_dir)
    fig3(args.outputs_dir, args.figures_dir)
    fig4(args.outputs_dir / "unique_condition_audit", args.figures_dir)
    fig5(args.outputs_dir, args.figures_dir, args.tables_dir)
    fig6(args.outputs_dir, args.figures_dir, args.tables_dir)
    supplementary_celltype(args.outputs_dir, args.supplementary_figures_dir, args.tables_dir)
    print(f"Wrote analysis figures to {args.figures_dir}")


if __name__ == "__main__":
    main()
