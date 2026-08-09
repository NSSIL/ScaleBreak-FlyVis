#!/usr/bin/env python
"""Capacity-, sampling-, and training-matched geometry controls.

All variants receive the same 721 raw samples, the same 165-frame sequences,
the same labels/splits, and the same temporal network.  The only changed input
is the fixed six-neighbour aggregation:

* native hex-coordinate neighbours;
* neighbours after a collision-free 32x32 coordinate assignment; or
* self-only aggregation (no neighbour message).

The analysis uses one copy of each legacy deterministic condition and omits
moving edges, whose rendered movies do not change with the nominal scale.
This experiment isolates the role of the neighbourhood graph for this
classifier; it does not match FlyVis optical-flow pretraining.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]


def parse_ints(text: str) -> list[int]:
    return [int(v.strip()) for v in text.split(",") if v.strip()]


def nearest_neighbors(xy: np.ndarray, k: int = 6) -> np.ndarray:
    dist = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=-1)
    order = np.argsort(dist, axis=1)
    return order[:, 1 : k + 1].astype(np.int64)


def neighbor_variants(coords: pd.DataFrame) -> dict[str, np.ndarray]:
    xy = coords[["x", "y"]].to_numpy(dtype=np.float32)
    gx = np.round((xy[:, 0] - xy[:, 0].min()) / np.ptp(xy[:, 0]) * 31).astype(np.float32)
    gy = np.round((xy[:, 1] - xy[:, 1].min()) / np.ptp(xy[:, 1]) * 31).astype(np.float32)
    if len(set(zip(gx.tolist(), gy.tolist()))) != len(coords):
        raise RuntimeError("The 32x32 projection must be collision-free for the matched control.")
    self_idx = np.repeat(np.arange(len(coords), dtype=np.int64)[:, None], 6, axis=1)
    return {
        "hex_six_neighbor": nearest_neighbors(xy, 6),
        "square32_six_neighbor": nearest_neighbors(np.column_stack([gx, gy]), 6),
        "self_only_no_neighbor_message": self_idx,
    }


def make_model(n_pixels: int, n_classes: int, neighbor_idx: np.ndarray, hidden: int, dropout: float):
    import torch
    import torch.nn as nn

    class MatchedTemporalGraph(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("neighbor_idx", torch.tensor(neighbor_idx, dtype=torch.long))
            self.temporal = nn.Sequential(
                nn.Conv1d(n_pixels * 2, hidden, kernel_size=7, padding=3, bias=False),
                nn.GroupNorm(8, hidden),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, bias=False),
                nn.GroupNorm(8, hidden),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(8, hidden),
                nn.ReLU(inplace=True),
            )
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, n_classes))

        def forward(self, x):
            neigh = x[:, :, self.neighbor_idx].mean(dim=-1)
            z = torch.cat([x, neigh], dim=-1).transpose(1, 2)
            h = self.temporal(z)
            return self.head(torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1))

    return MatchedTemporalGraph()


def split_train_val(train_idx: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    a, b = next(splitter.split(train_idx, y[train_idx]))
    return train_idx[a], train_idx[b]


def train_fold(x, y, train_idx, test_idx, seed, args, neighbors):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    model = make_model(x.shape[-1], int(y.max() + 1), neighbors, args.hidden, args.dropout)
    n_params = int(sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tr_idx, val_idx = split_train_val(train_idx, y, seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(x[tr_idx], dtype=torch.float32), torch.tensor(y[tr_idx], dtype=torch.long)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_x = torch.tensor(x[val_idx], dtype=torch.float32)
    val_y = y[val_idx]
    loss_fn = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = None
    stale = 0
    curves = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x).argmax(1).cpu().numpy()
        val_acc = float((val_pred == val_y).mean())
        curves.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_accuracy": val_acc})
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x[test_idx], dtype=torch.float32))
        prob = torch.softmax(logits, dim=1).cpu().numpy()
    return prob.argmax(axis=1), prob, best_acc, len(curves), n_params, curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "matched_geometry_controls")
    parser.add_argument("--seeds", default="42,84,123")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stimuli = np.load(args.outputs_dir / "flyvis_pilot_v2" / "stimuli" / "stimuli.npy", mmap_mode="r")
    meta = pd.read_csv(args.outputs_dir / "flyvis_pilot_v2" / "responses" / "metadata.csv")
    coords = pd.read_csv(args.outputs_dir / "flyvis_pilot_v2" / "stimuli" / "hex_coordinates.csv")
    keep = (meta["repeat"] == 0) & meta["feature_family"].isin(["moving_bar", "small_translating_target"])
    source_ids = np.flatnonzero(keep.to_numpy())
    x = np.asarray(stimuli[source_ids, :, 0, :], dtype=np.float32) - 0.5
    selected = meta.iloc[source_ids].reset_index(drop=True)
    le = LabelEncoder()
    y = le.fit_transform(selected["direction"].astype(str)).astype(np.int64)
    scales = selected["scale"].to_numpy(dtype=float)
    variants = neighbor_variants(coords)

    pd.DataFrame(
        [
            {
                "variant": name,
                "n_nodes": int(idx.shape[0]),
                "neighbors_per_node": int(idx.shape[1]),
                "input_values_identical_across_variants": True,
                "uses_unique_conditions_only": True,
            }
            for name, idx in variants.items()
        ]
    ).to_csv(args.out_dir / "table_design_match.csv", index=False)
    for name, idx in variants.items():
        pd.DataFrame(idx).to_csv(args.out_dir / f"neighbors_{name}.csv", index=False)

    rows = []
    pred_rows = []
    curve_rows = []
    seeds = parse_ints(args.seeds)
    for variant, neighbors in variants.items():
        for seed in seeds:
            for heldout in sorted(np.unique(scales)):
                train_idx = np.flatnonzero(scales != heldout)
                test_idx = np.flatnonzero(scales == heldout)
                fold_seed = seed + int(heldout)
                pred, prob, best_val, epochs_ran, n_params, curves = train_fold(
                    x, y, train_idx, test_idx, fold_seed, args, neighbors
                )
                rows.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "heldout_scale": heldout,
                        "accuracy": float(accuracy_score(y[test_idx], pred)),
                        "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], pred)),
                        "macro_f1": float(f1_score(y[test_idx], pred, average="macro", zero_division=0)),
                        "best_val_accuracy": best_val,
                        "epochs_ran": epochs_ran,
                        "n_parameters": n_params,
                        "n_train_unique_conditions": len(train_idx),
                        "n_test_unique_conditions": len(test_idx),
                    }
                )
                for curve in curves:
                    curve_rows.append({"variant": variant, "seed": seed, "heldout_scale": heldout, **curve})
                for relative, true, pp in zip(test_idx, y[test_idx], prob):
                    rec = {
                        "variant": variant,
                        "seed": seed,
                        "source_sample": int(source_ids[relative]),
                        "heldout_scale": heldout,
                        "true_label": str(le.inverse_transform([true])[0]),
                        "pred_label": str(le.inverse_transform([int(pp.argmax())])[0]),
                        "correct": bool(int(pp.argmax()) == true),
                    }
                    pred_rows.append(rec)

    by_scale = pd.DataFrame(rows)
    predictions = pd.DataFrame(pred_rows)
    curves = pd.DataFrame(curve_rows)
    by_scale.to_csv(args.out_dir / "table_matched_geometry_by_seed_scale.csv", index=False)
    predictions.to_csv(args.out_dir / "predictions_matched_geometry.csv", index=False)
    curves.to_csv(args.out_dir / "training_curves.csv", index=False)

    by_seed = predictions.groupby(["variant", "seed"], as_index=False)["correct"].mean().rename(columns={"correct": "accuracy"})
    summary = (
        by_seed.groupby("variant", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            sd_across_seeds=("accuracy", "std"),
            min_seed_accuracy=("accuracy", "min"),
            max_seed_accuracy=("accuracy", "max"),
            n_seeds=("seed", "nunique"),
        )
    )
    params = by_scale.groupby("variant", as_index=False)["n_parameters"].first()
    summary = summary.merge(params, on="variant", how="left")
    summary.to_csv(args.out_dir / "table_matched_geometry_summary.csv", index=False)
    try:
        summary.to_markdown(args.out_dir / "table_matched_geometry_summary.md", index=False)
    except Exception:
        pass

    (args.out_dir / "run_info.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "python": platform.python_version(),
                "runtime_seconds": time.time() - t0,
                "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "stimulus_shape_used": list(x.shape),
                "n_unique_scale_varying_conditions": len(x),
                "classes": list(map(str, le.classes_)),
                "claim_boundary": "Isolates neighborhood geometry in a supervised direction classifier; does not match optical-flow pretraining.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.out_dir / "REPORT.md").write_text(
        "\n".join(
            [
                "# Matched geometry control",
                "",
                "All variants use the same 721 samples, full temporal sequence, parameter count, training budget, LOSO folds, and unique scale-varying conditions.",
                "Only the fixed neighborhood aggregation changes. This isolates neighborhood geometry for the supervised classifier, not FlyVis pretraining or connectome causality.",
                "",
                "```csv",
                summary.round(4).to_csv(index=False).strip(),
                "```",
            ]
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
