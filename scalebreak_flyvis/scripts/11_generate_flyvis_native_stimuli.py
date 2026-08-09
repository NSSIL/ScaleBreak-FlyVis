#!/usr/bin/env python
"""Generate FlyVis-native Pilot v2 stimuli.

The output movie tensor follows FlyVis input convention:
    (sample, frame, channel, hex_pixel)

The stimulus values are luminance-like inputs on a grey 0.5 background. Apparent
scale is controlled in the retinal/hex coordinate frame; it is not physical
distance.
"""

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


def parse_numbers(text: str, cast=float) -> list[Any]:
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def flyvis_input_coordinates(flyvis_root: Path, model: str) -> tuple[np.ndarray, np.ndarray]:
    os.environ.setdefault("FLYVIS_ROOT_DIR", str(flyvis_root))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    from flyvis import results_dir
    from flyvis.network import NetworkView

    network_view = NetworkView(results_dir / model)
    cell_types = network_view.connectome.nodes.type[:].astype(str)
    u = network_view.connectome.nodes.u[:]
    v = network_view.connectome.nodes.v[:]
    r1 = np.flatnonzero(cell_types == "R1")
    # Convert axial-like u/v coordinates to a planar coordinate frame for
    # rendering geometry. The ordering remains exactly the FlyVis R1 input order.
    x = u[r1].astype(np.float32) + 0.5 * v[r1].astype(np.float32)
    y = (np.sqrt(3.0) / 2.0 * v[r1].astype(np.float32)).astype(np.float32)
    return x, y


def rotate_coords(x: np.ndarray, y: np.ndarray, angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    theta = np.deg2rad(angle_deg)
    along = x * np.cos(theta) + y * np.sin(theta)
    perp = -x * np.sin(theta) + y * np.cos(theta)
    return along, perp


def shape_mask(
    family: str,
    shape: str,
    scale: float,
    angle: float,
    x: np.ndarray,
    y: np.ndarray,
    t_frac: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> np.ndarray:
    # Position is a nuisance variable.  Applying the offset before rotating the
    # coordinates translates the complete trajectory without changing its
    # direction label.
    x = x - float(offset_x)
    y = y - float(offset_y)
    along, perp = rotate_coords(x, y, angle)
    field = 23.0
    if family == "moving_edge":
        pos = -field + 2 * field * t_frac
        return along <= pos
    if family == "moving_bar":
        pos = -field + 2 * field * t_frac
        return (np.abs(along - pos) <= max(scale, 1.0) / 2.0) & (np.abs(perp) <= 18.0)
    if family == "small_translating_target":
        pos = -field + 2 * field * t_frac
        return (along - pos) ** 2 + perp**2 <= (max(scale, 1.0) / 2.0) ** 2
    if family == "looming_disk":
        radius = max(0.75, scale * (0.25 + 0.75 * t_frac))
        # Angle controls a small approach offset, so angle is not a motion
        # direction claim here; it is a retinal-position nuisance.
        cx = 4.0 * (1.0 - t_frac) * np.cos(np.deg2rad(angle))
        cy = 4.0 * (1.0 - t_frac) * np.sin(np.deg2rad(angle))
        return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2
    if family == "static_shape":
        if shape == "disk":
            return x**2 + y**2 <= (max(scale, 1.0) / 2.0) ** 2
        xr, yr = rotate_coords(x, y, angle)
        if shape == "square":
            return (np.abs(xr) <= scale / 2.0) & (np.abs(yr) <= scale / 2.0)
        if shape == "triangle":
            half = scale / 2.0
            row_frac = np.clip((yr + half) / max(scale, 1e-6), 0, 1)
            return (yr >= -half) & (yr <= half) & (np.abs(xr) <= row_frac * half)
    raise ValueError(f"Unsupported family/shape: {family}/{shape}")


def render_movie(
    family: str,
    shape: str,
    scale: float,
    angle: float,
    contrast: float,
    x: np.ndarray,
    y: np.ndarray,
    n_frames: int,
    t_pre_frames: int,
    active_frames: int,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
    seed: int | None = None,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    movie = np.full((n_frames, 1, len(x)), 0.5, dtype=np.float32)
    target = 0.5 + 0.5 * float(contrast)
    area_accum = []
    for frame in range(n_frames):
        if frame < t_pre_frames:
            t_frac = 0.0
            active = family == "static_shape"
        else:
            denom = max(active_frames - 1, 1)
            t_frac = min(1.0, max(0.0, (frame - t_pre_frames) / denom))
            active = True
        if active:
            mask = shape_mask(
                family,
                shape,
                scale,
                angle,
                x,
                y,
                t_frac,
                offset_x=offset_x,
                offset_y=offset_y,
            )
            movie[frame, 0, mask] = target
            area_accum.append(float(mask.sum()))
    if noise_std > 0:
        # The original submitted generator recorded a different seed for each
        # repeat but never used it, making all ten repeats pixel-identical.  A
        # small seeded luminance perturbation makes repeats genuine independent
        # renderings while preserving the controlled label variables.
        noise_rng = np.random.default_rng(seed)
        movie += noise_rng.normal(0.0, noise_std, size=movie.shape).astype(np.float32)
        np.clip(movie, 0.0, 1.0, out=movie)
    return movie, {
        "mean_area_hex": float(np.mean(area_accum)) if area_accum else 0.0,
        "max_area_hex": float(np.max(area_accum)) if area_accum else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flyvis-root", type=Path, default=ROOT / "flyvis_data")
    parser.add_argument("--model", default="flow/0000/000")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "flyvis_pilot_v2" / "stimuli")
    parser.add_argument("--scales", default="2,3,4,6,8,12,16,24")
    parser.add_argument("--angles", default="0,60,120,180,240,300")
    parser.add_argument("--contrasts", default="1.0,0.3")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-frames", type=int, default=165)
    parser.add_argument("--dt", type=float, default=1 / 200)
    parser.add_argument("--t-pre", type=float, default=0.2)
    parser.add_argument("--active-frames", type=int, default=86)
    parser.add_argument(
        "--position-jitter",
        type=float,
        default=3.0,
        help="Maximum absolute x/y translation sampled independently per repeat. Use 0 to reproduce the legacy centered stimuli.",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Seeded per-pixel luminance-noise SD. Use 0 to reproduce the legacy deterministic repeats.",
    )
    parser.add_argument("--include-static-angles", action="store_true", help="Use all angles for static appendix controls.")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    flyvis_root = args.flyvis_root.resolve()
    x, y = flyvis_input_coordinates(flyvis_root, args.model)

    scales = parse_numbers(args.scales, float)
    angles = parse_numbers(args.angles, float)
    contrasts = parse_numbers(args.contrasts, float)
    rng = np.random.default_rng(args.seed)

    dynamic = [
        ("moving_edge", "edge"),
        ("moving_bar", "bar"),
        ("looming_disk", "disk"),
        ("small_translating_target", "target"),
    ]
    static = [("static_shape", "disk"), ("static_shape", "square"), ("static_shape", "triangle")]
    rows: list[dict[str, Any]] = []
    for family, shape in dynamic:
        for scale in scales:
            for angle in angles:
                for contrast in contrasts:
                    for repeat in range(args.repeats):
                        trial_seed = int(rng.integers(0, 2**31 - 1))
                        trial_rng = np.random.default_rng(trial_seed)
                        rows.append(
                            {
                                "sample": len(rows),
                                "trial_id": f"flyvis_v2_{len(rows):06d}",
                                "feature_family": family,
                                "shape": shape,
                                "scale": scale,
                                "apparent_scale": scale,
                                "angle": angle,
                                "direction": angle,
                                "contrast": contrast,
                                "repeat": repeat,
                                "seed": trial_seed,
                                "offset_x": float(trial_rng.uniform(-args.position_jitter, args.position_jitter)),
                                "offset_y": float(trial_rng.uniform(-args.position_jitter, args.position_jitter)),
                                "noise_std": float(args.noise_std),
                                "scale_varies_geometry": family != "moving_edge",
                                "appendix_control": False,
                                "dynamic": True,
                            }
                        )
    static_angles = angles if args.include_static_angles else [0.0]
    for family, shape in static:
        for scale in scales:
            for angle in static_angles:
                for contrast in contrasts:
                    for repeat in range(args.repeats):
                        trial_seed = int(rng.integers(0, 2**31 - 1))
                        trial_rng = np.random.default_rng(trial_seed)
                        rows.append(
                            {
                                "sample": len(rows),
                                "trial_id": f"flyvis_v2_{len(rows):06d}",
                                "feature_family": family,
                                "shape": shape,
                                "scale": scale,
                                "apparent_scale": scale,
                                "angle": angle,
                                "direction": np.nan,
                                "contrast": contrast,
                                "repeat": repeat,
                                "seed": trial_seed,
                                "offset_x": float(trial_rng.uniform(-args.position_jitter, args.position_jitter)),
                                "offset_y": float(trial_rng.uniform(-args.position_jitter, args.position_jitter)),
                                "noise_std": float(args.noise_std),
                                "scale_varies_geometry": True,
                                "appendix_control": True,
                                "dynamic": False,
                            }
                        )

    metadata = pd.DataFrame(rows)
    stim_path = out_dir / "stimuli.npy"
    stimuli = np.lib.format.open_memmap(
        stim_path, mode="w+", dtype=np.float32, shape=(len(metadata), args.n_frames, 1, len(x))
    )
    t_pre_frames = int(round(args.t_pre / args.dt))
    extras = []
    for i, row in metadata.iterrows():
        movie, extra = render_movie(
            row["feature_family"],
            row["shape"],
            float(row.scale),
            float(row.angle),
            float(row.contrast),
            x,
            y,
            args.n_frames,
            t_pre_frames,
            args.active_frames,
            offset_x=float(row.offset_x),
            offset_y=float(row.offset_y),
            seed=int(row.seed),
            noise_std=float(row.noise_std),
        )
        stimuli[i] = movie
        extras.append(extra)
    stimuli.flush()
    metadata = pd.concat([metadata, pd.DataFrame(extras)], axis=1)
    metadata["n_frames"] = args.n_frames
    metadata["hex_pixels"] = len(x)
    metadata["channel"] = 1
    metadata["dt"] = args.dt
    metadata["t_pre"] = args.t_pre
    metadata.to_csv(out_dir / "metadata.csv", index=False)
    pd.DataFrame({"hex_pixel": np.arange(len(x)), "x": x, "y": y}).to_csv(out_dir / "hex_coordinates.csv", index=False)
    write_json(
        {
            "model": args.model,
            "stimuli_path": str(stim_path),
            "metadata_path": str(out_dir / "metadata.csv"),
            "shape": [len(metadata), args.n_frames, 1, len(x)],
            "n_trials": int(len(metadata)),
            "seed": args.seed,
            "feature_families": sorted(metadata.feature_family.unique()),
            "position_jitter": args.position_jitter,
            "noise_std": args.noise_std,
            "note": (
                "Apparent scale is rendered in retinal hex-pixel coordinates. Moving-edge geometry is scale-neutral and is marked "
                "scale_varies_geometry=False. Repeat seeds control position jitter and luminance noise; set both amplitudes to zero "
                "only for legacy deterministic-repeat reproduction."
            ),
        },
        out_dir / "stimulus_manifest.json",
    )
    print(f"Wrote {len(metadata)} FlyVis-native stimuli to {stim_path} with shape {stimuli.shape}")


if __name__ == "__main__":
    main()
