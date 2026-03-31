#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BACKEND_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from featurewind.analysis.seeds_visual_sensitivity import (  # noqa: E402
    _load_featurewind_case,
    build_condition_context,
    build_static_trail,
    SensitivityCondition,
)


FIELD_ARROW_COLOR = "#475569"
POINT_COLOR = "#d1d5db"
POINT_EDGE = "#334155"
STATIC_COLOR = "#96360e"
ANIM_COLOR = "#2563eb"
RESPAWN_COLOR = "#d97706"
BACKGROUND_COLOR = "#ffffff"
GRIDLINE_COLOR = "#000000"
MASKED_CELL_COLOR = "#d4d4d8"
DEMO_GRID_RES = 8

FIXED_ADVECTION_STEP_SEC = 1.0 / 60.0
MAX_PARTICLE_LIFETIME_SEC = 8.0
ANIMATED_TRAIL_LENGTH_PX = 140.0
PARTICLE_COUNT = 18
CAPTURE_STEPS = (0, 18, 54)


@dataclass
class Particle:
    x: float
    y: float
    age_sec: float
    hist: list[tuple[float, float]]
    init_x: float
    init_y: float
    respawned: bool = False


def _sample_quiver(context) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_x = context.grid_x
    sample_y = context.grid_y
    sample_u = context.u_grid
    sample_v = context.v_grid
    sample_mask = context.unmasked
    magnitude = np.hypot(sample_u, sample_v)
    valid = sample_mask & (magnitude >= max(context.weak_threshold, 1e-9))
    if not np.any(valid):
        return (
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
        )

    arrow_length = 0.44 * min(
        (context.bbox[1] - context.bbox[0]) / context.grid_res,
        (context.bbox[3] - context.bbox[2]) / context.grid_res,
    )
    unit_u = np.zeros_like(sample_u)
    unit_v = np.zeros_like(sample_v)
    unit_u[valid] = sample_u[valid] / np.maximum(magnitude[valid], 1e-9)
    unit_v[valid] = sample_v[valid] / np.maximum(magnitude[valid], 1e-9)
    return (
        sample_x[valid],
        sample_y[valid],
        unit_u[valid] * arrow_length,
        unit_v[valid] * arrow_length,
    )


def _path_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def choose_demo_point(case, context) -> tuple[int, np.ndarray]:
    center = np.asarray(
        [
            0.5 * (case.bbox[0] + case.bbox[1]),
            0.5 * (case.bbox[2] + case.bbox[3]),
        ],
        dtype=float,
    )
    best_index = 0
    best_trail = None
    best_score = None
    for point_index, (x, y) in enumerate(case.positions):
        trail = build_static_trail(context, start_x=float(x), start_y=float(y), point_index=point_index, max_steps=120)
        if not trail.valid or trail.point_count < 10:
            continue
        path = np.asarray(trail.points, dtype=float)
        length = _path_length(path)
        distance_to_center = float(np.linalg.norm(np.asarray([x, y], dtype=float) - center))
        score = (length, -distance_to_center, trail.point_count)
        if best_score is None or score > best_score:
            best_index = point_index
            best_trail = path
            best_score = score
    if best_trail is None:
        x, y = case.positions[0]
        trail = build_static_trail(context, start_x=float(x), start_y=float(y), point_index=0, max_steps=120)
        best_index = 0
        best_trail = np.asarray(trail.points, dtype=float)
    return best_index, best_trail


def _add_arrowhead(ax, start: np.ndarray, end: np.ndarray, color: str, linewidth: float, mutation_scale: float, zorder: int) -> None:
    patch = FancyArrowPatch(
        posA=(float(start[0]), float(start[1])),
        posB=(float(end[0]), float(end[1])),
        arrowstyle="-|>",
        mutation_scale=float(mutation_scale),
        linewidth=float(linewidth),
        color=color,
        shrinkA=0.0,
        shrinkB=0.0,
        zorder=zorder,
    )
    ax.add_patch(patch)


def draw_field_background(ax, case, context) -> None:
    ax.set_facecolor(BACKGROUND_COLOR)
    xmin, xmax, ymin, ymax = case.bbox
    dx = (xmax - xmin) / context.grid_res
    dy = (ymax - ymin) / context.grid_res
    x_edges = np.linspace(xmin, xmax, context.grid_res + 1)
    y_edges = np.linspace(ymin, ymax, context.grid_res + 1)
    for i in range(context.grid_res):
        for j in range(context.grid_res):
            if bool(context.unmasked[i, j]):
                continue
            ax.add_patch(
                Rectangle(
                    (xmin + j * dx, ymin + i * dy),
                    dx,
                    dy,
                    facecolor=MASKED_CELL_COLOR,
                    edgecolor="none",
                    zorder=0,
                )
            )
    for x_edge in x_edges:
        ax.axvline(x_edge, color=GRIDLINE_COLOR, linewidth=1.05, alpha=1.0, zorder=0.4)
    for y_edge in y_edges:
        ax.axhline(y_edge, color=GRIDLINE_COLOR, linewidth=1.05, alpha=1.0, zorder=0.4)
    ax.scatter(
        case.positions[:, 0],
        case.positions[:, 1],
        s=42,
        color=POINT_COLOR,
        edgecolors=POINT_EDGE,
        linewidths=0.8,
        alpha=0.95,
        zorder=2,
    )
    qx, qy, qu, qv = _sample_quiver(context)
    if qx.size > 0:
        ax.quiver(
            qx,
            qy,
            qu,
            qv,
            angles="xy",
            scale_units="xy",
            scale=1.0,
            color=FIELD_ARROW_COLOR,
            alpha=0.95,
            width=0.0046,
            headwidth=4.4,
            headlength=5.0,
            headaxislength=4.0,
            zorder=1,
        )
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_static_generation(case, context, trail: np.ndarray, point_index: int, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.03, wspace=0.08)

    sample_indices = np.unique(
        np.linspace(0, trail.shape[0] - 1, min(7, trail.shape[0]), dtype=int)
    )
    sampled = trail[sample_indices]
    segment_colors = np.linspace(0.35, 0.95, max(1, sampled.shape[0] - 1))

    draw_field_background(axes[0], case, context)
    axes[0].scatter(
        [trail[0, 0]],
        [trail[0, 1]],
        s=190,
        facecolors="white",
        edgecolors=STATIC_COLOR,
        linewidths=2.2,
        zorder=5,
    )

    draw_field_background(axes[1], case, context)
    for idx in range(sampled.shape[0] - 1):
        a = sampled[idx]
        b = sampled[idx + 1]
        alpha = float(segment_colors[idx])
        axes[1].plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=STATIC_COLOR,
            linewidth=3.0,
            alpha=alpha,
            zorder=5,
            solid_capstyle="round",
        )
        _add_arrowhead(
            axes[1],
            a,
            b,
            STATIC_COLOR,
            linewidth=1.8,
            mutation_scale=11.5,
            zorder=6,
        )
        axes[1].scatter(
            [a[0]],
            [a[1]],
            s=110,
            color=(1.0, 1.0, 1.0, 0.98),
            edgecolors=STATIC_COLOR,
            linewidths=1.6,
            zorder=6,
        )
    axes[1].scatter(
        [sampled[-1, 0]],
        [sampled[-1, 1]],
        s=100,
        color=STATIC_COLOR,
        edgecolors="white",
        linewidths=1.0,
        zorder=7,
    )

    draw_field_background(axes[2], case, context)
    axes[2].plot(trail[:, 0], trail[:, 1], color="white", linewidth=6.0, zorder=5, solid_capstyle="round")
    axes[2].plot(trail[:, 0], trail[:, 1], color=STATIC_COLOR, linewidth=3.1, zorder=6, solid_capstyle="round")
    axes[2].scatter(
        [trail[0, 0]],
        [trail[0, 1]],
        s=190,
        facecolors="white",
        edgecolors=STATIC_COLOR,
        linewidths=2.2,
        zorder=7,
    )
    if trail.shape[0] >= 2:
        _add_arrowhead(axes[2], trail[-2], trail[-1], STATIC_COLOR, linewidth=2.4, mutation_scale=16.0, zorder=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_valid_spawn_cells(context) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(context.grid_res):
        for j in range(context.grid_res):
            if not context.unmasked[i, j]:
                continue
            mag = math.hypot(float(context.u_grid[i, j]), float(context.v_grid[i, j]))
            if mag >= float(context.weak_threshold):
                cells.append((i, j))
    return cells


def spawn_point_in_cell(context, cell: tuple[int, int], rng: np.random.Generator) -> tuple[float, float]:
    i, j = cell
    xmin, xmax, ymin, ymax = context.bbox
    dx = (xmax - xmin) / context.grid_res
    dy = (ymax - ymin) / context.grid_res
    return (
        float(xmin + j * dx + rng.random() * dx),
        float(ymin + i * dy + rng.random() * dy),
    )


def is_masked_at(context, x: float, y: float) -> bool:
    cell = context.world_to_cell(float(x), float(y))
    if cell is None:
        return True
    if context.is_masked_cell(*cell):
        return True
    sample = context.sample_active_field(float(x), float(y))
    return not bool(sample["valid"])


def trim_history_by_screen_length(context, hist: list[tuple[float, float]]) -> None:
    if len(hist) <= 2:
        return
    max_trail_px = ANIMATED_TRAIL_LENGTH_PX
    xmin, xmax, ymin, ymax = context.bbox
    sx_scale = context.canvas_px / (xmax - xmin)
    sy_scale = context.canvas_px / (ymax - ymin)
    cumulative = 0.0
    keep_length = len(hist)
    for idx in range(len(hist) - 1):
        x0, y0 = hist[idx]
        x1, y1 = hist[idx + 1]
        sx0 = (x0 - xmin) * sx_scale
        sy0 = (y0 - ymin) * sy_scale
        sx1 = (x1 - xmin) * sx_scale
        sy1 = (y1 - ymin) * sy_scale
        cumulative += math.hypot(sx1 - sx0, sy1 - sy0)
        if cumulative > max_trail_px:
            keep_length = max(2, idx + 2)
            break
    if len(hist) > keep_length:
        del hist[keep_length:]


def reset_particle(particle: Particle, x: float, y: float) -> None:
    particle.x = float(x)
    particle.y = float(y)
    particle.age_sec = 0.0
    particle.init_x = float(x)
    particle.init_y = float(y)
    particle.hist = [(float(x), float(y))]
    particle.respawned = True


def initialize_particles(context, rng: np.random.Generator) -> list[Particle]:
    valid_cells = build_valid_spawn_cells(context)
    particles: list[Particle] = []
    for _ in range(PARTICLE_COUNT):
        cell = valid_cells[int(rng.integers(0, len(valid_cells)))]
        x, y = spawn_point_in_cell(context, cell, rng)
        particles.append(Particle(x=x, y=y, age_sec=0.0, hist=[(x, y)], init_x=x, init_y=y))
    return particles


def simulate_particle_snapshots(context) -> dict[int, list[Particle]]:
    rng = np.random.default_rng(7)
    valid_cells = build_valid_spawn_cells(context)
    particles = initialize_particles(context, rng)
    snapshots = {
        0: [
            Particle(
                x=particle.x,
                y=particle.y,
                age_sec=particle.age_sec,
                hist=list(particle.hist),
                init_x=particle.init_x,
                init_y=particle.init_y,
                respawned=particle.respawned,
            )
            for particle in particles
        ]
    }

    for step in range(1, max(CAPTURE_STEPS) + 1):
        for particle in particles:
            particle.respawned = False
            advected = context.advect_along_field_rk2(particle.x, particle.y)
            if not bool(advected["valid"]):
                cell = valid_cells[int(rng.integers(0, len(valid_cells)))]
                x, y = spawn_point_in_cell(context, cell, rng)
                reset_particle(particle, x, y)
                continue

            particle.x = float(advected["x"])
            particle.y = float(advected["y"])
            particle.age_sec += FIXED_ADVECTION_STEP_SEC
            particle.hist.insert(0, (particle.x, particle.y))
            trim_history_by_screen_length(context, particle.hist)

            out_of_bounds = (
                particle.x < context.bbox[0]
                or particle.x > context.bbox[1]
                or particle.y < context.bbox[2]
                or particle.y > context.bbox[3]
            )
            if out_of_bounds or particle.age_sec > MAX_PARTICLE_LIFETIME_SEC or is_masked_at(context, particle.x, particle.y):
                cell = valid_cells[int(rng.integers(0, len(valid_cells)))]
                x, y = spawn_point_in_cell(context, cell, rng)
                reset_particle(particle, x, y)

        if step in CAPTURE_STEPS:
            snapshots[step] = [
                Particle(
                    x=particle.x,
                    y=particle.y,
                    age_sec=particle.age_sec,
                    hist=list(particle.hist),
                    init_x=particle.init_x,
                    init_y=particle.init_y,
                    respawned=particle.respawned,
                )
                for particle in particles
            ]
    return snapshots


def draw_particles(ax, particles: list[Particle], color: str, emphasize_respawn: bool = False) -> None:
    for particle in particles:
        hist = np.asarray(particle.hist, dtype=float)
        if hist.shape[0] >= 2:
            segments = hist[:-1], hist[1:]
            count = hist.shape[0] - 1
            for idx, (head, tail) in enumerate(zip(*segments)):
                alpha = 0.18 + 0.72 * ((count - idx) / max(count, 1))
                ax.plot(
                    [head[0], tail[0]],
                    [head[1], tail[1]],
                    color=color,
                    linewidth=2.2,
                    alpha=alpha,
                    zorder=5,
                    solid_capstyle="round",
                )
        head_color = RESPAWN_COLOR if emphasize_respawn and particle.respawned else color
        if hist.shape[0] >= 2:
            _add_arrowhead(
                ax,
                hist[1],
                hist[0],
                head_color,
                linewidth=1.8,
                mutation_scale=10.5,
                zorder=6,
            )
        ax.scatter(
            [particle.x],
            [particle.y],
            s=42,
            color="white",
            edgecolors=head_color,
            linewidths=1.2,
            zorder=6,
        )


def render_animated_generation(case, context, output_path: Path) -> None:
    snapshots = simulate_particle_snapshots(context)
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.03, wspace=0.08)

    for ax, step in zip(axes, CAPTURE_STEPS):
        draw_field_background(ax, case, context)
        particles = snapshots[step]
        draw_particles(ax, particles, ANIM_COLOR, emphasize_respawn=(step == CAPTURE_STEPS[-1]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tmap_path = BACKEND_ROOT / "datasets/examples/simple2d/simple2d_tsne.tmap"
    output_dir = BACKEND_ROOT.parent / "output/paper_figures/trail_generation"
    case = _load_featurewind_case(tmap_path)
    feature_index = case.feature_names.index("horizontal_signal")
    condition = SensitivityCondition(
        condition_id="reference",
        label="reference",
        grid_res=DEMO_GRID_RES,
        interpolation_method="linear",
        mask_radius=0,
        is_reference=True,
    )
    context = build_condition_context(case, feature_idx=feature_index, condition=condition)
    point_index, trail = choose_demo_point(case, context)

    static_path = output_dir / "simple_2d_tsne_static_trail_generation.png"
    animated_path = output_dir / "simple_2d_tsne_animated_trail_generation.png"
    render_static_generation(case, context, trail, point_index, static_path)
    render_animated_generation(case, context, animated_path)

    print(f"Static trail figure: {static_path}")
    print(f"Animated trail figure: {animated_path}")


if __name__ == "__main__":
    main()
