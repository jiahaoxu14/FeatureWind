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
COMPARE_FEATURE_COLORS = ("#0f766e", "#c2410c", "#be185d", "#4d7c0f")
BACKGROUND_COLOR = "#ffffff"
GRIDLINE_COLOR = "#000000"
MASKED_CELL_COLOR = "#d4d4d8"
DEMO_GRID_RES = 8

FIXED_ADVECTION_STEP_SEC = 1.0 / 60.0
MAX_PARTICLE_LIFETIME_SEC = 8.0
ANIMATED_TRAIL_LENGTH_PX = 140.0
PARTICLE_COUNT = 18
CAPTURE_STEPS = (0, 12, 30, 54)
STATIC_PROGRESS_FRACTIONS = (0.0, 0.28, 0.62, 1.0)


@dataclass
class Particle:
    x: float
    y: float
    age_sec: float
    hist: list[tuple[float, float]]
    init_x: float
    init_y: float
    respawned: bool = False


def build_full_static_trail(context, start_x: float, start_y: float, point_index: int):
    return build_static_trail(
        context,
        start_x=float(start_x),
        start_y=float(start_y),
        point_index=int(point_index),
        max_steps=None,
    )


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
        trail = build_full_static_trail(context, start_x=float(x), start_y=float(y), point_index=point_index)
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
        trail = build_full_static_trail(context, start_x=float(x), start_y=float(y), point_index=0)
        best_index = 0
        best_trail = np.asarray(trail.points, dtype=float)
    return best_index, best_trail


def build_all_valid_trails(case, context) -> list[np.ndarray]:
    trails: list[np.ndarray] = []
    for point_index, (x, y) in enumerate(case.positions):
        trail = build_full_static_trail(context, start_x=float(x), start_y=float(y), point_index=point_index)
        if not trail.valid or trail.point_count < 2:
            continue
        trails.append(np.asarray(trail.points, dtype=float))
    return trails


def choose_comparison_point(case, contexts: list) -> tuple[int, list[np.ndarray]]:
    center = np.asarray(
        [
            0.5 * (case.bbox[0] + case.bbox[1]),
            0.5 * (case.bbox[2] + case.bbox[3]),
        ],
        dtype=float,
    )
    best_index = 0
    best_trails: list[np.ndarray] | None = None
    best_score = None

    for point_index, (x, y) in enumerate(case.positions):
        trails: list[np.ndarray] = []
        lengths: list[float] = []
        valid = True
        for context in contexts:
            trail = build_full_static_trail(context, start_x=float(x), start_y=float(y), point_index=point_index)
            if not trail.valid or trail.point_count < 8:
                valid = False
                break
            path = np.asarray(trail.points, dtype=float)
            trails.append(path)
            lengths.append(_path_length(path))
        if not valid:
            continue
        distance_to_center = float(np.linalg.norm(np.asarray([x, y], dtype=float) - center))
        score = (min(lengths), sum(lengths), -distance_to_center)
        if best_score is None or score > best_score:
            best_index = point_index
            best_trails = trails
            best_score = score

    if best_trails is None:
        x, y = case.positions[0]
        fallback_trails = []
        for context in contexts:
            trail = build_full_static_trail(context, start_x=float(x), start_y=float(y), point_index=0)
            fallback_trails.append(np.asarray(trail.points, dtype=float))
        return 0, fallback_trails

    return best_index, best_trails


def choose_random_comparison_point(case, contexts: list, seed: int = 20260401) -> tuple[int, list[np.ndarray]]:
    valid_choices: list[tuple[int, list[np.ndarray]]] = []
    for point_index, (x, y) in enumerate(case.positions):
        trails: list[np.ndarray] = []
        valid = True
        for context in contexts:
            trail = build_full_static_trail(context, start_x=float(x), start_y=float(y), point_index=point_index)
            if not trail.valid or trail.point_count < 8:
                valid = False
                break
            trails.append(np.asarray(trail.points, dtype=float))
        if valid:
            valid_choices.append((point_index, trails))

    if not valid_choices:
        return choose_comparison_point(case, contexts)

    rng = np.random.default_rng(int(seed))
    chosen = valid_choices[int(rng.integers(0, len(valid_choices)))]
    return chosen


def choose_two_far_random_comparison_points(
    case,
    contexts: list,
    seed: int = 20260401,
) -> list[tuple[int, list[np.ndarray]]]:
    valid_choices: list[tuple[int, list[np.ndarray]]] = []
    for point_index, (x, y) in enumerate(case.positions):
        trails: list[np.ndarray] = []
        valid = True
        for context in contexts:
            trail = build_static_trail(
                context,
                start_x=float(x),
                start_y=float(y),
                point_index=point_index,
                max_steps=120,
            )
            if not trail.valid or trail.point_count < 8:
                valid = False
                break
            trails.append(np.asarray(trail.points, dtype=float))
        if valid:
            valid_choices.append((point_index, trails))

    if not valid_choices:
        point_index, trails = choose_comparison_point(case, contexts)
        return [(point_index, trails)]
    if len(valid_choices) == 1:
        return [valid_choices[0]]

    rng = np.random.default_rng(int(seed))
    first_choice = valid_choices[int(rng.integers(0, len(valid_choices)))]
    first_index = first_choice[0]
    first_pos = np.asarray(case.positions[first_index], dtype=float)

    farthest_choice = None
    farthest_distance = -1.0
    for candidate in valid_choices:
        candidate_index = candidate[0]
        if candidate_index == first_index:
            continue
        candidate_pos = np.asarray(case.positions[candidate_index], dtype=float)
        distance = float(np.linalg.norm(candidate_pos - first_pos))
        if distance > farthest_distance:
            farthest_distance = distance
            farthest_choice = candidate

    if farthest_choice is None:
        return [first_choice]
    return [first_choice, farthest_choice]


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


def draw_field_background(ax, case, context, show_vectors: bool = True) -> None:
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
    if show_vectors and qx.size > 0:
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


def _get_static_sampling(trail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sample_indices = np.unique(
        np.linspace(0, trail.shape[0] - 1, min(7, trail.shape[0]), dtype=int)
    )
    sampled = trail[sample_indices]
    segment_colors = np.linspace(0.35, 0.95, max(1, sampled.shape[0] - 1))
    return sampled, segment_colors


def draw_static_panel(ax, case, context, trail: np.ndarray, stage_index: int) -> None:
    sampled, segment_colors = _get_static_sampling(trail)
    draw_field_background(ax, case, context)
    if stage_index == 0:
        ax.scatter(
            [trail[0, 0]],
            [trail[0, 1]],
            s=190,
            facecolors="white",
            edgecolors=STATIC_COLOR,
            linewidths=2.2,
            zorder=5,
        )
        return

    if stage_index in (1, 2):
        progress_fraction = STATIC_PROGRESS_FRACTIONS[stage_index]
        keep_count = max(stage_index + 1, int(math.ceil(sampled.shape[0] * progress_fraction)))
        stage_sampled = sampled[:keep_count]
        for idx in range(stage_sampled.shape[0] - 1):
            a = stage_sampled[idx]
            b = stage_sampled[idx + 1]
            alpha = float(segment_colors[min(idx, len(segment_colors) - 1)])
            ax.plot(
                [a[0], b[0]],
                [a[1], b[1]],
                color=STATIC_COLOR,
                linewidth=3.0,
                alpha=alpha,
                zorder=5,
                solid_capstyle="round",
            )
            _add_arrowhead(
                ax,
                a,
                b,
                STATIC_COLOR,
                linewidth=1.8,
                mutation_scale=11.5,
                zorder=6,
            )
            ax.scatter(
                [a[0]],
                [a[1]],
                s=110,
                color=(1.0, 1.0, 1.0, 0.98),
                edgecolors=STATIC_COLOR,
                linewidths=1.6,
                zorder=6,
            )
        ax.scatter(
            [stage_sampled[-1, 0]],
            [stage_sampled[-1, 1]],
            s=100,
            color=STATIC_COLOR,
            edgecolors="white",
            linewidths=1.0,
            zorder=7,
        )
        return

    ax.plot(trail[:, 0], trail[:, 1], color="white", linewidth=6.0, zorder=5, solid_capstyle="round")
    ax.plot(trail[:, 0], trail[:, 1], color=STATIC_COLOR, linewidth=3.1, zorder=6, solid_capstyle="round")
    sample_marker_indices = np.unique(
        np.linspace(0, trail.shape[0] - 1, min(6, trail.shape[0]), dtype=int)
    )
    sample_markers = trail[sample_marker_indices]
    ax.scatter(
        sample_markers[:, 0],
        sample_markers[:, 1],
        s=96,
        color=(1.0, 1.0, 1.0, 0.98),
        edgecolors=STATIC_COLOR,
        linewidths=1.5,
        zorder=7,
    )
    ax.scatter(
        [sample_markers[-1, 0]],
        [sample_markers[-1, 1]],
        s=102,
        color=STATIC_COLOR,
        edgecolors="white",
        linewidths=1.0,
        zorder=8,
    )
    ax.scatter(
        [trail[0, 0]],
        [trail[0, 1]],
        s=190,
        facecolors="white",
        edgecolors=STATIC_COLOR,
        linewidths=2.2,
        zorder=7,
    )
    if trail.shape[0] >= 2:
        _add_arrowhead(ax, trail[-2], trail[-1], STATIC_COLOR, linewidth=2.4, mutation_scale=16.0, zorder=9)


def _render_single_panel(case, context, draw_panel_fn, output_path: Path, *panel_args) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.1, 5.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.03)
    draw_panel_fn(ax, case, context, *panel_args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_all_static_trails(case, context, trails: list[np.ndarray], output_path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 6.3), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.03)

    draw_field_background(ax, case, context)
    for trail in trails:
        ax.plot(
            trail[:, 0],
            trail[:, 1],
            color="white",
            linewidth=3.8,
            alpha=0.92,
            zorder=4,
            solid_capstyle="round",
        )
        ax.plot(
            trail[:, 0],
            trail[:, 1],
            color=STATIC_COLOR,
            linewidth=1.9,
            alpha=0.85,
            zorder=5,
            solid_capstyle="round",
        )
        ax.scatter(
            [trail[0, 0]],
            [trail[0, 1]],
            s=30,
            facecolors="white",
            edgecolors=STATIC_COLOR,
            linewidths=1.0,
            zorder=6,
        )
        if trail.shape[0] >= 2:
            _add_arrowhead(
                ax,
                trail[-2],
                trail[-1],
                STATIC_COLOR,
                linewidth=1.3,
                mutation_scale=10.5,
                zorder=7,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_static_feature_comparison(
    case,
    contexts: list,
    trails: list[np.ndarray],
    output_path: Path,
    *,
    show_trail_markers: bool = True,
    seed_marker_facecolor: str = "white",
    seed_marker_edgecolor: str = "#111827",
    seed_marker_linewidth: float = 2.2,
    seed_marker_size: float = 210.0,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 6.3), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.03)

    draw_field_background(ax, case, contexts[0], show_vectors=False)
    for trail, color in zip(trails, COMPARE_FEATURE_COLORS):
        ax.plot(trail[:, 0], trail[:, 1], color="white", linewidth=6.0, zorder=5, solid_capstyle="round")
        ax.plot(trail[:, 0], trail[:, 1], color=color, linewidth=3.0, zorder=6, solid_capstyle="round")
        if show_trail_markers:
            marker_indices = np.unique(
                np.linspace(0, trail.shape[0] - 1, min(5, trail.shape[0]), dtype=int)
            )
            markers = trail[marker_indices]
            ax.scatter(
                markers[:, 0],
                markers[:, 1],
                s=78,
                color=(1.0, 1.0, 1.0, 0.98),
                edgecolors=color,
                linewidths=1.3,
                zorder=7,
            )
            ax.scatter(
                [markers[-1, 0]],
                [markers[-1, 1]],
                s=84,
                color=color,
                edgecolors="white",
                linewidths=0.9,
                zorder=8,
            )
        if trail.shape[0] >= 2:
            _add_arrowhead(ax, trail[-2], trail[-1], color, linewidth=2.2, mutation_scale=14.0, zorder=9)

    start = trails[0][0]
    ax.scatter(
        [start[0]],
        [start[1]],
        s=float(seed_marker_size),
        facecolors=seed_marker_facecolor,
        edgecolors=seed_marker_edgecolor,
        linewidths=float(seed_marker_linewidth),
        zorder=10,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_static_feature_comparison_multi_seed(
    case,
    contexts: list,
    seeded_trails: list[tuple[int, list[np.ndarray]]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 6.3), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.03)

    draw_field_background(ax, case, contexts[0], show_vectors=False)
    for point_index, trails in seeded_trails:
        for trail, color in zip(trails, COMPARE_FEATURE_COLORS):
            ax.plot(trail[:, 0], trail[:, 1], color="white", linewidth=6.0, zorder=5, solid_capstyle="round")
            ax.plot(trail[:, 0], trail[:, 1], color=color, linewidth=3.0, zorder=6, solid_capstyle="round")
            if trail.shape[0] >= 2:
                _add_arrowhead(ax, trail[-2], trail[-1], color, linewidth=2.2, mutation_scale=14.0, zorder=9)

        start = np.asarray(case.positions[point_index], dtype=float)
        ax.scatter(
            [start[0]],
            [start[1]],
            s=64.0,
            facecolors="#111827",
            edgecolors="#111827",
            linewidths=0.0,
            zorder=10,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_static_generation(case, context, trail: np.ndarray, point_index: int, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20.6, 5.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.02, right=0.995, top=0.99, bottom=0.03, wspace=0.04)

    for stage_index, ax in enumerate(axes):
        draw_static_panel(ax, case, context, trail, stage_index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    for stage_index in range(4):
        panel_path = output_path.with_name(f"{output_path.stem}_stage_{stage_index + 1}{output_path.suffix}")
        _render_single_panel(case, context, draw_static_panel, panel_path, trail, stage_index)


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


def initialize_particles(context, rng: np.random.Generator, particle_count: int = PARTICLE_COUNT) -> list[Particle]:
    valid_cells = build_valid_spawn_cells(context)
    particles: list[Particle] = []
    for _ in range(int(particle_count)):
        cell = valid_cells[int(rng.integers(0, len(valid_cells)))]
        x, y = spawn_point_in_cell(context, cell, rng)
        particles.append(Particle(x=x, y=y, age_sec=0.0, hist=[(x, y)], init_x=x, init_y=y))
    return particles


def simulate_particle_snapshots(context, particle_count: int = PARTICLE_COUNT) -> dict[int, list[Particle]]:
    rng = np.random.default_rng(7)
    valid_cells = build_valid_spawn_cells(context)
    particles = initialize_particles(context, rng, particle_count=particle_count)
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


def draw_particles(
    ax,
    particles: list[Particle],
    color: str,
    emphasize_respawn: bool = False,
    show_particle_dots: bool = True,
) -> None:
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
        if show_particle_dots:
            ax.scatter(
                [particle.x],
                [particle.y],
                s=42,
                color="white",
                edgecolors=head_color,
                linewidths=1.2,
                zorder=6,
            )


def draw_animated_panel(
    ax,
    case,
    context,
    snapshots: dict[int, list[Particle]],
    stage_index: int,
    show_particle_dots: bool = True,
) -> None:
    draw_field_background(ax, case, context)
    step = CAPTURE_STEPS[stage_index]
    particles = snapshots[step]
    draw_particles(
        ax,
        particles,
        ANIM_COLOR,
        emphasize_respawn=(step == CAPTURE_STEPS[-1]),
        show_particle_dots=show_particle_dots,
    )


def render_animated_generation(
    case,
    context,
    output_path: Path,
    *,
    write_stage_panels: bool = True,
    particle_count: int = PARTICLE_COUNT,
    show_particle_dots: bool = True,
) -> None:
    snapshots = simulate_particle_snapshots(context, particle_count=particle_count)
    fig, axes = plt.subplots(1, 4, figsize=(20.6, 5.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.02, right=0.995, top=0.99, bottom=0.03, wspace=0.04)

    for stage_index, ax in enumerate(axes):
        draw_animated_panel(ax, case, context, snapshots, stage_index, show_particle_dots=show_particle_dots)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    if write_stage_panels:
        for stage_index in range(4):
            panel_path = output_path.with_name(f"{output_path.stem}_stage_{stage_index + 1}{output_path.suffix}")
            _render_single_panel(case, context, draw_animated_panel, panel_path, snapshots, stage_index, show_particle_dots)


def render_animated_feature_comparison(
    case,
    contexts: list,
    output_path: Path,
    *,
    particle_count_total: int | None = None,
    show_particle_dots: bool = True,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 6.3), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.03, right=0.99, top=0.99, bottom=0.03)

    draw_field_background(ax, case, contexts[0], show_vectors=False)
    total_count = int(particle_count_total) if particle_count_total is not None else (PARTICLE_COUNT * len(contexts))
    base_count = max(1, total_count // max(len(contexts), 1))
    remainder = max(0, total_count - (base_count * len(contexts)))
    per_feature_counts = [base_count + (1 if idx < remainder else 0) for idx in range(len(contexts))]

    for idx, (context, color) in enumerate(zip(contexts, COMPARE_FEATURE_COLORS)):
        snapshots = simulate_particle_snapshots(context, particle_count=per_feature_counts[idx])
        particles = snapshots[CAPTURE_STEPS[-1]]
        draw_particles(ax, particles, color, emphasize_respawn=False, show_particle_dots=show_particle_dots)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tmap_path = BACKEND_ROOT / "datasets/examples/simple2d/simple2d_tsne.tmap"
    output_dir = BACKEND_ROOT.parent / "output/paper_figures/trail_generation"
    case = _load_featurewind_case(tmap_path)
    condition = SensitivityCondition(
        condition_id="reference",
        label="reference",
        grid_res=DEMO_GRID_RES,
        interpolation_method="linear",
        mask_radius=0,
        is_reference=True,
    )
    feature_names = ["horizontal_signal", "vertical_signal"]
    contexts = [
        build_condition_context(case, feature_idx=case.feature_names.index(feature_name), condition=condition)
        for feature_name in feature_names
    ]
    context = contexts[0]
    point_index, trail = choose_demo_point(case, context)
    all_trails = build_all_valid_trails(case, context)
    comparison_point_index, comparison_trails = choose_comparison_point(case, contexts)

    static_path = output_dir / "simple_2d_tsne_static_trail_generation.png"
    animated_path = output_dir / "simple_2d_tsne_animated_trail_generation.png"
    static_all_points_path = output_dir / "simple_2d_tsne_static_trail_generation_all_points.png"
    static_comparison_path = output_dir / "simple_2d_tsne_static_trail_feature_comparison.png"
    animated_comparison_path = output_dir / "simple_2d_tsne_animated_trail_feature_comparison.png"
    random_signal_b_context = build_condition_context(
        case,
        feature_idx=case.feature_names.index("random_signal_b"),
        condition=condition,
    )
    random_signal_b_all_trails = build_all_valid_trails(case, random_signal_b_context)
    random_signal_b_static_all_points_path = output_dir / "simple_2d_tsne_random_signal_b_static_trail_generation_all_points.png"
    random_signal_b_animated_path = output_dir / "simple_2d_tsne_random_signal_b_animated_trail_generation.png"
    random_signal_b_animated_p500_path = output_dir / "simple_2d_tsne_random_signal_b_animated_trail_generation_p500.png"
    all_feature_names = list(case.feature_names)
    all_feature_contexts = [
        build_condition_context(case, feature_idx=case.feature_names.index(feature_name), condition=condition)
        for feature_name in all_feature_names
    ]
    all_feature_seeded_trails = choose_two_far_random_comparison_points(case, all_feature_contexts)
    all_feature_static_comparison_path = output_dir / "simple_2d_tsne_all_features_static_trail_feature_comparison_random_point.png"
    all_feature_animated_comparison_p500_path = output_dir / "simple_2d_tsne_all_features_animated_trail_feature_comparison_p500.png"
    legacy_random_static_comparison_path = output_dir / "simple_2d_tsne_random_features_static_trail_feature_comparison_random_point.png"
    legacy_random_animated_comparison_p500_path = output_dir / "simple_2d_tsne_random_features_animated_trail_feature_comparison_p500.png"
    render_static_generation(case, context, trail, point_index, static_path)
    render_animated_generation(case, context, animated_path)
    render_all_static_trails(case, context, all_trails, static_all_points_path)
    render_static_feature_comparison(case, contexts, comparison_trails, static_comparison_path)
    render_animated_feature_comparison(case, contexts, animated_comparison_path)
    render_all_static_trails(case, random_signal_b_context, random_signal_b_all_trails, random_signal_b_static_all_points_path)
    render_animated_generation(case, random_signal_b_context, random_signal_b_animated_path, write_stage_panels=False)
    render_animated_generation(
        case,
        random_signal_b_context,
        random_signal_b_animated_p500_path,
        write_stage_panels=False,
        particle_count=500,
        show_particle_dots=False,
    )
    render_static_feature_comparison_multi_seed(
        case,
        all_feature_contexts,
        all_feature_seeded_trails,
        all_feature_static_comparison_path,
    )
    render_animated_feature_comparison(
        case,
        all_feature_contexts,
        all_feature_animated_comparison_p500_path,
        particle_count_total=500,
        show_particle_dots=False,
    )
    render_static_feature_comparison_multi_seed(
        case,
        all_feature_contexts,
        all_feature_seeded_trails,
        legacy_random_static_comparison_path,
    )
    render_animated_feature_comparison(
        case,
        all_feature_contexts,
        legacy_random_animated_comparison_p500_path,
        particle_count_total=500,
        show_particle_dots=False,
    )

    print(f"Static trail figure: {static_path}")
    print(f"Animated trail figure: {animated_path}")
    print(f"All-points static trail figure: {static_all_points_path}")
    print(f"Static feature comparison figure: {static_comparison_path} (point {comparison_point_index})")
    print(f"Animated feature comparison figure: {animated_comparison_path}")
    print(f"Random signal B all-points static trail figure: {random_signal_b_static_all_points_path}")
    print(f"Random signal B animated trail figure: {random_signal_b_animated_path}")
    print(f"Random signal B animated trail figure (500 particles): {random_signal_b_animated_p500_path}")
    print(
        f"All-feature static comparison figure: {all_feature_static_comparison_path} "
        f"(random points {[point_index for point_index, _ in all_feature_seeded_trails]})"
    )
    print(f"All-feature animated comparison figure (500 particles): {all_feature_animated_comparison_p500_path}")


if __name__ == "__main__":
    main()
