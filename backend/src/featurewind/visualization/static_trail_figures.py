from __future__ import annotations

import base64
import io
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


TRAIL_EXPORT_FEATURE_COLOR_OVERRIDES = {
    "kernel_length": "#96360e",
}


def _safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    cleaned = cleaned.strip("._")
    return cleaned or "dataset"


def _coerce_point_array(points: Any) -> np.ndarray:
    if isinstance(points, (list, tuple)) and points and isinstance(points[0], dict):
        rows = []
        for point in points:
            if not isinstance(point, dict) or "x" not in point or "y" not in point:
                raise ValueError("Expected point objects with numeric 'x' and 'y' fields.")
            rows.append([float(point["x"]), float(point["y"])])
        arr = np.asarray(rows, dtype=float)
    else:
        arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Expected a 2D array of [x, y] points.")
    return arr


def _coerce_feature_matrix(feature_values: Any, n_points: int) -> np.ndarray:
    arr = np.asarray(feature_values, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D feature matrix.")
    if arr.shape[0] != n_points:
        raise ValueError("Feature matrix row count must match the number of positions.")
    return arr


def _decode_canvas_snapshot(snapshot_data_url: Any) -> np.ndarray | None:
    if not isinstance(snapshot_data_url, str) or not snapshot_data_url.strip():
        return None
    raw = snapshot_data_url.strip()
    if raw.startswith("data:"):
        parts = raw.split(",", 1)
        if len(parts) != 2:
            return None
        raw = parts[1]
    try:
        image_bytes = base64.b64decode(raw, validate=False)
    except Exception:
        return None
    try:
        return mpimg.imread(io.BytesIO(image_bytes), format="png")
    except Exception:
        return None


def _coerce_canvas_view(canvas_view: Any) -> dict[str, float] | None:
    if not isinstance(canvas_view, dict):
        return None
    required = ("xmin", "xmax", "ymin", "ymax")
    try:
        parsed = {key: float(canvas_view[key]) for key in required}
    except Exception:
        return None
    if not (parsed["xmax"] > parsed["xmin"] and parsed["ymax"] > parsed["ymin"]):
        return None
    return parsed


def _sample_count(n_points: int, target: int = 12, min_points: int = 8, max_points: int = 15) -> int:
    if n_points <= 0:
        return 0
    if n_points < min_points:
        return n_points
    if n_points <= max_points:
        return n_points
    return max(min_points, min(max_points, int(target)))


def sample_trail_points(points: Any, *, target_count: int = 12) -> tuple[np.ndarray, np.ndarray]:
    path = _coerce_point_array(points)
    if path.shape[0] == 1:
        return path.copy(), np.asarray([0.0], dtype=float)

    count = _sample_count(path.shape[0], target=target_count)
    if count <= 1:
        return path[[0]].copy(), np.asarray([0.0], dtype=float)

    deltas = np.diff(path, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = float(cumulative[-1])
    if not math.isfinite(total_length) or total_length <= 1e-12:
        indices = np.linspace(0, path.shape[0] - 1, count)
        sampled = np.vstack([
            path[int(round(idx))]
            for idx in indices
        ])
        progress = np.linspace(0.0, 1.0, sampled.shape[0], dtype=float)
        return sampled, progress

    target_distances = np.linspace(0.0, total_length, count, dtype=float)
    sampled = []
    for distance in target_distances:
        seg_idx = min(int(np.searchsorted(cumulative, distance, side="right")) - 1, len(seg_lengths) - 1)
        seg_idx = max(0, seg_idx)
        start_len = cumulative[seg_idx]
        seg_len = float(seg_lengths[seg_idx])
        if seg_len <= 1e-12:
            sampled.append(path[seg_idx].copy())
            continue
        t = (distance - start_len) / seg_len
        sampled.append(path[seg_idx] * (1.0 - t) + path[seg_idx + 1] * t)

    sampled_arr = np.asarray(sampled, dtype=float)
    progress = target_distances / total_length
    return sampled_arr, progress


def interpolate_feature_profiles(
    sample_points: Any,
    positions: Any,
    feature_values: Any,
    *,
    neighbors: int = 12,
    power: float = 2.0,
) -> np.ndarray:
    query = _coerce_point_array(sample_points)
    points = _coerce_point_array(positions)
    feats = _coerce_feature_matrix(feature_values, points.shape[0])
    k = max(1, min(int(neighbors), points.shape[0]))

    rows = []
    for point in query:
        dists = np.linalg.norm(points - point[None, :], axis=1)
        nearest_idx = np.argpartition(dists, k - 1)[:k]
        local_dists = dists[nearest_idx]
        if np.any(local_dists <= 1e-9):
            rows.append(feats[nearest_idx[int(np.argmin(local_dists))]])
            continue
        weights = 1.0 / np.power(local_dists, power)
        weights /= np.sum(weights)
        rows.append(np.sum(feats[nearest_idx] * weights[:, None], axis=0))
    return np.asarray(rows, dtype=float)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(values.shape[0], dtype=float)
    return ranks


def monotonicity_score(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.shape[0] < 2:
        return 0.0
    progress = np.linspace(0.0, 1.0, arr.shape[0], dtype=float)
    x = _rankdata(progress)
    y = _rankdata(arr)
    x -= np.mean(x)
    y -= np.mean(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 1e-12:
        return 0.0
    return float(abs(np.dot(x, y) / denom))


def _signed_monotonicity(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.shape[0] < 2:
        return 0.0
    progress = np.linspace(0.0, 1.0, arr.shape[0], dtype=float)
    x = _rankdata(progress)
    y = _rankdata(arr)
    x -= np.mean(x)
    y -= np.mean(y)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def select_target_and_controls(
    sampled_feature_values: Any,
    feature_names: list[str],
    feature_colors: list[str],
    *,
    active_feature_indices: list[int] | None = None,
    preferred_feature_index: int | None = None,
    max_controls: int = 2,
) -> tuple[int, list[int], dict[int, float]]:
    values = np.asarray(sampled_feature_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("sampled_feature_values must have shape (samples, n_features).")

    monotonic_scores = {
        idx: monotonicity_score(values[:, idx])
        for idx in range(values.shape[1])
    }

    valid_active = [
        int(idx)
        for idx in (active_feature_indices or [])
        if isinstance(idx, (int, np.integer)) and 0 <= int(idx) < values.shape[1]
    ]
    candidate_indices = valid_active or list(range(values.shape[1]))

    target_idx: int
    if preferred_feature_index is not None and 0 <= int(preferred_feature_index) < values.shape[1]:
        target_idx = int(preferred_feature_index)
    elif len(candidate_indices) == 1:
        target_idx = candidate_indices[0]
    else:
        target_idx = max(candidate_indices, key=lambda idx: monotonic_scores[idx])

    target_color = feature_colors[target_idx] if target_idx < len(feature_colors) else None
    control_candidates = []
    for idx in range(values.shape[1]):
        if idx == target_idx:
            continue
        same_color = target_color is not None and idx < len(feature_colors) and feature_colors[idx] == target_color
        control_candidates.append((same_color, monotonic_scores[idx], idx))
    control_candidates.sort(key=lambda item: (item[0], item[1], item[2]))

    controls: list[int] = []
    used_colors = set()
    for _, _, idx in control_candidates:
        color = feature_colors[idx] if idx < len(feature_colors) else None
        if color in used_colors:
            continue
        controls.append(idx)
        if color is not None:
            used_colors.add(color)
        if len(controls) >= max_controls:
            break

    if len(controls) < max_controls:
        for _, _, idx in control_candidates:
            if idx in controls:
                continue
            controls.append(idx)
            if len(controls) >= max_controls:
                break

    return target_idx, controls[:max_controls], monotonic_scores


def _trail_color(feature_idx: int | None, target_idx: int, feature_colors: list[str]) -> str:
    if feature_idx is not None and 0 <= int(feature_idx) < len(feature_colors):
        return feature_colors[int(feature_idx)]
    if 0 <= int(target_idx) < len(feature_colors):
        return feature_colors[int(target_idx)]
    return "#1f2937"


def _format_rho(values: Any) -> str:
    return f"{_signed_monotonicity(values):+.2f}"


def resolve_export_feature_colors(feature_names: list[str], feature_colors: list[str]) -> list[str]:
    colors = list(feature_colors or [])
    if len(colors) < len(feature_names):
        colors.extend(["#1f2937"] * (len(feature_names) - len(colors)))
    elif len(colors) > len(feature_names):
        colors = colors[:len(feature_names)]

    for idx, feature_name in enumerate(feature_names):
        override = TRAIL_EXPORT_FEATURE_COLOR_OVERRIDES.get(str(feature_name).strip().lower())
        if override is not None:
            colors[idx] = override
    return colors


def render_static_trail_figure(
    *,
    dataset_name: str,
    point_index: int | None,
    trail_points: Any,
    sample_points_arr: np.ndarray,
    progress: np.ndarray,
    sampled_feature_values: np.ndarray,
    positions: Any,
    feature_names: list[str],
    feature_colors: list[str],
    trail_feature_index: int | None,
    target_idx: int,
    control_indices: list[int],
    canvas_snapshot_data_url: str | None,
    canvas_view: dict[str, float] | None,
    output_png: Path,
    output_pdf: Path,
) -> None:
    trail = _coerce_point_array(trail_points)
    all_positions = _coerce_point_array(positions)
    target_color = feature_colors[target_idx] if target_idx < len(feature_colors) else "#0f766e"
    trail_color = _trail_color(trail_feature_index, target_idx, feature_colors)
    canvas_image = _decode_canvas_snapshot(canvas_snapshot_data_url)
    parsed_canvas_view = _coerce_canvas_view(canvas_view)

    fig = plt.figure(figsize=(10.8, 4.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.45])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_line = fig.add_subplot(gs[0, 1])
    fig.patch.set_facecolor("white")

    ax_map.set_facecolor("#fbfaf7")
    if canvas_image is not None:
        image_h, image_w = canvas_image.shape[:2]
        ax_map.imshow(canvas_image, origin="upper")
        if parsed_canvas_view is not None:
            sx = (sample_points_arr[:, 0] - parsed_canvas_view["xmin"]) / (parsed_canvas_view["xmax"] - parsed_canvas_view["xmin"]) * image_w
            sy = image_h - (sample_points_arr[:, 1] - parsed_canvas_view["ymin"]) / (parsed_canvas_view["ymax"] - parsed_canvas_view["ymin"]) * image_h
            ax_map.scatter(
                sx,
                sy,
                s=28,
                color=trail_color,
                edgecolors="white",
                linewidths=0.9,
                zorder=5,
            )
            if sample_points_arr.shape[0] >= 2:
                ax_map.scatter(
                    [sx[0]],
                    [sy[0]],
                    s=58,
                    color="white",
                    edgecolors=trail_color,
                    linewidths=1.3,
                    zorder=6,
                )
                ax_map.scatter(
                    [sx[-1]],
                    [sy[-1]],
                    s=58,
                    color=trail_color,
                    edgecolors="white",
                    linewidths=1.3,
                    zorder=6,
                )
        ax_map.set_xlim(0, image_w)
        ax_map.set_ylim(image_h, 0)
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        for spine in ax_map.spines.values():
            spine.set_visible(False)
    else:
        ax_map.scatter(
            all_positions[:, 0],
            all_positions[:, 1],
            s=16,
            color="#d1d5db",
            alpha=0.45,
            linewidths=0,
            zorder=1,
        )
        ax_map.plot(trail[:, 0], trail[:, 1], color=trail_color, linewidth=2.6, alpha=0.95, zorder=3)
        ax_map.scatter(
            sample_points_arr[:, 0],
            sample_points_arr[:, 1],
            s=40,
            color=trail_color,
            edgecolors="white",
            linewidths=0.9,
            zorder=4,
        )
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        for spine in ax_map.spines.values():
            spine.set_visible(False)

    ax_line.set_facecolor("#fbfaf7")
    ax_line.grid(axis="y", color="#ddd6c8", linewidth=0.8, alpha=0.7)
    for spine in ("top", "right"):
        ax_line.spines[spine].set_visible(False)
    ax_line.spines["left"].set_color("#8a8278")
    ax_line.spines["bottom"].set_color("#8a8278")
    ax_line.tick_params(colors="#3a342e", labelsize=10)

    target_values = sampled_feature_values[:, target_idx]
    ax_line.plot(
        progress,
        target_values,
        color=target_color,
        linewidth=2.8,
        marker="o",
        markersize=4.6,
        label=f"{feature_names[target_idx]}  (rho={_format_rho(target_values)})",
        zorder=4,
    )
    for control_idx in control_indices:
        color = feature_colors[control_idx] if control_idx < len(feature_colors) else "#6b7280"
        control_values = sampled_feature_values[:, control_idx]
        ax_line.plot(
            progress,
            control_values,
            color=color,
            linewidth=1.8,
            marker="o",
            markersize=3.8,
            alpha=0.85,
            label=f"{feature_names[control_idx]}  (rho={_format_rho(control_values)})",
            zorder=3,
        )
    ax_line.set_xlim(0.0, 1.0)
    ax_line.set_ylim(
        max(0.0, float(np.min(sampled_feature_values)) - 0.05),
        min(1.0, float(np.max(sampled_feature_values)) + 0.05),
    )
    ax_line.set_xlabel("Trail Progress", fontsize=11, color="#1f1b17")
    ax_line.set_ylabel("Normalized Feature Value", fontsize=11, color="#1f1b17")
    ax_line.set_title("Feature Profiles Along Trail", fontsize=12.5, fontweight="bold", color="#1f1b17")
    legend = ax_line.legend(frameon=False, loc="best", fontsize=9.8)
    for text in legend.get_texts():
        text.set_color("#1f1b17")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_static_trail_figures(
    *,
    dataset_name: str,
    positions: Any,
    feature_values: Any,
    feature_names: list[str],
    feature_colors: list[str],
    trails: list[dict[str, Any]],
    active_feature_indices: list[int] | None,
    canvas_snapshot_data_url: str | None,
    canvas_view: dict[str, float] | None,
    output_root: str | Path,
) -> dict[str, Any]:
    all_positions = _coerce_point_array(positions)
    feats = _coerce_feature_matrix(feature_values, all_positions.shape[0])
    if not feature_names:
        raise ValueError("feature_names is required.")
    if feats.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match feature_values column count.")
    if not isinstance(trails, list) or len(trails) == 0:
        raise ValueError("At least one static trail is required.")

    resolved_feature_colors = resolve_export_feature_colors(feature_names, feature_colors)
    output_dir = Path(output_root) / "static_trails" / _safe_slug(Path(dataset_name).stem)
    exported: list[dict[str, Any]] = []

    for ordinal, trail in enumerate(trails, start=1):
        trail_points = trail.get("points")
        if not isinstance(trail_points, list) or len(trail_points) < 2:
            continue
        sampled_points, progress = sample_trail_points(trail_points)
        sampled_values = interpolate_feature_profiles(sampled_points, all_positions, feats)
        preferred_feature_index = trail.get("featureIndex")
        if preferred_feature_index is None and active_feature_indices and len(active_feature_indices) == 1:
            preferred_feature_index = active_feature_indices[0]
        target_idx, control_indices, monotonic_scores = select_target_and_controls(
            sampled_values,
            feature_names,
            resolved_feature_colors,
            active_feature_indices=active_feature_indices,
            preferred_feature_index=preferred_feature_index,
            max_controls=2,
        )

        point_index = trail.get("pointIndex")
        point_token = f"point_{int(point_index):03d}" if isinstance(point_index, int) else f"trail_{ordinal:02d}"
        stem = f"{point_token}_{_safe_slug(feature_names[target_idx])}"
        output_png = output_dir / f"{stem}.png"
        output_pdf = output_dir / f"{stem}.pdf"
        render_static_trail_figure(
            dataset_name=Path(dataset_name).stem,
            point_index=int(point_index) if isinstance(point_index, int) else None,
            trail_points=trail_points,
            sample_points_arr=sampled_points,
            progress=progress,
            sampled_feature_values=sampled_values,
            positions=all_positions,
            feature_names=feature_names,
            feature_colors=resolved_feature_colors,
            trail_feature_index=None if trail.get("featureIndex") is None else int(trail["featureIndex"]),
            target_idx=target_idx,
            control_indices=control_indices,
            canvas_snapshot_data_url=canvas_snapshot_data_url,
            canvas_view=canvas_view,
            output_png=output_png,
            output_pdf=output_pdf,
        )
        exported.append(
            {
                "pointIndex": int(point_index) if isinstance(point_index, int) else None,
                "featureIndex": None if trail.get("featureIndex") is None else int(trail["featureIndex"]),
                "targetFeatureIndex": int(target_idx),
                "targetFeature": feature_names[target_idx],
                "targetMonotonicity": float(monotonic_scores[target_idx]),
                "controlFeatureIndices": [int(idx) for idx in control_indices],
                "controlFeatures": [feature_names[idx] for idx in control_indices],
                "sampleCount": int(sampled_points.shape[0]),
                "png": str(output_png),
                "pdf": str(output_pdf),
            }
        )

    if not exported:
        raise ValueError("None of the provided static trails contained enough points to export.")

    return {
        "output_dir": str(output_dir),
        "figures": exported,
    }
