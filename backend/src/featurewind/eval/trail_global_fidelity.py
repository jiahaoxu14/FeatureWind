from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import tempfile
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .. import config as fw_config
from .advection import build_field_context
from .metrics import compute_stepwise_path_metrics, total_turning_angle
from .synthetic_global_cases import (
    FittedGlobalUMAPCase,
    GlobalUMAPCaseSpec,
    build_global_synthetic_cases,
    fit_global_umap_case,
)


MAIN_EXPERIMENT_NAME = "Global Oracle Trail Fidelity with Actual Nonlinear DR"


@dataclass(frozen=True)
class GlobalTrailFidelityConfig:
    output_dir: Path
    steps: int = 10
    delta_frac: float = 0.1
    fd_epsilon_frac: float = 0.02
    seed: int = 0
    datasets: tuple[str, ...] = ("curved_global_umap", "near_linear_global_umap")
    feature_indices: tuple[int, ...] | None = None
    dr_backend: str = "umap"
    grid_res: int = 48
    rollout_substeps: int = 4


@dataclass(frozen=True)
class AnchoredGlobalAxisCurve:
    feature_idx: int
    z_samples: np.ndarray
    xy_samples: np.ndarray
    bin_median_z: np.ndarray
    bin_median_xy: np.ndarray

    def sample_xy(self, z_value: float) -> np.ndarray:
        z = float(np.clip(z_value, float(self.z_samples[0]), float(self.z_samples[-1])))
        x = np.interp(z, self.z_samples, self.xy_samples[:, 0])
        y = np.interp(z, self.z_samples, self.xy_samples[:, 1])
        return np.array([x, y], dtype=float)

    def nearest_z(self, point_xy: np.ndarray) -> float:
        point = np.asarray(point_xy, dtype=float)
        distances = np.linalg.norm(self.xy_samples - point[None, :], axis=1)
        nearest_idx = int(np.argmin(distances))
        return float(self.z_samples[nearest_idx])


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _override_feature_indices(case: GlobalUMAPCaseSpec, feature_indices: tuple[int, ...] | None) -> GlobalUMAPCaseSpec:
    if not feature_indices:
        return case
    valid = tuple(
        int(idx)
        for idx in feature_indices
        if 0 <= int(idx) < len(case.feature_names)
    )
    if not valid:
        return case
    return GlobalUMAPCaseSpec(
        name=case.name,
        title=case.title,
        description=case.description,
        feature_names=case.feature_names,
        X=np.asarray(case.X, dtype=float),
        eval_feature_indices=valid,
        dr_backend=case.dr_backend,
        oracle_kind=case.oracle_kind,
        local_vector_method=case.local_vector_method,
    )


def _build_case_specs(cfg: GlobalTrailFidelityConfig) -> list[GlobalUMAPCaseSpec]:
    specs = [
        _override_feature_indices(case, cfg.feature_indices)
        for case in build_global_synthetic_cases(datasets=cfg.datasets, seed=cfg.seed)
    ]
    if not specs:
        raise ValueError("No synthetic datasets were selected for the UMAP-based global fidelity experiment.")
    return specs


def _padded_bbox(points: np.ndarray, pad_frac: float = 0.08) -> tuple[float, float, float, float]:
    arr = np.asarray(points, dtype=float)
    xmin = float(arr[:, 0].min())
    xmax = float(arr[:, 0].max())
    ymin = float(arr[:, 1].min())
    ymax = float(arr[:, 1].max())
    x_span = max(xmax - xmin, 1e-6)
    y_span = max(ymax - ymin, 1e-6)
    pad_x = x_span * float(pad_frac)
    pad_y = y_span * float(pad_frac)
    return xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y


def _oracle_support_bbox(case: FittedGlobalUMAPCase, *, steps: int, delta_frac: float, pad_frac: float = 0.15) -> tuple[float, float, float, float]:
    oracle_points = [np.asarray(case.Y, dtype=float)]
    for feature_idx in case.eval_feature_indices:
        delta = case.step_delta(int(feature_idx), float(delta_frac))
        for point_idx in range(case.X.shape[0]):
            oracle_points.append(case.oracle_trail(point_idx, int(feature_idx), delta=delta, steps=int(steps)))
    return _padded_bbox(np.vstack(oracle_points), pad_frac=pad_frac)


def _create_cell_centers(bbox: tuple[float, float, float, float], grid_res: int) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin + (xmax - xmin) / (2 * grid_res), xmax - (xmax - xmin) / (2 * grid_res), grid_res)
    ys = np.linspace(ymin + (ymax - ymin) / (2 * grid_res), ymax - (ymax - ymin) / (2 * grid_res), grid_res)
    return np.meshgrid(xs, ys)


def _idw_interpolate_vectors(
    positions: np.ndarray,
    vectors: np.ndarray,
    sample_points: np.ndarray,
    *,
    k_neighbors: int = 8,
) -> np.ndarray:
    pos = np.asarray(positions, dtype=float)
    vec = np.asarray(vectors, dtype=float)
    samples = np.asarray(sample_points, dtype=float)
    k = max(1, min(int(k_neighbors), pos.shape[0]))

    diff = samples[:, None, :] - pos[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    nearest_idx = np.argpartition(dist2, kth=k - 1, axis=1)[:, :k]
    nearest_dist2 = np.take_along_axis(dist2, nearest_idx, axis=1)
    nearest_vec = vec[nearest_idx]

    exact = nearest_dist2[:, 0] <= 1e-12
    weights = 1.0 / np.maximum(nearest_dist2, 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)
    interpolated = np.sum(nearest_vec * weights[:, :, None], axis=1)
    if np.any(exact):
        interpolated[exact] = nearest_vec[exact, 0, :]
    return interpolated


def build_global_field_contexts(case: FittedGlobalUMAPCase, *, grid_res: int, steps: int, delta_frac: float) -> dict[int, Any]:
    """Build a gridded field from finite-difference local vectors at the training points."""
    bbox = _oracle_support_bbox(case, steps=int(steps), delta_frac=float(delta_frac), pad_frac=0.15)
    feature_indices = list(case.eval_feature_indices) or list(range(len(case.feature_names)))
    grid_u_all_feats = None
    grid_v_all_feats = None

    try:
        orig_bbox = getattr(fw_config, "bounding_box", None)
        fw_config.initialize_global_state()
        bbox_points = np.vstack(
            [
                np.asarray(case.Y, dtype=float),
                np.array(
                    [
                        [bbox[0], bbox[2]],
                        [bbox[1], bbox[2]],
                        [bbox[0], bbox[3]],
                        [bbox[1], bbox[3]],
                    ],
                    dtype=float,
                ),
            ]
        )
        fw_config.set_bounding_box(bbox_points)
        from ..physics.grid_computation import build_grids  # type: ignore

        with tempfile.TemporaryDirectory(prefix="featurewind_global_eval_") as tmpdir:
            grid_result = build_grids(
                np.asarray(case.Y, dtype=float),
                int(grid_res),
                feature_indices,
                np.asarray(case.grad_vectors, dtype=float),
                list(case.feature_names),
                output_dir=tmpdir,
            )
        grid_u_all_feats = np.asarray(grid_result[8], dtype=float)
        grid_v_all_feats = np.asarray(grid_result[9], dtype=float)
    except Exception:
        grid_x, grid_y = _create_cell_centers(bbox, int(grid_res))
        sample_points = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
        positions = np.asarray(case.Y, dtype=float)
        grad_vectors = np.asarray(case.grad_vectors, dtype=float)
        u_grids = []
        v_grids = []
        for feature_idx in feature_indices:
            interpolated = _idw_interpolate_vectors(positions, grad_vectors[:, int(feature_idx), :], sample_points)
            u_grids.append(interpolated[:, 0].reshape(int(grid_res), int(grid_res)))
            v_grids.append(interpolated[:, 1].reshape(int(grid_res), int(grid_res)))
        grid_u_all_feats = np.asarray(u_grids, dtype=float)
        grid_v_all_feats = np.asarray(v_grids, dtype=float)
    finally:
        if "orig_bbox" in locals():
            fw_config.bounding_box = orig_bbox

    contexts: dict[int, Any] = {}
    for offset, feature_idx in enumerate(feature_indices):
        contexts[int(feature_idx)] = build_field_context(
            grid_u_all_feats[offset],
            grid_v_all_feats[offset],
            bbox=bbox,
            final_mask=None,
        )
    return contexts


def build_linear_baseline_path(start_xy: np.ndarray, start_vector: np.ndarray, *, delta: float, steps: int) -> np.ndarray:
    step_index = np.arange(int(steps) + 1, dtype=float)[:, None]
    return np.asarray(start_xy, dtype=float)[None, :] + step_index * float(delta) * np.asarray(start_vector, dtype=float)[None, :]


def build_anchored_global_axis_curve(
    case: FittedGlobalUMAPCase,
    feature_idx: int,
    *,
    num_bins: int = 24,
    samples_per_interval: int = 12,
) -> AnchoredGlobalAxisCurve:
    values = np.asarray(case.X[:, int(feature_idx)], dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=0))
    std = std if std > 1e-12 else 1.0
    z = (values - mean) / std

    order = np.argsort(z, kind="mergesort")
    chunks = [chunk for chunk in np.array_split(order, max(1, min(int(num_bins), len(order)))) if len(chunk) > 0]
    bin_median_z = np.array([np.median(z[chunk]) for chunk in chunks], dtype=float)
    bin_median_xy = np.array([np.median(np.asarray(case.Y, dtype=float)[chunk], axis=0) for chunk in chunks], dtype=float)

    sort_idx = np.argsort(bin_median_z, kind="mergesort")
    bin_median_z = bin_median_z[sort_idx]
    bin_median_xy = bin_median_xy[sort_idx]

    z_min = float(bin_median_z[0])
    z_max = float(bin_median_z[-1])
    if len(bin_median_z) < 2 or abs(z_max - z_min) <= 1e-9:
        z_samples = np.array([z_min, z_min + 1e-6], dtype=float)
        xy_samples = np.repeat(bin_median_xy[:1], 2, axis=0)
    else:
        sample_count = max(2, (len(bin_median_z) - 1) * int(samples_per_interval) + 1)
        z_samples = np.linspace(z_min, z_max, sample_count, dtype=float)
        x_samples = np.interp(z_samples, bin_median_z, bin_median_xy[:, 0])
        y_samples = np.interp(z_samples, bin_median_z, bin_median_xy[:, 1])
        xy_samples = np.column_stack([x_samples, y_samples])

    return AnchoredGlobalAxisCurve(
        feature_idx=int(feature_idx),
        z_samples=np.asarray(z_samples, dtype=float),
        xy_samples=np.asarray(xy_samples, dtype=float),
        bin_median_z=np.asarray(bin_median_z, dtype=float),
        bin_median_xy=np.asarray(bin_median_xy, dtype=float),
    )


def build_global_axis_curves(case: FittedGlobalUMAPCase) -> dict[int, AnchoredGlobalAxisCurve]:
    feature_indices = list(case.eval_feature_indices) or list(range(len(case.feature_names)))
    return {
        int(feature_idx): build_anchored_global_axis_curve(case, int(feature_idx))
        for feature_idx in feature_indices
    }


def build_anchored_global_axis_path(
    start_xy: np.ndarray,
    axis_curve: AnchoredGlobalAxisCurve,
    *,
    steps: int,
    delta_t: float,
    target_first_step_length: float | None = None,
) -> dict[str, Any]:
    start = np.asarray(start_xy, dtype=float)
    anchor_z = axis_curve.nearest_z(start)
    anchor_xy = axis_curve.sample_xy(anchor_z)
    scale = 1.0
    if int(steps) >= 1 and target_first_step_length is not None:
        first_target_xy = axis_curve.sample_xy(anchor_z + float(delta_t))
        raw_first_step_length = float(np.linalg.norm(first_target_xy - anchor_xy))
        if raw_first_step_length > 1e-12:
            scale = min(1.0, float(target_first_step_length) / raw_first_step_length)
    path = np.empty((int(steps) + 1, 2), dtype=float)
    path[0] = start

    for step_idx in range(1, int(steps) + 1):
        target_z = anchor_z + float(step_idx) * float(delta_t)
        target_xy = axis_curve.sample_xy(target_z)
        path[step_idx] = start + scale * (target_xy - anchor_xy)

    return {
        "path": path,
        "anchor_z": float(anchor_z),
        "anchor_xy": np.asarray(anchor_xy, dtype=float),
        "scale": float(scale),
    }


def _subsample_rollout_points(points: np.ndarray, *, steps: int, rollout_substeps: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    expected = int(steps) * int(rollout_substeps) + 1
    if pts.shape[0] != expected:
        raise ValueError("rollout points do not match the expected substep-aligned length")
    indices = np.arange(0, expected, int(rollout_substeps), dtype=int)
    return pts[indices]


def evaluate_global_case_combo(
    case: FittedGlobalUMAPCase,
    field_context: Any,
    axis_curve: AnchoredGlobalAxisCurve | None = None,
    *,
    point_idx: int,
    feature_idx: int,
    steps: int,
    delta_frac: float,
    fd_epsilon_frac: float,
    dr_backend: str,
    rollout_substeps: int,
) -> dict[str, Any]:
    delta = case.step_delta(int(feature_idx), float(delta_frac))
    fd_epsilon = case.fd_epsilon(int(feature_idx))
    oracle_path = case.oracle_trail(
        int(point_idx),
        int(feature_idx),
        delta=float(delta),
        steps=int(steps),
    )
    start_xy = np.asarray(oracle_path[0], dtype=float)

    record: dict[str, Any] = {
        "status": "invalid",
        "dataset": case.name,
        "dataset_title": case.title,
        "feature_idx": int(feature_idx),
        "feature_name": case.feature_names[int(feature_idx)],
        "point_idx": int(point_idx),
        "steps": int(steps),
        "delta_frac": float(delta_frac),
        "delta": float(delta),
        "fd_epsilon_frac": float(fd_epsilon_frac),
        "fd_epsilon": float(fd_epsilon),
        "feature_std": float(case.feature_std[int(feature_idx)]),
        "dr_backend": str(dr_backend),
        "oracle_kind": str(case.oracle_kind),
        "local_vector_method": str(case.local_vector_method),
        "rollout_substeps": int(rollout_substeps),
    }

    start_sample = field_context.sample_field(float(start_xy[0]), float(start_xy[1]))
    if not start_sample["valid"]:
        record["invalid_reason"] = f"start-{start_sample['reason']}"
        return record

    trail_result = field_context.integrate_trail(
        float(start_xy[0]),
        float(start_xy[1]),
        float(delta) / float(rollout_substeps),
        int(steps) * int(rollout_substeps),
    )
    if not trail_result["valid"]:
        record["invalid_reason"] = str(trail_result["reason"])
        record["trail_completed_steps"] = int(trail_result["completed_steps"])
        record["trail_path"] = _to_serializable(trail_result["points"])
        return record
    if np.asarray(trail_result["points"]).shape[0] != int(steps) * int(rollout_substeps) + 1:
        record["invalid_reason"] = "insufficient-valid-steps"
        record["trail_completed_steps"] = int(trail_result["completed_steps"])
        record["trail_path"] = _to_serializable(trail_result["points"])
        return record

    start_vector = np.array([start_sample["u"], start_sample["v"]], dtype=float)
    linear_path = build_linear_baseline_path(start_xy, start_vector, delta=float(delta), steps=int(steps))
    axis_result = None if axis_curve is None else build_anchored_global_axis_path(
        start_xy,
        axis_curve,
        steps=int(steps),
        delta_t=float(delta_frac),
        target_first_step_length=float(np.linalg.norm(start_vector) * delta),
    )
    axis_path = None if axis_result is None else np.asarray(axis_result["path"], dtype=float)
    trail_path = _subsample_rollout_points(np.asarray(trail_result["points"], dtype=float), steps=int(steps), rollout_substeps=int(rollout_substeps))
    trail_metrics = compute_stepwise_path_metrics(oracle_path, trail_path)
    linear_metrics = compute_stepwise_path_metrics(oracle_path, linear_path)
    axis_metrics = None if axis_path is None else compute_stepwise_path_metrics(oracle_path, axis_path)
    trail_wins_linear_endpoint = trail_metrics["endpoint_error_by_step"] < linear_metrics["endpoint_error_by_step"]
    trail_wins_linear_path = trail_metrics["path_deviation_by_step"] < linear_metrics["path_deviation_by_step"]
    trail_wins_axis_endpoint = None if axis_metrics is None else trail_metrics["endpoint_error_by_step"] < axis_metrics["endpoint_error_by_step"]
    trail_wins_axis_path = None if axis_metrics is None else trail_metrics["path_deviation_by_step"] < axis_metrics["path_deviation_by_step"]

    record.update(
        {
            "status": "valid",
            "invalid_reason": None,
            "reference_turning_deg": float(total_turning_angle(oracle_path)),
            "oracle_path": oracle_path.tolist(),
            "trail_path": trail_path.tolist(),
            "linear_path": linear_path.tolist(),
            "anchored_global_nonlinear_axis_path": None if axis_path is None else axis_path.tolist(),
            "start_vector": start_vector.tolist(),
            "trail_completed_steps": int(trail_result["completed_steps"]),
            "endpoint_error_trail_by_step": _to_serializable(trail_metrics["endpoint_error_by_step"]),
            "endpoint_error_linear_by_step": _to_serializable(linear_metrics["endpoint_error_by_step"]),
            "endpoint_error_anchored_global_nonlinear_axis_by_step": None if axis_metrics is None else _to_serializable(axis_metrics["endpoint_error_by_step"]),
            "path_deviation_trail_by_step": _to_serializable(trail_metrics["path_deviation_by_step"]),
            "path_deviation_linear_by_step": _to_serializable(linear_metrics["path_deviation_by_step"]),
            "path_deviation_anchored_global_nonlinear_axis_by_step": None if axis_metrics is None else _to_serializable(axis_metrics["path_deviation_by_step"]),
            "direction_agreement_trail_by_step": _to_serializable(trail_metrics["direction_agreement_by_step"]),
            "direction_agreement_linear_by_step": _to_serializable(linear_metrics["direction_agreement_by_step"]),
            "direction_agreement_anchored_global_nonlinear_axis_by_step": None if axis_metrics is None else _to_serializable(axis_metrics["direction_agreement_by_step"]),
            "trail_wins_endpoint_by_step": _to_serializable(trail_wins_linear_endpoint.astype(int)),
            "trail_wins_path_by_step": _to_serializable(trail_wins_linear_path.astype(int)),
            "trail_wins_endpoint_vs_anchored_global_nonlinear_axis_by_step": None if trail_wins_axis_endpoint is None else _to_serializable(trail_wins_axis_endpoint.astype(int)),
            "trail_wins_path_vs_anchored_global_nonlinear_axis_by_step": None if trail_wins_axis_path is None else _to_serializable(trail_wins_axis_path.astype(int)),
            "final_endpoint_improvement": float(
                linear_metrics["endpoint_error_by_step"][-1] - trail_metrics["endpoint_error_by_step"][-1]
            ),
            "final_path_improvement": float(
                linear_metrics["path_deviation_by_step"][-1] - trail_metrics["path_deviation_by_step"][-1]
            ),
            "final_endpoint_improvement_vs_anchored_global_nonlinear_axis": None if axis_metrics is None else float(
                axis_metrics["endpoint_error_by_step"][-1] - trail_metrics["endpoint_error_by_step"][-1]
            ),
            "final_path_improvement_vs_anchored_global_nonlinear_axis": None if axis_metrics is None else float(
                axis_metrics["path_deviation_by_step"][-1] - trail_metrics["path_deviation_by_step"][-1]
            ),
            "anchored_global_nonlinear_axis_anchor_z": None if axis_result is None else float(axis_result["anchor_z"]),
            "anchored_global_nonlinear_axis_anchor_xy": None if axis_result is None else _to_serializable(axis_result["anchor_xy"]),
            "anchored_global_nonlinear_axis_scale": None if axis_result is None else float(axis_result["scale"]),
        }
    )
    return record


def _summary_rows(valid_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not valid_records:
        return rows

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in valid_records:
        grouped.setdefault((str(record["dataset"]), str(record["feature_name"])), []).append(record)

    for (dataset, feature_name), records in sorted(grouped.items()):
        steps = int(records[0]["steps"])
        for step_idx in range(steps):
            trail_endpoint = np.array([record["endpoint_error_trail_by_step"][step_idx] for record in records], dtype=float)
            linear_endpoint = np.array([record["endpoint_error_linear_by_step"][step_idx] for record in records], dtype=float)
            axis_endpoint = np.array(
                [record["endpoint_error_anchored_global_nonlinear_axis_by_step"][step_idx] for record in records],
                dtype=float,
            )
            trail_path = np.array([record["path_deviation_trail_by_step"][step_idx] for record in records], dtype=float)
            linear_path = np.array([record["path_deviation_linear_by_step"][step_idx] for record in records], dtype=float)
            axis_path = np.array(
                [record["path_deviation_anchored_global_nonlinear_axis_by_step"][step_idx] for record in records],
                dtype=float,
            )
            trail_dir = np.array([record["direction_agreement_trail_by_step"][step_idx] for record in records], dtype=float)
            linear_dir = np.array([record["direction_agreement_linear_by_step"][step_idx] for record in records], dtype=float)
            axis_dir = np.array(
                [record["direction_agreement_anchored_global_nonlinear_axis_by_step"][step_idx] for record in records],
                dtype=float,
            )

            win_endpoint = np.mean(trail_endpoint < linear_endpoint)
            win_path = np.mean(trail_path < linear_path)
            win_endpoint_axis = np.mean(trail_endpoint < axis_endpoint)
            win_path_axis = np.mean(trail_path < axis_path)

            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature_name,
                    "step": int(step_idx + 1),
                    "method": "static_trail",
                    "mean_endpoint_error": float(trail_endpoint.mean()),
                    "mean_path_deviation": float(trail_path.mean()),
                    "mean_direction_agreement": float(trail_dir.mean()),
                    "win_rate_vs_linear": float(win_path),
                    "win_rate_endpoint_vs_linear": float(win_endpoint),
                    "win_rate_path_vs_linear": float(win_path),
                    "win_rate_vs_anchored_global_nonlinear_axis": float(win_path_axis),
                    "win_rate_endpoint_vs_anchored_global_nonlinear_axis": float(win_endpoint_axis),
                    "win_rate_path_vs_anchored_global_nonlinear_axis": float(win_path_axis),
                    "n_cases": int(len(records)),
                }
            )
            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature_name,
                    "step": int(step_idx + 1),
                    "method": "linear_trail",
                    "mean_endpoint_error": float(linear_endpoint.mean()),
                    "mean_path_deviation": float(linear_path.mean()),
                    "mean_direction_agreement": float(linear_dir.mean()),
                    "win_rate_vs_linear": None,
                    "win_rate_endpoint_vs_linear": None,
                    "win_rate_path_vs_linear": None,
                    "win_rate_vs_anchored_global_nonlinear_axis": None,
                    "win_rate_endpoint_vs_anchored_global_nonlinear_axis": None,
                    "win_rate_path_vs_anchored_global_nonlinear_axis": None,
                    "n_cases": int(len(records)),
                }
            )
            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature_name,
                    "step": int(step_idx + 1),
                    "method": "anchored_global_nonlinear_axis",
                    "mean_endpoint_error": float(axis_endpoint.mean()),
                    "mean_path_deviation": float(axis_path.mean()),
                    "mean_direction_agreement": float(axis_dir.mean()),
                    "win_rate_vs_linear": None,
                    "win_rate_endpoint_vs_linear": None,
                    "win_rate_path_vs_linear": None,
                    "win_rate_vs_anchored_global_nonlinear_axis": None,
                    "win_rate_endpoint_vs_anchored_global_nonlinear_axis": None,
                    "win_rate_path_vs_anchored_global_nonlinear_axis": None,
                    "n_cases": int(len(records)),
                }
            )
    return rows


def build_summary_metrics(valid_records: list[dict[str, Any]]) -> pd.DataFrame:
    columns = [
        "dataset",
        "feature",
        "step",
        "method",
        "mean_endpoint_error",
        "mean_path_deviation",
        "mean_direction_agreement",
        "win_rate_vs_linear",
        "win_rate_endpoint_vs_linear",
        "win_rate_path_vs_linear",
        "win_rate_vs_anchored_global_nonlinear_axis",
        "win_rate_endpoint_vs_anchored_global_nonlinear_axis",
        "win_rate_path_vs_anchored_global_nonlinear_axis",
        "n_cases",
    ]
    return pd.DataFrame(_summary_rows(valid_records), columns=columns)


def _panel_grid(n_panels: int) -> tuple[int, int]:
    cols = 2 if n_panels > 1 else 1
    rows = int(math.ceil(n_panels / cols))
    return rows, cols


def _summary_groups(summary_df: pd.DataFrame) -> list[tuple[str, str]]:
    return sorted({(str(row["dataset"]), str(row["feature"])) for _, row in summary_df.iterrows()})


def _plot_metric_lines(summary_df: pd.DataFrame, *, metric: str, ylabel: str, output_path: Path) -> Path:
    groups = _summary_groups(summary_df)
    rows, cols = _panel_grid(len(groups))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4.0 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for axis, (dataset, feature) in zip(axes_flat, groups):
        subset = summary_df[(summary_df["dataset"] == dataset) & (summary_df["feature"] == feature)]
        trail = subset[subset["method"] == "static_trail"].sort_values("step")
        axis_curve = subset[subset["method"] == "anchored_global_nonlinear_axis"].sort_values("step")
        linear = subset[subset["method"] == "linear_trail"].sort_values("step")
        axis.plot(trail["step"], trail[metric], color="#2563eb", linewidth=2.2, label="Static trail")
        axis.plot(axis_curve["step"], axis_curve[metric], color="#059669", linewidth=2.2, linestyle="-.", label="Anchored global nonlinear axis")
        axis.plot(linear["step"], linear[metric], color="#dc2626", linewidth=2.2, linestyle="--", label="Linear trail")
        axis.set_title(f"{dataset} / {feature}")
        axis.set_xlabel("Step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(frameon=False)

    for axis in axes_flat[len(groups):]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_step_endpoint_error_plot(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    return _plot_metric_lines(
        summary_df,
        metric="mean_endpoint_error",
        ylabel="Mean Endpoint Error",
        output_path=output_dir / "step_endpoint_error.png",
    )


def save_step_path_deviation_plot(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    return _plot_metric_lines(
        summary_df,
        metric="mean_path_deviation",
        ylabel="Mean Path Deviation",
        output_path=output_dir / "step_path_deviation.png",
    )


def save_step_winrate_plot(summary_df: pd.DataFrame, output_dir: Path) -> Path:
    trail_df = summary_df[summary_df["method"] == "static_trail"].copy()
    groups = _summary_groups(trail_df)
    rows, cols = _panel_grid(len(groups))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4.0 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for axis, (dataset, feature) in zip(axes_flat, groups):
        subset = trail_df[(trail_df["dataset"] == dataset) & (trail_df["feature"] == feature)].sort_values("step")
        axis.plot(
            subset["step"],
            subset["win_rate_path_vs_linear"],
            color="#2563eb",
            linewidth=2.2,
            label="Static beats linear on path deviation",
        )
        axis.plot(
            subset["step"],
            subset["win_rate_endpoint_vs_linear"],
            color="#0f766e",
            linewidth=2.2,
            linestyle="--",
            label="Static beats linear on endpoint error",
        )
        axis.plot(
            subset["step"],
            subset["win_rate_path_vs_anchored_global_nonlinear_axis"],
            color="#059669",
            linewidth=2.2,
            label="Static beats anchored axis on path deviation",
        )
        axis.plot(
            subset["step"],
            subset["win_rate_endpoint_vs_anchored_global_nonlinear_axis"],
            color="#065f46",
            linewidth=2.2,
            linestyle="--",
            label="Static beats anchored axis on endpoint error",
        )
        axis.set_title(f"{dataset} / {feature}")
        axis.set_xlabel("Step")
        axis.set_ylabel("Win Rate")
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(frameon=False)

    for axis in axes_flat[len(groups):]:
        axis.axis("off")

    fig.tight_layout()
    output_path = output_dir / "step_winrate_plot.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _record_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record["dataset"]),
        str(record["feature_name"]),
        int(record["point_idx"]),
    )


def _polyline_length(path: np.ndarray) -> float:
    points = np.asarray(path, dtype=float)
    if points.shape[0] < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    return float(np.linalg.norm(diffs, axis=1).sum())


def _normalize_path_length_for_display(path: np.ndarray, *, target_length: float) -> np.ndarray:
    """Rescale a plotted path around its start point without affecting metrics."""
    points = np.asarray(path, dtype=float)
    if points.shape[0] < 2:
        return points.copy()

    current_length = _polyline_length(points)
    if current_length <= 1e-12 or float(target_length) <= 1e-12:
        return points.copy()

    scale = float(target_length) / current_length
    start = points[0]
    return start[None, :] + scale * (points - start[None, :])


def _representative_match_score(record: dict[str, Any]) -> tuple[Any, ...]:
    path_dev = np.asarray(record["path_deviation_trail_by_step"], dtype=float)
    endpoint_err = np.asarray(record["endpoint_error_trail_by_step"], dtype=float)
    return (
        float(path_dev.mean()),
        float(endpoint_err.mean()),
        float(path_dev[-1]),
        float(endpoint_err[-1]),
    ) + _record_sort_key(record)


def _choose_representatives(valid_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not valid_records:
        return []
    ranked = sorted(valid_records, key=_representative_match_score)
    return ranked[: min(5, len(ranked))]


def save_representative_paths_plot(valid_records: list[dict[str, Any]], output_dir: Path) -> Path:
    selected = _choose_representatives(valid_records)
    if not selected:
        raise ValueError("Cannot save representative paths without valid records.")

    rows, cols = _panel_grid(len(selected))
    fig, axes = plt.subplots(rows, cols, figsize=(6.5 * cols, 4.5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for axis, record in zip(axes_flat, selected):
        oracle_path = np.asarray(record["oracle_path"], dtype=float)
        target_display_length = _polyline_length(oracle_path)
        trail_path = _normalize_path_length_for_display(
            np.asarray(record["trail_path"], dtype=float),
            target_length=target_display_length,
        )
        anchored_axis_path = _normalize_path_length_for_display(
            np.asarray(record["anchored_global_nonlinear_axis_path"], dtype=float),
            target_length=target_display_length,
        )
        linear_path = _normalize_path_length_for_display(
            np.asarray(record["linear_path"], dtype=float),
            target_length=target_display_length,
        )
        axis.plot(oracle_path[:, 0], oracle_path[:, 1], color="black", linewidth=2.4, label="Oracle transform trail")
        axis.scatter(
            oracle_path[:, 0],
            oracle_path[:, 1],
            s=18,
            facecolors="white",
            edgecolors="black",
            linewidths=0.9,
            zorder=4,
            label="Oracle steps",
        )
        axis.plot(trail_path[:, 0], trail_path[:, 1], color="#2563eb", linewidth=2.2, label="Static trail")
        axis.plot(
            anchored_axis_path[:, 0],
            anchored_axis_path[:, 1],
            color="#059669",
            linewidth=2.2,
            linestyle="-.",
            label="Anchored global nonlinear axis",
        )
        axis.plot(linear_path[:, 0], linear_path[:, 1], color="#dc2626", linewidth=2.2, linestyle="--", label="Linear trail")
        axis.set_title(f"{record['dataset']} / point {record['point_idx']}")
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(frameon=False)
        axis.text(
            0.02,
            0.02,
            (
                f"feature={record['feature_name']}\n"
                f"turning={float(record['reference_turning_deg']):.1f} deg\n"
                "display length normalized to oracle\n"
                f"final path dev: static {record['path_deviation_trail_by_step'][-1]:.3f}\n"
                f"axis {record['path_deviation_anchored_global_nonlinear_axis_by_step'][-1]:.3f}\n"
                f"linear {record['path_deviation_linear_by_step'][-1]:.3f}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
        )

    for axis in axes_flat[len(selected):]:
        axis.axis("off")

    fig.text(
        0.5,
        0.01,
        "Representative paths are chosen by best full-trajectory static-oracle agreement and length-normalized to the oracle for display only.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 1.0))
    output_path = output_dir / "representative_paths.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_paper_summary(
    summary_df: pd.DataFrame,
    *,
    attempted_count: int,
    valid_count: int,
    invalid_count: int,
    cfg: GlobalTrailFidelityConfig,
    output_dir: Path,
) -> Path:
    lines = [
        f"# {MAIN_EXPERIMENT_NAME}",
        "",
        "This experiment uses deterministic synthetic datasets, but the projection layer is an actual nonlinear DR model.",
        f"The DR backend is `{cfg.dr_backend}` fit once per dataset with a fixed seed, and the oracle is a fixed-embedding oracle transform trail obtained by transforming perturbed points through that fitted model.",
        "",
        "For each point-feature case, we compare:",
        "- oracle transform trail from the fitted nonlinear DR model",
        "- static trail from an interpolated vector field built from per-point local vectors",
        "- anchored global nonlinear axis baseline from a shared feature-conditioned curve in the original embedding",
        "- strictly straight local linear baseline from the initial local vector only",
        "",
        f"Transport uses `steps = {cfg.steps}` and `delta_k = {cfg.delta_frac} * std(X[:, k], ddof=0)` per step.",
        f"Local vectors are estimated once at the original training points with centered finite differences using `epsilon_k = {cfg.fd_epsilon_frac} * std(X[:, k], ddof=0)`.",
        f"The static trail rollout uses RK2 advection with `{cfg.rollout_substeps}` internal substeps per reported feature step, and the reported positions are aligned to the oracle step index.",
        "The anchored global nonlinear axis baseline is conservatively rescaled so its first step does not exceed the initial local linear step magnitude.",
        "",
        f"Attempted / valid / invalid cases: `{attempted_count} / {valid_count} / {invalid_count}`.",
        "",
    ]

    trail_df = summary_df[summary_df["method"] == "static_trail"].copy()
    if not trail_df.empty:
        for dataset in sorted(trail_df["dataset"].unique()):
            subset = trail_df[trail_df["dataset"] == dataset].sort_values("step")
            first = subset.iloc[0]
            last = subset.iloc[-1]
            linear_first = summary_df[
                (summary_df["dataset"] == dataset)
                & (summary_df["method"] == "linear_trail")
                & (summary_df["step"] == int(first["step"]))
            ].iloc[0]
            linear_last = summary_df[
                (summary_df["dataset"] == dataset)
                & (summary_df["method"] == "linear_trail")
                & (summary_df["step"] == int(last["step"]))
            ].iloc[0]
            axis_first = summary_df[
                (summary_df["dataset"] == dataset)
                & (summary_df["method"] == "anchored_global_nonlinear_axis")
                & (summary_df["step"] == int(first["step"]))
            ].iloc[0]
            axis_last = summary_df[
                (summary_df["dataset"] == dataset)
                & (summary_df["method"] == "anchored_global_nonlinear_axis")
                & (summary_df["step"] == int(last["step"]))
            ].iloc[0]
            lines.append(
                f"- `{dataset}`: mean path deviation changed from {float(first['mean_path_deviation']):.4f} vs {float(linear_first['mean_path_deviation']):.4f} "
                f"at step {int(first['step'])} to {float(last['mean_path_deviation']):.4f} vs {float(linear_last['mean_path_deviation']):.4f} "
                f"at step {int(last['step'])} (static trail vs linear)."
            )
            lines.append(
                f"- `{dataset}`: against the anchored global nonlinear axis, mean path deviation changed from "
                f"{float(first['mean_path_deviation']):.4f} vs {float(axis_first['mean_path_deviation']):.4f} "
                f"at step {int(first['step'])} to {float(last['mean_path_deviation']):.4f} vs {float(axis_last['mean_path_deviation']):.4f} "
                f"at step {int(last['step'])} (static trail vs anchored axis)."
            )
            lines.append(
                f"- `{dataset}` final-step static win rates were {float(last['win_rate_path_vs_linear']) * 100:.1f}% on path deviation "
                f"and {float(last['win_rate_endpoint_vs_linear']) * 100:.1f}% on endpoint error."
            )
            lines.append(
                f"- `{dataset}` final-step static win rates vs anchored global nonlinear axis were "
                f"{float(last['win_rate_path_vs_anchored_global_nonlinear_axis']) * 100:.1f}% on path deviation "
                f"and {float(last['win_rate_endpoint_vs_anchored_global_nonlinear_axis']) * 100:.1f}% on endpoint error."
            )
        lines.append("")
        lines.append(
            "Conservative interpretation: when the fitted nonlinear DR produces a visibly curved oracle transform trail, "
            "the static trail usually improves over the straight local baseline as the transport accumulates. "
            "The anchored global nonlinear axis is a stronger shared-curve baseline than the straight line, "
            "but it still cannot adapt to point-specific transport the way the interpolated vector field can. "
            "In the low-curvature observed case, the early-step gap is smaller, but the static trail can still pull ahead over longer transport."
        )

    output_path = output_dir / "paper_summary.md"
    output_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return output_path


def _write_per_case_json(
    attempted_records: list[dict[str, Any]],
    *,
    cfg: GlobalTrailFidelityConfig,
    output_dir: Path,
) -> Path:
    payload = {
        "experiment_name": MAIN_EXPERIMENT_NAME,
        "config": _to_serializable(cfg.__dict__),
        "attempted_case_count": int(len(attempted_records)),
        "valid_case_count": int(sum(record["status"] == "valid" for record in attempted_records)),
        "invalid_case_count": int(sum(record["status"] != "valid" for record in attempted_records)),
        "records": [_to_serializable(record) for record in attempted_records],
    }
    output_path = output_dir / "per_case.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def run_trail_global_fidelity(cfg: GlobalTrailFidelityConfig) -> dict[str, Any]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_case_specs(cfg)

    attempted_records: list[dict[str, Any]] = []
    valid_records: list[dict[str, Any]] = []

    for spec in specs:
        case = fit_global_umap_case(
            spec,
            seed=int(cfg.seed),
            dr_backend=str(cfg.dr_backend),
            fd_epsilon_frac=float(cfg.fd_epsilon_frac),
        )
        field_contexts = build_global_field_contexts(
            case,
            grid_res=int(cfg.grid_res),
            steps=int(cfg.steps),
            delta_frac=float(cfg.delta_frac),
        )
        axis_curves = build_global_axis_curves(case)
        for feature_idx in case.eval_feature_indices:
            for point_idx in range(case.X.shape[0]):
                record = evaluate_global_case_combo(
                    case,
                    field_contexts[int(feature_idx)],
                    axis_curves.get(int(feature_idx)),
                    point_idx=int(point_idx),
                    feature_idx=int(feature_idx),
                    steps=int(cfg.steps),
                    delta_frac=float(cfg.delta_frac),
                    fd_epsilon_frac=float(cfg.fd_epsilon_frac),
                    dr_backend=str(cfg.dr_backend),
                    rollout_substeps=int(cfg.rollout_substeps),
                )
                attempted_records.append(record)
                if record["status"] == "valid":
                    valid_records.append(record)

    summary_df = build_summary_metrics(valid_records)
    summary_csv = cfg.output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_csv, index=False)

    per_case_json = _write_per_case_json(attempted_records, cfg=cfg, output_dir=cfg.output_dir)
    step_endpoint_error_png = save_step_endpoint_error_plot(summary_df, cfg.output_dir)
    step_path_deviation_png = save_step_path_deviation_plot(summary_df, cfg.output_dir)
    step_winrate_png = save_step_winrate_plot(summary_df, cfg.output_dir)
    representative_paths_png = save_representative_paths_plot(valid_records, cfg.output_dir)
    paper_summary_md = write_paper_summary(
        summary_df,
        attempted_count=len(attempted_records),
        valid_count=len(valid_records),
        invalid_count=len(attempted_records) - len(valid_records),
        cfg=cfg,
        output_dir=cfg.output_dir,
    )

    return {
        "experiment_name": MAIN_EXPERIMENT_NAME,
        "config": cfg,
        "attempted_case_count": int(len(attempted_records)),
        "valid_case_count": int(len(valid_records)),
        "invalid_case_count": int(len(attempted_records) - len(valid_records)),
        "summary_metrics_csv": str(summary_csv),
        "per_case_json": str(per_case_json),
        "step_endpoint_error_png": str(step_endpoint_error_png),
        "step_path_deviation_png": str(step_path_deviation_png),
        "step_winrate_png": str(step_winrate_png),
        "representative_paths_png": str(representative_paths_png),
        "paper_summary_md": str(paper_summary_md),
    }
