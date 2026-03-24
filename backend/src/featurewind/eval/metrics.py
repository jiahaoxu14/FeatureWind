from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _as_path(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("path must have shape (n, 2)")
    if arr.shape[0] < 2:
        raise ValueError("path must contain at least two points")
    return arr


def path_length(path: np.ndarray) -> float:
    pts = _as_path(path)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def resample_path(path: np.ndarray, num_samples: int) -> np.ndarray:
    pts = _as_path(path)
    if num_samples < 2:
        raise ValueError("num_samples must be >= 2")
    if pts.shape[0] == num_samples:
        return pts.copy()

    deltas = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = float(cumulative[-1])
    if total <= 1e-12:
        return np.repeat(pts[:1], num_samples, axis=0)

    targets = np.linspace(0.0, total, int(num_samples))
    out = np.empty((int(num_samples), 2), dtype=float)

    for idx, target in enumerate(targets):
        seg_idx = int(np.searchsorted(cumulative, target, side="right") - 1)
        seg_idx = max(0, min(seg_idx, len(seg_lengths) - 1))
        seg_start = cumulative[seg_idx]
        seg_len = seg_lengths[seg_idx]
        if seg_len <= 1e-12:
            out[idx] = pts[seg_idx]
            continue
        alpha = (target - seg_start) / seg_len
        out[idx] = pts[seg_idx] + alpha * (pts[seg_idx + 1] - pts[seg_idx])
    return out


def endpoint_error(predicted: np.ndarray, reference: np.ndarray) -> float:
    pred = _as_path(predicted)
    ref = _as_path(reference)
    return float(np.linalg.norm(pred[-1] - ref[-1]))


def aligned_step_endpoint_errors(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    pred = _as_path(predicted)
    ref = _as_path(reference)
    if pred.shape != ref.shape:
        raise ValueError("aligned step endpoint errors require matching path shapes")
    return np.linalg.norm(pred[1:] - ref[1:], axis=1)


def mean_path_deviation(predicted: np.ndarray, reference: np.ndarray, samples: int | None = None) -> float:
    pred = _as_path(predicted)
    ref = _as_path(reference)
    n_samples = samples or max(pred.shape[0], ref.shape[0])
    pred_resampled = resample_path(pred, n_samples)
    ref_resampled = resample_path(ref, n_samples)
    return float(np.linalg.norm(pred_resampled - ref_resampled, axis=1).mean())


def _segment_directions(path: np.ndarray) -> np.ndarray:
    pts = _as_path(path)
    deltas = np.diff(pts, axis=0)
    norms = np.linalg.norm(deltas, axis=1, keepdims=True)
    norms = np.where(norms > 1e-12, norms, 1.0)
    return deltas / norms


def aligned_prefix_path_deviations(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    pred = _as_path(predicted)
    ref = _as_path(reference)
    if pred.shape != ref.shape:
        raise ValueError("aligned prefix path deviations require matching path shapes")
    point_errors = np.linalg.norm(pred - ref, axis=1)
    prefix_sums = np.cumsum(point_errors)
    counts = np.arange(1, pred.shape[0] + 1, dtype=float)
    return prefix_sums[1:] / counts[1:]


def direction_agreement(predicted: np.ndarray, reference: np.ndarray, samples: int | None = None) -> float:
    pred = _as_path(predicted)
    ref = _as_path(reference)
    n_samples = samples or max(pred.shape[0], ref.shape[0])
    pred_dirs = _segment_directions(resample_path(pred, n_samples))
    ref_dirs = _segment_directions(resample_path(ref, n_samples))
    cosines = np.sum(pred_dirs * ref_dirs, axis=1)
    cosines = np.clip(cosines, -1.0, 1.0)
    return float(cosines.mean())


def aligned_step_direction_agreement(predicted: np.ndarray, reference: np.ndarray) -> np.ndarray:
    pred = _as_path(predicted)
    ref = _as_path(reference)
    if pred.shape != ref.shape:
        raise ValueError("aligned step direction agreement requires matching path shapes")
    pred_dirs = _segment_directions(pred)
    ref_dirs = _segment_directions(ref)
    cosines = np.sum(pred_dirs * ref_dirs, axis=1)
    return np.clip(cosines, -1.0, 1.0)


def total_turning_angle(path: np.ndarray) -> float:
    directions = _segment_directions(path)
    if directions.shape[0] < 2:
        return 0.0
    dots = np.sum(directions[:-1] * directions[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    radians = np.arccos(dots)
    return float(np.degrees(np.abs(radians).sum()))


def curvature_bin(turning_degrees: float) -> str:
    value = float(turning_degrees)
    if value < 15.0:
        return "low"
    if value <= 45.0:
        return "medium"
    return "high"


def compute_case_metrics(reference: np.ndarray, trail: np.ndarray, baseline: np.ndarray) -> dict[str, float | bool | str]:
    ref = _as_path(reference)
    trail_path = _as_path(trail)
    baseline_path = _as_path(baseline)

    turning_deg = total_turning_angle(ref)
    trail_endpoint = endpoint_error(trail_path, ref)
    baseline_endpoint = endpoint_error(baseline_path, ref)
    trail_path_dev = mean_path_deviation(trail_path, ref)
    baseline_path_dev = mean_path_deviation(baseline_path, ref)

    return {
        "endpoint_error_trail": trail_endpoint,
        "endpoint_error_baseline": baseline_endpoint,
        "mean_path_deviation_trail": trail_path_dev,
        "mean_path_deviation_baseline": baseline_path_dev,
        "direction_agreement_trail": direction_agreement(trail_path, ref),
        "direction_agreement_baseline": direction_agreement(baseline_path, ref),
        "reference_turning_deg": turning_deg,
        "curvature_bin": curvature_bin(turning_deg),
        "trail_wins_endpoint": bool(trail_endpoint < baseline_endpoint),
        "trail_wins_mean_path": bool(trail_path_dev < baseline_path_dev),
    }


def compute_stepwise_path_metrics(reference: np.ndarray, predicted: np.ndarray) -> dict[str, np.ndarray]:
    ref = _as_path(reference)
    pred = _as_path(predicted)
    if pred.shape != ref.shape:
        raise ValueError("stepwise path metrics require oracle and predicted paths with matching shapes")

    return {
        "endpoint_error_by_step": aligned_step_endpoint_errors(pred, ref),
        "path_deviation_by_step": aligned_prefix_path_deviations(pred, ref),
        "direction_agreement_by_step": aligned_step_direction_agreement(pred, ref),
    }


def compute_stepwise_case_metrics(reference: np.ndarray, trail: np.ndarray, baseline: np.ndarray) -> dict[str, np.ndarray]:
    ref = _as_path(reference)
    trail_path = _as_path(trail)
    baseline_path = _as_path(baseline)
    if trail_path.shape != ref.shape or baseline_path.shape != ref.shape:
        raise ValueError("stepwise metrics require oracle, trail, and baseline paths with matching shapes")

    trail_metrics = compute_stepwise_path_metrics(ref, trail_path)
    baseline_metrics = compute_stepwise_path_metrics(ref, baseline_path)
    trail_endpoint = trail_metrics["endpoint_error_by_step"]
    baseline_endpoint = baseline_metrics["endpoint_error_by_step"]
    trail_path_dev = trail_metrics["path_deviation_by_step"]
    baseline_path_dev = baseline_metrics["path_deviation_by_step"]

    return {
        "endpoint_error_trail_by_step": trail_endpoint,
        "endpoint_error_baseline_by_step": baseline_endpoint,
        "path_deviation_trail_by_step": trail_path_dev,
        "path_deviation_baseline_by_step": baseline_path_dev,
        "direction_agreement_trail_by_step": trail_metrics["direction_agreement_by_step"],
        "direction_agreement_baseline_by_step": baseline_metrics["direction_agreement_by_step"],
        "trail_wins_endpoint_by_step": trail_endpoint < baseline_endpoint,
        "trail_wins_path_by_step": trail_path_dev < baseline_path_dev,
    }
