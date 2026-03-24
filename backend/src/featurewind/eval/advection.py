from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


FIELD_EPS = 1e-8


def bilinear_sample(grid: np.ndarray, gx: float, gy: float) -> float:
    """Mirror the frontend's bilinear grid sampling in Python."""
    arr = np.asarray(grid, dtype=float)
    height, width = arr.shape
    x = float(np.clip(gx, 0.0, max(width - 1, 0)))
    y = float(np.clip(gy, 0.0, max(height - 1, 0)))

    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = min(width - 1, x0 + 1)
    y1 = min(height - 1, y0 + 1)
    tx = x - x0
    ty = y - y0

    v00 = arr[y0, x0]
    v10 = arr[y0, x1]
    v01 = arr[y1, x0]
    v11 = arr[y1, x1]

    top = (1.0 - tx) * v00 + tx * v10
    bottom = (1.0 - tx) * v01 + tx * v11
    return float((1.0 - ty) * top + ty * bottom)


def robust_p99(magnitudes: np.ndarray) -> float:
    arr = np.asarray(magnitudes, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    p99 = float(np.percentile(arr, 99))
    return p99 if p99 > 0 else 1.0


def _advect_rk2_generic(field_context: Any, x: float, y: float, delta_feature: float) -> dict[str, Any]:
    start = field_context.sample_field(x, y)
    if not start["valid"]:
        return {
            "valid": False,
            "reason": f"start-{start['reason']}",
            "start": start,
            "midpoint": None,
            "end": None,
            "midX": None,
            "midY": None,
            "nx": None,
            "ny": None,
        }

    mid_x = float(x + start["u"] * delta_feature * 0.5)
    mid_y = float(y + start["v"] * delta_feature * 0.5)
    midpoint = field_context.sample_field(mid_x, mid_y)
    if not midpoint["valid"]:
        return {
            "valid": False,
            "reason": f"midpoint-{midpoint['reason']}",
            "start": start,
            "midpoint": midpoint,
            "end": None,
            "midX": mid_x,
            "midY": mid_y,
            "nx": None,
            "ny": None,
        }

    nx = float(x + midpoint["u"] * delta_feature)
    ny = float(y + midpoint["v"] * delta_feature)
    if not (np.isfinite(nx) and np.isfinite(ny)):
        return {
            "valid": False,
            "reason": "nonfinite-endpoint",
            "start": start,
            "midpoint": midpoint,
            "end": None,
            "midX": mid_x,
            "midY": mid_y,
            "nx": nx,
            "ny": ny,
        }

    end = field_context.sample_field(nx, ny)
    if not end["valid"]:
        return {
            "valid": False,
            "reason": f"end-{end['reason']}",
            "start": start,
            "midpoint": midpoint,
            "end": end,
            "midX": mid_x,
            "midY": mid_y,
            "nx": nx,
            "ny": ny,
        }

    return {
        "valid": True,
        "reason": None,
        "start": start,
        "midpoint": midpoint,
        "end": end,
        "midX": mid_x,
        "midY": mid_y,
        "nx": nx,
        "ny": ny,
        "x": nx,
        "y": ny,
    }


def _integrate_trail_generic(field_context: Any, start_x: float, start_y: float, delta_feature: float, steps: int) -> dict[str, Any]:
    points = [np.array([float(start_x), float(start_y)], dtype=float)]
    step_records: list[dict[str, Any]] = []

    x = float(start_x)
    y = float(start_y)
    for step_idx in range(int(steps)):
        advected = field_context.advect_rk2(x, y, delta_feature)
        record = {
            "step": step_idx,
            "x": x,
            "y": y,
            "reason": advected["reason"],
            "midX": advected.get("midX"),
            "midY": advected.get("midY"),
            "nx": advected.get("nx"),
            "ny": advected.get("ny"),
        }
        step_records.append(record)
        if not advected["valid"]:
            return {
                "valid": False,
                "reason": advected["reason"],
                "points": np.asarray(points, dtype=float),
                "completed_steps": step_idx,
                "step_records": step_records,
            }
        x = float(advected["x"])
        y = float(advected["y"])
        points.append(np.array([x, y], dtype=float))

    return {
        "valid": True,
        "reason": None,
        "points": np.asarray(points, dtype=float),
        "completed_steps": int(steps),
        "step_records": step_records,
    }


@dataclass
class FieldContext:
    """Single-feature 2D vector field with FeatureWind-like sampling semantics."""

    bbox: tuple[float, float, float, float]
    u_grid: np.ndarray
    v_grid: np.ndarray
    unmasked: np.ndarray | None = None
    weak_threshold: float = 1e-6
    p99: float = 1.0

    def __post_init__(self) -> None:
        self.u_grid = np.asarray(self.u_grid, dtype=float)
        self.v_grid = np.asarray(self.v_grid, dtype=float)
        if self.u_grid.shape != self.v_grid.shape:
            raise ValueError("u_grid and v_grid must have the same shape")
        if self.unmasked is not None:
            self.unmasked = np.asarray(self.unmasked, dtype=bool)
            if self.unmasked.shape != self.u_grid.shape:
                raise ValueError("unmasked mask must match grid shape")
        self.height, self.width = self.u_grid.shape
        self.xmin, self.xmax, self.ymin, self.ymax = map(float, self.bbox)

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
            return None
        j = int(np.floor((x - self.xmin) / (self.xmax - self.xmin) * self.width))
        i = int(np.floor((y - self.ymin) / (self.ymax - self.ymin) * self.height))
        j = max(0, min(self.width - 1, j))
        i = max(0, min(self.height - 1, i))
        return i, j

    def world_to_grid(self, x: float, y: float) -> tuple[float, float]:
        gx = (x - self.xmin) / (self.xmax - self.xmin) * (self.width - 1)
        gy = (y - self.ymin) / (self.ymax - self.ymin) * (self.height - 1)
        return float(gx), float(gy)

    def is_masked_cell(self, i: int, j: int) -> bool:
        if i < 0 or i >= self.height or j < 0 or j >= self.width:
            return True
        if self.unmasked is None:
            return False
        return not bool(self.unmasked[i, j])

    def sample_field(self, x: float, y: float) -> dict[str, Any]:
        cell = self.world_to_cell(x, y)
        if cell is None:
            return {"valid": False, "reason": "out-of-bbox", "u": 0.0, "v": 0.0, "mag": 0.0}
        if self.is_masked_cell(*cell):
            return {"valid": False, "reason": "masked-support", "u": 0.0, "v": 0.0, "mag": 0.0}

        gx, gy = self.world_to_grid(x, y)
        u = bilinear_sample(self.u_grid, gx, gy)
        v = bilinear_sample(self.v_grid, gx, gy)
        mag = float(np.hypot(u, v))

        if not np.isfinite(mag) or mag <= FIELD_EPS:
            return {"valid": False, "reason": "zero-or-nonfinite-magnitude", "u": u, "v": v, "mag": mag}
        if mag < float(self.weak_threshold):
            return {"valid": False, "reason": "below-weak-threshold", "u": u, "v": v, "mag": mag}
        return {"valid": True, "reason": None, "u": u, "v": v, "mag": mag}

    def check_start_eligibility(self, x: float, y: float, *, point_idx: int | None = None) -> dict[str, Any]:
        return {"valid": True, "reason": None}

    def advect_rk2(self, x: float, y: float, delta_feature: float) -> dict[str, Any]:
        return _advect_rk2_generic(self, x, y, delta_feature)

    def integrate_trail(self, start_x: float, start_y: float, delta_feature: float, steps: int) -> dict[str, Any]:
        return _integrate_trail_generic(self, start_x, start_y, delta_feature, steps)


@dataclass
class PointCloudFieldContext:
    """Evaluation-only smooth field sampler from pointwise gradients in embedding space."""

    positions: np.ndarray
    vectors: np.ndarray
    bbox: tuple[float, float, float, float]
    weak_threshold: float = 1e-6
    p99: float = 1.0
    k_neighbors: int = 12
    support_k: int = 8
    support_percentile: float = 0.85
    query_support_factor: float = 1.5
    bbox_margin_scale: float = 0.5

    def __post_init__(self) -> None:
        self.positions = np.asarray(self.positions, dtype=float)
        self.vectors = np.asarray(self.vectors, dtype=float)
        if self.positions.ndim != 2 or self.positions.shape[1] != 2:
            raise ValueError("positions must have shape (n, 2)")
        if self.vectors.shape != self.positions.shape:
            raise ValueError("vectors must have shape (n, 2)")
        self.xmin, self.xmax, self.ymin, self.ymax = map(float, self.bbox)
        self.n_points = int(self.positions.shape[0])
        self.k_neighbors = max(1, min(int(self.k_neighbors), self.n_points))
        self.support_k = max(1, min(int(self.support_k), self.n_points))

        diff = self.positions[:, None, :] - self.positions[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        if self.n_points > 1:
            dist = dist.copy()
            np.fill_diagonal(dist, np.inf)
            support_index = max(0, self.support_k - 1)
            kth_sorted = np.partition(dist, support_index, axis=1)[:, support_index]
        else:
            kth_sorted = np.zeros(1, dtype=float)
        self.anchor_support_radius = np.asarray(kth_sorted, dtype=float)
        percentile = np.clip(float(self.support_percentile) * 100.0, 0.0, 100.0)
        self.interior_support_radius = float(np.percentile(self.anchor_support_radius, percentile))
        self.query_support_limit = max(self.interior_support_radius * float(self.query_support_factor), 1e-6)

    def _distance_to_bbox_edge(self, x: float, y: float) -> float:
        return float(min(x - self.xmin, self.xmax - x, y - self.ymin, self.ymax - y))

    def _query_neighbors(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        query = np.array([float(x), float(y)], dtype=float)
        dists = np.linalg.norm(self.positions - query[None, :], axis=1)
        if self.k_neighbors >= self.n_points:
            order = np.argsort(dists)
        else:
            order = np.argpartition(dists, self.k_neighbors - 1)[: self.k_neighbors]
            order = order[np.argsort(dists[order])]
        return order, dists[order]

    def check_start_eligibility(self, x: float, y: float, *, point_idx: int | None = None) -> dict[str, Any]:
        if point_idx is None or point_idx < 0 or point_idx >= self.n_points:
            _, dists = self._query_neighbors(x, y)
            local_support_radius = float(dists[min(len(dists) - 1, self.support_k - 1)])
        else:
            local_support_radius = float(self.anchor_support_radius[int(point_idx)])
        edge_margin = self._distance_to_bbox_edge(float(x), float(y))
        payload = {
            "local_support_radius": local_support_radius,
            "interior_support_radius": float(self.interior_support_radius),
            "query_support_limit": float(self.query_support_limit),
            "bbox_edge_margin": edge_margin,
        }
        if local_support_radius > float(self.interior_support_radius):
            return {"valid": False, "reason": "outside-eval-support-margin", **payload}
        if edge_margin < float(self.bbox_margin_scale) * max(local_support_radius, 1e-6):
            return {"valid": False, "reason": "near-eval-bbox-edge", **payload}
        return {"valid": True, "reason": None, **payload}

    def sample_field(self, x: float, y: float) -> dict[str, Any]:
        if not (np.isfinite(x) and np.isfinite(y)):
            return {"valid": False, "reason": "nonfinite-query", "u": 0.0, "v": 0.0, "mag": 0.0}
        if x < self.xmin or x > self.xmax or y < self.ymin or y > self.ymax:
            return {"valid": False, "reason": "out-of-bbox", "u": 0.0, "v": 0.0, "mag": 0.0}

        order, dists = self._query_neighbors(float(x), float(y))
        kth_dist = float(dists[-1])
        if kth_dist > float(self.query_support_limit):
            return {
                "valid": False,
                "reason": "outside-eval-support",
                "u": 0.0,
                "v": 0.0,
                "mag": 0.0,
                "kth_neighbor_dist": kth_dist,
                "query_support_limit": float(self.query_support_limit),
            }

        sigma = max(kth_dist, float(self.interior_support_radius) * 0.5, 1e-6)
        weights = np.exp(-0.5 * (dists / sigma) ** 2)
        weight_sum = float(weights.sum())
        if weight_sum <= FIELD_EPS or not np.isfinite(weight_sum):
            return {
                "valid": False,
                "reason": "outside-eval-support",
                "u": 0.0,
                "v": 0.0,
                "mag": 0.0,
                "kth_neighbor_dist": kth_dist,
                "query_support_limit": float(self.query_support_limit),
            }

        vector = np.sum(self.vectors[order] * (weights / weight_sum)[:, None], axis=0)
        u = float(vector[0])
        v = float(vector[1])
        mag = float(np.hypot(u, v))
        payload = {
            "u": u,
            "v": v,
            "mag": mag,
            "kth_neighbor_dist": kth_dist,
            "query_support_limit": float(self.query_support_limit),
            "weight_sum": weight_sum,
        }
        if not np.isfinite(mag) or mag <= FIELD_EPS:
            return {"valid": False, "reason": "zero-or-nonfinite-magnitude", **payload}
        if mag < float(self.weak_threshold):
            return {"valid": False, "reason": "below-weak-threshold", **payload}
        return {"valid": True, "reason": None, **payload}

    def advect_rk2(self, x: float, y: float, delta_feature: float) -> dict[str, Any]:
        return _advect_rk2_generic(self, x, y, delta_feature)

    def integrate_trail(self, start_x: float, start_y: float, delta_feature: float, steps: int) -> dict[str, Any]:
        return _integrate_trail_generic(self, start_x, start_y, delta_feature, steps)


def build_field_context(
    u_grid: np.ndarray,
    v_grid: np.ndarray,
    bbox: tuple[float, float, float, float],
    final_mask: np.ndarray | None = None,
) -> FieldContext:
    """Create a field context with the same weak-threshold heuristic as the UI."""
    mag = np.hypot(np.asarray(u_grid, dtype=float), np.asarray(v_grid, dtype=float))
    p99 = robust_p99(mag)
    weak_threshold = max(1e-6, 0.015 * p99)
    unmasked = None if final_mask is None else np.logical_not(np.asarray(final_mask, dtype=bool))
    return FieldContext(
        bbox=tuple(map(float, bbox)),
        u_grid=np.asarray(u_grid, dtype=float),
        v_grid=np.asarray(v_grid, dtype=float),
        unmasked=unmasked,
        weak_threshold=weak_threshold,
        p99=p99,
    )


def build_point_cloud_field_context(
    positions: np.ndarray,
    vectors: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    k_neighbors: int = 12,
    support_k: int = 8,
    support_percentile: float = 0.85,
    query_support_factor: float = 1.5,
    bbox_margin_scale: float = 0.5,
) -> PointCloudFieldContext:
    magnitudes = np.linalg.norm(np.asarray(vectors, dtype=float), axis=1)
    p99 = robust_p99(magnitudes)
    weak_threshold = max(1e-6, 0.015 * p99)
    return PointCloudFieldContext(
        positions=np.asarray(positions, dtype=float),
        vectors=np.asarray(vectors, dtype=float),
        bbox=tuple(map(float, bbox)),
        weak_threshold=weak_threshold,
        p99=p99,
        k_neighbors=int(k_neighbors),
        support_k=int(support_k),
        support_percentile=float(support_percentile),
        query_support_factor=float(query_support_factor),
        bbox_margin_scale=float(bbox_margin_scale),
    )
