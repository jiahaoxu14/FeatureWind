from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Georgia"]
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np


LABEL_PALETTE = [
    "#93c5fd",
    "#fde68a",
    "#a7f3d0",
    "#fbcfe8",
    "#bfdbfe",
    "#fef08a",
    "#ddd6fe",
    "#bae6fd",
    "#d9f99d",
    "#fecdd3",
]

REFERENCE_TRAIL_COLOR = "#111111"
CONDITION_TRAIL_COLOR = "#96360e"
FIELD_ARROW_COLOR = "#6b7280"
BACKGROUND_COLOR = "#ffffff"
GRIDLINE_COLOR = "#000000"
MASKED_CELL_COLOR = "#d4d4d8"
TEXT_FONT_FAMILY = "Georgia"

FIXED_ADVECTION_STEP_SEC = 1.0 / 60.0
DISPLAY_SPEED_PX_BASE = 120.0
FIELD_EPS = 1e-8
COMMON_LATTICE_RES = 100
TRAIL_SAMPLE_COUNT = 50
DEFAULT_CANVAS_PX = 600
DEFAULT_SPEED_SCALE = 1.0
DEFAULT_MAX_FIGURES = 8


@dataclass(frozen=True)
class SensitivityCondition:
    condition_id: str
    label: str
    grid_res: int
    interpolation_method: str
    mask_radius: int
    is_reference: bool = False


@dataclass(frozen=True)
class SeedsDisplayCase:
    tmap_path: Path
    positions: np.ndarray
    grad_vectors: np.ndarray
    feature_names: list[str]
    point_labels: list[str] | None
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class CommonLattice:
    bbox: tuple[float, float, float, float]
    resolution: int
    grid_x: np.ndarray
    grid_y: np.ndarray
    points: np.ndarray


@dataclass(frozen=True)
class CommonLatticeSamples:
    support: np.ndarray
    vectors: np.ndarray


@dataclass(frozen=True)
class StaticTrailResult:
    condition_id: str
    point_index: int
    points: np.ndarray
    stop_reason: str
    stop_step: int

    @property
    def point_count(self) -> int:
        return int(self.points.shape[0])

    @property
    def valid(self) -> bool:
        return self.point_count >= 2


@dataclass(frozen=True)
class ConditionMetrics:
    condition_id: str
    condition_label: str
    grid_res: int
    interpolation_method: str
    mask_radius: int
    is_reference: bool
    point_index: int
    support_iou: float
    field_cosine_mean: float
    trail_mean_deviation: float
    trail_endpoint_shift: float
    trail_stop_reason: str
    trail_point_count: int


@dataclass(frozen=True)
class AnalysisResult:
    output_dir: Path
    feature_name: str
    point_index: int
    conditions: list[SensitivityCondition]
    metrics: list[ConditionMetrics]
    figure_png: Path | None
    figure_pdf: Path | None
    panel_pngs: list[Path]
    panel_pdfs: list[Path]
    metrics_csv: Path
    summary_md: Path


@dataclass(frozen=True)
class GridConditionContext:
    condition: SensitivityCondition
    bbox: tuple[float, float, float, float]
    grid_x: np.ndarray
    grid_y: np.ndarray
    u_grid: np.ndarray
    v_grid: np.ndarray
    unmasked: np.ndarray
    weak_threshold: float
    p99: float
    canvas_px: int = DEFAULT_CANVAS_PX
    speed_scale: float = DEFAULT_SPEED_SCALE

    @property
    def grid_res(self) -> int:
        return int(self.u_grid.shape[0])

    def world_to_cell(self, x: float, y: float) -> tuple[int, int] | None:
        xmin, xmax, ymin, ymax = self.bbox
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        if x < xmin or x > xmax or y < ymin or y > ymax:
            return None
        j = max(0, min(self.grid_res - 1, int(math.floor((x - xmin) / (xmax - xmin) * self.grid_res))))
        i = max(0, min(self.grid_res - 1, int(math.floor((y - ymin) / (ymax - ymin) * self.grid_res))))
        return i, j

    def world_to_grid(self, x: float, y: float) -> tuple[float, float]:
        xmin, xmax, ymin, ymax = self.bbox
        gx = (x - xmin) / (xmax - xmin) * (self.grid_res - 1)
        gy = (y - ymin) / (ymax - ymin) * (self.grid_res - 1)
        return float(gx), float(gy)

    def is_masked_cell(self, i: int, j: int) -> bool:
        if i < 0 or i >= self.grid_res or j < 0 or j >= self.grid_res:
            return True
        return not bool(self.unmasked[i, j])

    def sample_vector(self, x: float, y: float) -> tuple[float, float]:
        gx, gy = self.world_to_grid(float(x), float(y))
        return (
            float(bilinear_sample(self.u_grid, gx, gy)),
            float(bilinear_sample(self.v_grid, gx, gy)),
        )

    def sample_active_field(self, x: float, y: float) -> dict[str, float | bool | str | None]:
        cell = self.world_to_cell(float(x), float(y))
        if cell is None or self.is_masked_cell(*cell):
            return {
                "valid": False,
                "reason": "masked-or-out-of-bounds",
                "u": 0.0,
                "v": 0.0,
                "mag": 0.0,
                "screen_dir_x": 0.0,
                "screen_dir_y": 0.0,
            }

        u, v = self.sample_vector(float(x), float(y))
        mag = float(math.hypot(u, v))
        if not (math.isfinite(mag) and mag > FIELD_EPS):
            return {
                "valid": False,
                "reason": "zero-or-nonfinite-magnitude",
                "u": u,
                "v": v,
                "mag": mag,
                "screen_dir_x": 0.0,
                "screen_dir_y": 0.0,
            }
        if mag < float(self.weak_threshold):
            return {
                "valid": False,
                "reason": "below-weak-threshold",
                "u": u,
                "v": v,
                "mag": mag,
                "screen_dir_x": 0.0,
                "screen_dir_y": 0.0,
            }

        dir_x = u / (mag + FIELD_EPS)
        dir_y = v / (mag + FIELD_EPS)
        xmin, xmax, ymin, ymax = self.bbox
        sx_scale = float(self.canvas_px) / (xmax - xmin)
        sy_scale = float(self.canvas_px) / (ymax - ymin)
        screen_dx = dir_x * sx_scale
        screen_dy = -dir_y * sy_scale
        screen_mag = float(math.hypot(screen_dx, screen_dy))
        if not (math.isfinite(screen_mag) and screen_mag > FIELD_EPS):
            return {
                "valid": False,
                "reason": "zero-or-nonfinite-screen-direction",
                "u": u,
                "v": v,
                "mag": mag,
                "screen_dir_x": 0.0,
                "screen_dir_y": 0.0,
            }

        return {
            "valid": True,
            "reason": None,
            "u": u,
            "v": v,
            "mag": mag,
            "screen_dir_x": screen_dx / screen_mag,
            "screen_dir_y": screen_dy / screen_mag,
        }

    def world_delta_from_screen_direction(self, screen_dir_x: float, screen_dir_y: float, distance_px: float) -> tuple[float, float]:
        xmin, xmax, ymin, ymax = self.bbox
        sx_scale = float(self.canvas_px) / (xmax - xmin)
        sy_scale = float(self.canvas_px) / (ymax - ymin)
        dx = (float(screen_dir_x) * float(distance_px)) / sx_scale
        dy = -(float(screen_dir_y) * float(distance_px)) / sy_scale
        return float(dx), float(dy)

    def advect_along_field_rk2(self, x: float, y: float) -> dict[str, float | bool | str | None]:
        start = self.sample_active_field(float(x), float(y))
        if not bool(start["valid"]):
            return {"valid": False, "reason": f"start-{start['reason']}"}

        distance_px = max(1.0, DISPLAY_SPEED_PX_BASE * float(self.speed_scale)) * FIXED_ADVECTION_STEP_SEC
        half_dx, half_dy = self.world_delta_from_screen_direction(
            float(start["screen_dir_x"]),
            float(start["screen_dir_y"]),
            distance_px * 0.5,
        )
        mid_x = float(x) + half_dx
        mid_y = float(y) + half_dy
        midpoint = self.sample_active_field(mid_x, mid_y)
        if not bool(midpoint["valid"]):
            return {"valid": False, "reason": f"midpoint-{midpoint['reason']}"}

        full_dx, full_dy = self.world_delta_from_screen_direction(
            float(midpoint["screen_dir_x"]),
            float(midpoint["screen_dir_y"]),
            distance_px,
        )
        nx = float(x) + full_dx
        ny = float(y) + full_dy
        if not (math.isfinite(nx) and math.isfinite(ny)):
            return {"valid": False, "reason": "nonfinite-endpoint"}

        end = self.sample_active_field(nx, ny)
        if not bool(end["valid"]):
            return {"valid": False, "reason": f"end-{end['reason']}"}

        return {"valid": True, "reason": None, "x": nx, "y": ny}


def build_oat_conditions(
    *,
    reference_grid_res: int = 25,
    reference_interpolation: str = "linear",
    reference_mask_radius: int = 1,
) -> list[SensitivityCondition]:
    return [
        SensitivityCondition(
            condition_id="reference",
            label="reference",
            grid_res=int(reference_grid_res),
            interpolation_method=str(reference_interpolation),
            mask_radius=int(reference_mask_radius),
            is_reference=True,
        ),
        SensitivityCondition(
            condition_id="grid_15",
            label="grid 15",
            grid_res=15,
            interpolation_method=str(reference_interpolation),
            mask_radius=int(reference_mask_radius),
        ),
        SensitivityCondition(
            condition_id="grid_20",
            label="grid 20",
            grid_res=20,
            interpolation_method=str(reference_interpolation),
            mask_radius=int(reference_mask_radius),
        ),
        SensitivityCondition(
            condition_id="grid_30",
            label="grid 30",
            grid_res=30,
            interpolation_method=str(reference_interpolation),
            mask_radius=int(reference_mask_radius),
        ),
        SensitivityCondition(
            condition_id="nearest",
            label="nearest",
            grid_res=int(reference_grid_res),
            interpolation_method="nearest",
            mask_radius=int(reference_mask_radius),
        ),
        SensitivityCondition(
            condition_id="radius_0",
            label="radius 0",
            grid_res=int(reference_grid_res),
            interpolation_method=str(reference_interpolation),
            mask_radius=0,
        ),
        SensitivityCondition(
            condition_id="radius_2",
            label="radius 2",
            grid_res=int(reference_grid_res),
            interpolation_method=str(reference_interpolation),
            mask_radius=2,
        ),
        SensitivityCondition(
            condition_id="radius_3",
            label="radius 3",
            grid_res=int(reference_grid_res),
            interpolation_method=str(reference_interpolation),
            mask_radius=3,
        ),
    ]


def _load_featurewind_case(tmap_path: str | Path) -> SeedsDisplayCase:
    from featurewind import config as fw_config
    from featurewind.preprocessing.csv_label_utils import humanize_point_labels
    from featurewind.preprocessing.data_processing import preprocess_tangent_map
    from featurewind.preprocessing.dataset_layout_utils import orient_dataset_for_display

    path = Path(tmap_path).resolve()
    valid_points, grad_vectors, positions, feature_names = preprocess_tangent_map(str(path))
    point_labels = None
    try:
        raw_labels = [getattr(point, "tmap_label", None) for point in valid_points]
        point_labels = humanize_point_labels(feature_names, raw_labels)
    except Exception:
        point_labels = None

    positions_arr = np.asarray(positions, dtype=float)
    grad_vectors_arr = np.asarray(grad_vectors, dtype=float)
    positions_arr, grad_vectors_arr, _layout_transform = orient_dataset_for_display(
        feature_names,
        positions_arr,
        grad_vectors_arr,
    )

    fw_config.initialize_global_state()
    fw_config.set_bounding_box(positions_arr)
    bbox = tuple(float(value) for value in fw_config.bounding_box)

    return SeedsDisplayCase(
        tmap_path=path,
        positions=positions_arr,
        grad_vectors=grad_vectors_arr,
        feature_names=[str(name) for name in feature_names],
        point_labels=None if point_labels is None else [str(label) for label in point_labels],
        bbox=bbox,
    )


def resolve_feature_index(feature_names: list[str], feature_name: str) -> int:
    target = str(feature_name).strip().lower()
    exact_matches = [idx for idx, name in enumerate(feature_names) if str(name).strip().lower() == target]
    if len(exact_matches) == 1:
        return int(exact_matches[0])
    if len(exact_matches) > 1:
        raise ValueError(f"Feature name {feature_name!r} matched multiple columns.")

    substring_matches = [idx for idx, name in enumerate(feature_names) if target in str(name).strip().lower()]
    if len(substring_matches) == 1:
        return int(substring_matches[0])
    if len(substring_matches) > 1:
        raise ValueError(f"Feature name {feature_name!r} is ambiguous across columns.")
    raise ValueError(f"Feature name {feature_name!r} was not found.")


def _robust_p99(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return 1.0
    percentile = float(np.percentile(finite, 99))
    return percentile if percentile > 0 else 1.0


def build_condition_context(
    case: SeedsDisplayCase,
    *,
    feature_idx: int,
    condition: SensitivityCondition,
    canvas_px: int = DEFAULT_CANVAS_PX,
    speed_scale: float = DEFAULT_SPEED_SCALE,
) -> GridConditionContext:
    from featurewind import config as fw_config
    from featurewind.physics.grid_computation import build_grids

    orig_bbox = getattr(fw_config, "bounding_box", None)
    orig_radius = getattr(fw_config, "MASK_DILATE_RADIUS_CELLS", 1)
    orig_include_hull = getattr(fw_config, "MASK_INCLUDE_INTERPOLATION_HULL", True)

    try:
        fw_config.initialize_global_state()
        fw_config.set_bounding_box(case.positions)
        fw_config.MASK_DILATE_RADIUS_CELLS = int(condition.mask_radius)
        fw_config.MASK_INCLUDE_INTERPOLATION_HULL = True

        with tempfile.TemporaryDirectory(prefix="featurewind_seeds_visual_sensitivity_") as tmpdir:
            grid_result = build_grids(
                case.positions,
                int(condition.grid_res),
                [int(feature_idx)],
                case.grad_vectors,
                case.feature_names,
                output_dir=tmpdir,
                interpolation_method=str(condition.interpolation_method),
            )

        grid_x = np.asarray(grid_result[3], dtype=float)
        grid_y = np.asarray(grid_result[4], dtype=float)
        u_grid = np.asarray(grid_result[5][0], dtype=float)
        v_grid = np.asarray(grid_result[6][0], dtype=float)
        final_mask = np.asarray(grid_result[12], dtype=bool)
        unmasked = np.logical_not(final_mask)
        mag = np.hypot(u_grid, v_grid)
        p99 = _robust_p99(mag)
        weak_threshold = max(1e-6, 0.015 * p99)

        return GridConditionContext(
            condition=condition,
            bbox=tuple(float(value) for value in fw_config.bounding_box),
            grid_x=grid_x,
            grid_y=grid_y,
            u_grid=u_grid,
            v_grid=v_grid,
            unmasked=unmasked,
            weak_threshold=float(weak_threshold),
            p99=float(p99),
            canvas_px=int(canvas_px),
            speed_scale=float(speed_scale),
        )
    finally:
        fw_config.bounding_box = orig_bbox
        fw_config.MASK_DILATE_RADIUS_CELLS = orig_radius
        fw_config.MASK_INCLUDE_INTERPOLATION_HULL = orig_include_hull


def bilinear_sample(grid: np.ndarray, gx: float, gy: float) -> float:
    arr = np.asarray(grid, dtype=float)
    height, width = arr.shape
    x = float(np.clip(float(gx), 0.0, max(width - 1, 0)))
    y = float(np.clip(float(gy), 0.0, max(height - 1, 0)))

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(width - 1, x0 + 1)
    y1 = min(height - 1, y0 + 1)
    tx = x - x0
    ty = y - y0

    top = (1.0 - tx) * arr[y0, x0] + tx * arr[y0, x1]
    bottom = (1.0 - tx) * arr[y1, x0] + tx * arr[y1, x1]
    return float((1.0 - ty) * top + ty * bottom)


def vectorized_bilinear_sample(grid: np.ndarray, gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=float)
    height, width = arr.shape
    x = np.clip(np.asarray(gx, dtype=float), 0.0, max(width - 1, 0))
    y = np.clip(np.asarray(gy, dtype=float), 0.0, max(height - 1, 0))

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = np.minimum(width - 1, x0 + 1)
    y1 = np.minimum(height - 1, y0 + 1)
    tx = x - x0
    ty = y - y0

    top = (1.0 - tx) * arr[y0, x0] + tx * arr[y0, x1]
    bottom = (1.0 - tx) * arr[y1, x0] + tx * arr[y1, x1]
    return (1.0 - ty) * top + ty * bottom


def build_common_lattice(bbox: tuple[float, float, float, float], resolution: int = COMMON_LATTICE_RES) -> CommonLattice:
    xmin, xmax, ymin, ymax = map(float, bbox)
    xs = np.linspace(xmin + (xmax - xmin) / (2 * resolution), xmax - (xmax - xmin) / (2 * resolution), resolution)
    ys = np.linspace(ymin + (ymax - ymin) / (2 * resolution), ymax - (ymax - ymin) / (2 * resolution), resolution)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points = np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])
    return CommonLattice(
        bbox=(xmin, xmax, ymin, ymax),
        resolution=int(resolution),
        grid_x=grid_x,
        grid_y=grid_y,
        points=points,
    )


def sample_context_on_lattice(context: GridConditionContext, lattice: CommonLattice) -> CommonLatticeSamples:
    points = np.asarray(lattice.points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    xmin, xmax, ymin, ymax = context.bbox

    in_bbox = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= xmin)
        & (x <= xmax)
        & (y >= ymin)
        & (y <= ymax)
    )
    i = np.floor((y - ymin) / (ymax - ymin) * context.grid_res).astype(int)
    j = np.floor((x - xmin) / (xmax - xmin) * context.grid_res).astype(int)
    i = np.clip(i, 0, context.grid_res - 1)
    j = np.clip(j, 0, context.grid_res - 1)

    support = np.zeros(points.shape[0], dtype=bool)
    support[in_bbox] = context.unmasked[i[in_bbox], j[in_bbox]]

    gx = (x - xmin) / (xmax - xmin) * (context.grid_res - 1)
    gy = (y - ymin) / (ymax - ymin) * (context.grid_res - 1)
    u = vectorized_bilinear_sample(context.u_grid, gx, gy)
    v = vectorized_bilinear_sample(context.v_grid, gx, gy)
    vectors = np.column_stack([u, v])
    vectors[~support] = 0.0

    return CommonLatticeSamples(support=support, vectors=vectors)


def compute_support_iou(reference_support: np.ndarray, condition_support: np.ndarray) -> float:
    ref = np.asarray(reference_support, dtype=bool)
    cond = np.asarray(condition_support, dtype=bool)
    union = np.logical_or(ref, cond)
    if not np.any(union):
        return 1.0
    intersection = np.logical_and(ref, cond)
    return float(np.sum(intersection) / np.sum(union))


def compute_field_cosine_mean(
    reference_vectors: np.ndarray,
    condition_vectors: np.ndarray,
    reference_support: np.ndarray,
    condition_support: np.ndarray,
) -> float:
    ref = np.asarray(reference_vectors, dtype=float)
    cond = np.asarray(condition_vectors, dtype=float)
    mask = np.asarray(reference_support, dtype=bool) & np.asarray(condition_support, dtype=bool)
    if not np.any(mask):
        return float("nan")

    ref_masked = ref[mask]
    cond_masked = cond[mask]
    ref_norm = np.linalg.norm(ref_masked, axis=1)
    cond_norm = np.linalg.norm(cond_masked, axis=1)
    valid = (ref_norm > 1e-12) & (cond_norm > 1e-12)
    if not np.any(valid):
        return float("nan")
    cosine = np.sum(ref_masked[valid] * cond_masked[valid], axis=1) / (ref_norm[valid] * cond_norm[valid])
    return float(np.mean(np.clip(cosine, -1.0, 1.0)))


def resample_polyline(points: np.ndarray, count: int = TRAIL_SAMPLE_COUNT) -> np.ndarray:
    path = np.asarray(points, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if path.shape[0] == 0:
        raise ValueError("points must contain at least one row")
    if path.shape[0] == 1 or int(count) <= 1:
        return path[[0]].copy()

    deltas = np.diff(path, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = float(cumulative[-1])
    if not math.isfinite(total_length) or total_length <= 1e-12:
        indices = np.linspace(0, path.shape[0] - 1, int(count))
        return np.vstack([path[int(round(value))] for value in indices])

    target_distances = np.linspace(0.0, total_length, int(count))
    sampled = np.zeros((int(count), 2), dtype=float)
    for idx, distance in enumerate(target_distances):
        seg_idx = min(int(np.searchsorted(cumulative, distance, side="right")) - 1, len(seg_lengths) - 1)
        seg_idx = max(0, seg_idx)
        seg_len = float(seg_lengths[seg_idx])
        if seg_len <= 1e-12:
            sampled[idx] = path[seg_idx]
            continue
        start_len = cumulative[seg_idx]
        t = (distance - start_len) / seg_len
        sampled[idx] = path[seg_idx] * (1.0 - t) + path[seg_idx + 1] * t
    return sampled


def compute_trail_distance_metrics(
    reference_points: np.ndarray,
    condition_points: np.ndarray,
    *,
    bbox: tuple[float, float, float, float],
    sample_count: int = TRAIL_SAMPLE_COUNT,
) -> tuple[float, float]:
    reference_sampled = resample_polyline(reference_points, count=sample_count)
    condition_sampled = resample_polyline(condition_points, count=sample_count)
    distances = np.linalg.norm(reference_sampled - condition_sampled, axis=1)

    xmin, xmax, ymin, ymax = map(float, bbox)
    bbox_diag = max(math.hypot(xmax - xmin, ymax - ymin), 1e-12)
    mean_deviation = float(np.mean(distances) / bbox_diag)
    endpoint_shift = float(np.linalg.norm(reference_sampled[-1] - condition_sampled[-1]) / bbox_diag)
    return mean_deviation, endpoint_shift


def build_static_trail(
    context: GridConditionContext,
    *,
    start_x: float,
    start_y: float,
    point_index: int,
    max_steps: int | None = None,
) -> StaticTrailResult:
    grid_res = context.grid_res
    xmin, xmax, ymin, ymax = context.bbox

    dx_world = (xmax - xmin) / grid_res
    dy_world = (ymax - ymin) / grid_res
    step_px = max(1.0, DISPLAY_SPEED_PX_BASE * float(context.speed_scale)) * FIXED_ADVECTION_STEP_SEC
    cell_px_x = float(context.canvas_px) / grid_res
    cell_px_y = float(context.canvas_px) / grid_res
    visit_limit = max(64, int(math.ceil((math.hypot(cell_px_x, cell_px_y) / max(step_px, 1e-6)) * 2.0)))
    progress_threshold = max(1e-6, math.hypot(dx_world, dy_world) * 0.35)
    recent_window_size = 16

    if max_steps is None:
        max_steps = max(64, min(4000, grid_res * grid_res * 4))

    start_cell = context.world_to_cell(float(start_x), float(start_y))
    if start_cell is None or context.is_masked_cell(*start_cell):
        return StaticTrailResult(
            condition_id=context.condition.condition_id,
            point_index=int(point_index),
            points=np.empty((0, 2), dtype=float),
            stop_reason="start-masked-or-out-of-bounds",
            stop_step=0,
        )

    start_sample = context.sample_active_field(float(start_x), float(start_y))
    if not bool(start_sample["valid"]):
        return StaticTrailResult(
            condition_id=context.condition.condition_id,
            point_index=int(point_index),
            points=np.asarray([[float(start_x), float(start_y)]], dtype=float),
            stop_reason=f"start-{start_sample['reason']}",
            stop_step=0,
        )

    points = [[float(start_x), float(start_y)]]
    recent_positions = [np.asarray([float(start_x), float(start_y)], dtype=float)]
    same_cell_key: str | None = None
    same_cell_visit_count = 0
    x = float(start_x)
    y = float(start_y)

    for step in range(int(max_steps)):
        cell = context.world_to_cell(x, y)
        if cell is None or context.is_masked_cell(*cell):
            return StaticTrailResult(
                condition_id=context.condition.condition_id,
                point_index=int(point_index),
                points=np.asarray(points, dtype=float),
                stop_reason="current-cell-masked-or-out-of-bounds",
                stop_step=int(step),
            )

        key = f"{cell[0]}:{cell[1]}"
        if key == same_cell_key:
            same_cell_visit_count += 1
        else:
            same_cell_key = key
            same_cell_visit_count = 1

        advected = context.advect_along_field_rk2(x, y)
        if not bool(advected["valid"]):
            return StaticTrailResult(
                condition_id=context.condition.condition_id,
                point_index=int(point_index),
                points=np.asarray(points, dtype=float),
                stop_reason=str(advected["reason"]),
                stop_step=int(step),
            )

        nx = float(advected["x"])
        ny = float(advected["y"])
        next_cell = context.world_to_cell(nx, ny)
        if next_cell is None or context.is_masked_cell(*next_cell):
            return StaticTrailResult(
                condition_id=context.condition.condition_id,
                point_index=int(point_index),
                points=np.asarray(points, dtype=float),
                stop_reason="next-cell-masked-or-out-of-bounds",
                stop_step=int(step),
            )

        if next_cell == cell:
            start_idx = max(0, len(recent_positions) - (recent_window_size - 1))
            recent_start = recent_positions[start_idx]
            recent_net_displacement = float(math.hypot(nx - recent_start[0], ny - recent_start[1]))
            if same_cell_visit_count > visit_limit and recent_net_displacement <= progress_threshold:
                return StaticTrailResult(
                    condition_id=context.condition.condition_id,
                    point_index=int(point_index),
                    points=np.asarray(points, dtype=float),
                    stop_reason="loop-guard",
                    stop_step=int(step),
                )

        points.append([nx, ny])
        recent_positions.append(np.asarray([nx, ny], dtype=float))
        if len(recent_positions) > recent_window_size:
            recent_positions.pop(0)
        x = nx
        y = ny

    return StaticTrailResult(
        condition_id=context.condition.condition_id,
        point_index=int(point_index),
        points=np.asarray(points, dtype=float),
        stop_reason="max-steps",
        stop_step=int(max_steps),
    )


def choose_seed_point(
    case: SeedsDisplayCase,
    contexts: list[GridConditionContext],
    *,
    primary_point_index: int = 35,
    secondary_point_index: int = 197,
) -> tuple[int, dict[str, StaticTrailResult]]:
    candidate_indices = []
    for candidate in (int(primary_point_index), int(secondary_point_index)):
        if 0 <= candidate < case.positions.shape[0] and candidate not in candidate_indices:
            candidate_indices.append(candidate)
    for candidate in range(case.positions.shape[0]):
        if candidate not in candidate_indices:
            candidate_indices.append(candidate)

    for point_index in candidate_indices:
        start_x, start_y = map(float, case.positions[int(point_index)])
        trail_map: dict[str, StaticTrailResult] = {}
        all_valid = True
        for context in contexts:
            trail = build_static_trail(
                context,
                start_x=start_x,
                start_y=start_y,
                point_index=int(point_index),
            )
            trail_map[context.condition.condition_id] = trail
            if not trail.valid:
                all_valid = False
                break
        if all_valid:
            return int(point_index), trail_map

    raise ValueError("No Seeds point produced a valid kernel_length trail across all sensitivity conditions.")


def build_label_color_map(labels: list[str] | None) -> dict[str, str]:
    if not labels:
        return {}
    unique = sorted({str(label) for label in labels})
    return {
        label: LABEL_PALETTE[idx % len(LABEL_PALETTE)]
        for idx, label in enumerate(unique)
    }


def _value_range(values: list[float]) -> tuple[float, float]:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return float("nan"), float("nan")
    return min(finite), max(finite)


def _format_metric(value: float, *, digits: int) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    cleaned = str(color or "").strip().lstrip("#")
    if len(cleaned) != 6:
        return (160, 160, 160)
    return tuple(int(cleaned[idx:idx + 2], 16) for idx in (0, 2, 4))


def _rgba(color: str, alpha: float) -> tuple[float, float, float, float]:
    r, g, b = _hex_to_rgb(color)
    return (r / 255.0, g / 255.0, b / 255.0, float(alpha))


def _darken_color(color: str, factor: float = 0.58) -> tuple[float, float, float, float]:
    r, g, b = _hex_to_rgb(color)
    return (
        max(0.0, min(1.0, (r / 255.0) * float(factor))),
        max(0.0, min(1.0, (g / 255.0) * float(factor))),
        max(0.0, min(1.0, (b / 255.0) * float(factor))),
        0.9,
    )


def _load_ui_snapshot_image(path: Path) -> np.ndarray | None:
    try:
        image = mpimg.imread(path)
    except Exception:
        return None
    arr = np.asarray(image)
    if arr.ndim < 2:
        return None
    return arr


def load_ui_snapshots(
    snapshot_dir: str | Path | None,
    conditions: list[SensitivityCondition],
) -> dict[str, np.ndarray]:
    if snapshot_dir is None:
        return {}

    root = Path(snapshot_dir).expanduser().resolve()
    if not root.is_dir():
        return {}

    suffixes = (".png", ".jpg", ".jpeg", ".webp")
    snapshots: dict[str, np.ndarray] = {}
    for condition in conditions:
        found_path = None
        for suffix in suffixes:
            candidate = root / f"{condition.condition_id}{suffix}"
            if candidate.exists():
                found_path = candidate
                break
        if found_path is None:
            continue
        image = _load_ui_snapshot_image(found_path)
        if image is not None:
            snapshots[condition.condition_id] = image
    return snapshots


def _panel_title(metrics: ConditionMetrics, label: str) -> str:
    return (
        f"{label}\n"
        f"IoU {_format_metric(metrics.support_iou, digits=2)}  |  "
        f"cos {_format_metric(metrics.field_cosine_mean, digits=2)}  |  "
        f"dev {_format_metric(metrics.trail_mean_deviation, digits=3)}"
    )


def compute_condition_metrics(
    *,
    reference_context: GridConditionContext,
    reference_trail: StaticTrailResult,
    context: GridConditionContext,
    trail: StaticTrailResult,
    lattice: CommonLattice,
    sample_count: int = TRAIL_SAMPLE_COUNT,
) -> ConditionMetrics:
    reference_samples = sample_context_on_lattice(reference_context, lattice)
    condition_samples = sample_context_on_lattice(context, lattice)

    support_iou = compute_support_iou(reference_samples.support, condition_samples.support)
    field_cosine_mean = compute_field_cosine_mean(
        reference_samples.vectors,
        condition_samples.vectors,
        reference_samples.support,
        condition_samples.support,
    )
    trail_mean_deviation, trail_endpoint_shift = compute_trail_distance_metrics(
        reference_trail.points,
        trail.points,
        bbox=reference_context.bbox,
        sample_count=sample_count,
    )

    return ConditionMetrics(
        condition_id=context.condition.condition_id,
        condition_label=context.condition.label,
        grid_res=int(context.condition.grid_res),
        interpolation_method=str(context.condition.interpolation_method),
        mask_radius=int(context.condition.mask_radius),
        is_reference=bool(context.condition.is_reference),
        point_index=int(trail.point_index),
        support_iou=float(support_iou),
        field_cosine_mean=float(field_cosine_mean),
        trail_mean_deviation=float(trail_mean_deviation),
        trail_endpoint_shift=float(trail_endpoint_shift),
        trail_stop_reason=str(trail.stop_reason),
        trail_point_count=int(trail.point_count),
    )


def _draw_frontend_like_points(
    *,
    ax: Any,
    positions: np.ndarray,
    fill_color: str,
) -> None:
    rgb = _hex_to_rgb(fill_color)
    outer = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 0.15)
    inner = (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, 0.30)
    ax.scatter(positions[:, 0], positions[:, 1], s=132, color=outer, linewidths=0.0, zorder=1)
    ax.scatter(positions[:, 0], positions[:, 1], s=64, color=inner, linewidths=0.0, zorder=2)
    ax.scatter(positions[:, 0], positions[:, 1], s=28, color=(1.0, 1.0, 1.0, 0.92), linewidths=0.0, zorder=3)
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=14,
        color=fill_color,
        edgecolors="#111827",
        linewidths=0.55,
        zorder=4,
    )


def _add_arrowhead(
    ax: Any,
    start: np.ndarray,
    end: np.ndarray,
    *,
    color: Any,
    mutation_scale: float,
    linewidth: float,
    zorder: int,
) -> None:
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
        joinstyle="miter",
        capstyle="round",
    )
    ax.add_patch(patch)


def _draw_grid_overlay(ax: Any, bbox: tuple[float, float, float, float], grid_res: int) -> None:
    xmin, xmax, ymin, ymax = map(float, bbox)
    x_edges = np.linspace(xmin, xmax, int(grid_res) + 1)
    y_edges = np.linspace(ymin, ymax, int(grid_res) + 1)
    for x_edge in x_edges:
        ax.axvline(
            float(x_edge),
            color=GRIDLINE_COLOR,
            linewidth=0.75,
            alpha=1.0,
            zorder=0.85,
            solid_capstyle="butt",
        )
    for y_edge in y_edges:
        ax.axhline(
            float(y_edge),
            color=GRIDLINE_COLOR,
            linewidth=0.75,
            alpha=1.0,
            zorder=0.85,
            solid_capstyle="butt",
        )


def _draw_masked_cells(ax: Any, context: GridConditionContext) -> None:
    xmin, xmax, ymin, ymax = map(float, context.bbox)
    dx = (xmax - xmin) / float(context.grid_res)
    dy = (ymax - ymin) / float(context.grid_res)
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
                    zorder=0.35,
                )
            )


def _render_condition_panel(
    *,
    ax: Any,
    case: SeedsDisplayCase,
    context: GridConditionContext,
    metrics: ConditionMetrics,
    trail: np.ndarray,
    reference_trail: np.ndarray,
    label_colors: dict[str, str],
    ui_snapshot: np.ndarray | None,
) -> None:
    ax.set_facecolor(BACKGROUND_COLOR)
    if ui_snapshot is not None:
        ax.imshow(
            ui_snapshot,
            extent=(case.bbox[0], case.bbox[1], case.bbox[2], case.bbox[3]),
            origin="upper",
            interpolation="nearest",
            zorder=0,
        )
    elif label_colors and case.point_labels:
        for label, color in label_colors.items():
            label_mask = np.asarray([str(value) == label for value in case.point_labels], dtype=bool)
            if not np.any(label_mask):
                continue
            _draw_frontend_like_points(
                ax=ax,
                positions=case.positions[label_mask],
                fill_color=color,
            )
    else:
        _draw_frontend_like_points(
            ax=ax,
            positions=case.positions,
            fill_color="#d9d9d9",
        )
    _draw_masked_cells(ax, context)
    _draw_grid_overlay(ax, case.bbox, context.grid_res)
    ax.plot(
        trail[:, 0],
        trail[:, 1],
        color=(1.0, 1.0, 1.0, 0.96),
        linewidth=6.0,
        alpha=1.0,
        zorder=5,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    ax.plot(
        trail[:, 0],
        trail[:, 1],
        color=_darken_color(CONDITION_TRAIL_COLOR),
        linewidth=4.2,
        alpha=1.0,
        zorder=6,
        solid_capstyle="round",
        solid_joinstyle="round",
    )
    ax.plot(
        trail[:, 0],
        trail[:, 1],
        color=CONDITION_TRAIL_COLOR,
        linewidth=3.0,
        alpha=0.98,
        zorder=7,
        solid_capstyle="round",
        solid_joinstyle="round",
    )

    if trail.shape[0] >= 2:
        _add_arrowhead(
            ax,
            trail[-2],
            trail[-1],
            color=_darken_color(CONDITION_TRAIL_COLOR),
            mutation_scale=15.0,
            linewidth=3.2,
            zorder=8,
        )
        _add_arrowhead(
            ax,
            trail[-2],
            trail[-1],
            color=_rgba(CONDITION_TRAIL_COLOR, 0.98),
            mutation_scale=13.0,
            linewidth=2.1,
            zorder=9,
        )

    ax.scatter(
        [trail[0, 0]],
        [trail[0, 1]],
        s=136,
        facecolors=(1.0, 1.0, 1.0, 0.98),
        edgecolors=CONDITION_TRAIL_COLOR,
        linewidths=1.7,
        zorder=10,
    )

    ax.plot(
        reference_trail[:, 0],
        reference_trail[:, 1],
        color=REFERENCE_TRAIL_COLOR,
        linewidth=2.15,
        linestyle=(0, (4.0, 3.0)),
        alpha=0.98,
        zorder=11,
        solid_capstyle="round",
        dash_capstyle="round",
    )
    if reference_trail.shape[0] >= 2:
        _add_arrowhead(
            ax,
            reference_trail[-2],
            reference_trail[-1],
            color=REFERENCE_TRAIL_COLOR,
            mutation_scale=13.0,
            linewidth=2.0,
            zorder=12,
        )

    ax.set_xlim(case.bbox[0], case.bbox[1])
    ax.set_ylim(case.bbox[2], case.bbox[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(
        _panel_title(metrics, context.condition.label),
        fontsize=10.5,
        color="#1f1b17",
        pad=8,
        fontfamily=TEXT_FONT_FAMILY,
    )


def render_visual_stability_figure(
    *,
    case: SeedsDisplayCase,
    contexts: list[GridConditionContext],
    metrics_by_id: dict[str, ConditionMetrics],
    trails_by_id: dict[str, StaticTrailResult],
    ui_snapshots_by_id: dict[str, np.ndarray] | None,
    output_png: Path,
    output_pdf: Path,
) -> None:
    if len(contexts) == 0:
        raise ValueError("At least one condition is required to render a combined figure.")
    if len(contexts) > DEFAULT_MAX_FIGURES:
        raise ValueError(f"Combined figure supports at most {DEFAULT_MAX_FIGURES} conditions.")

    label_colors = build_label_color_map(case.point_labels)
    reference_trail = trails_by_id["reference"].points

    n_panels = len(contexts)
    ncols = 4 if n_panels > 4 else min(4, max(1, n_panels))
    nrows = int(math.ceil(n_panels / ncols))
    fig_width = 5.0 * ncols
    fig_height = 4.2 * nrows + 0.8
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.035, right=0.99, wspace=0.12, hspace=0.18)

    axes_flat = np.atleast_1d(axes).ravel()

    for ax, context in zip(axes_flat, contexts):
        metrics = metrics_by_id[context.condition.condition_id]
        trail = trails_by_id[context.condition.condition_id].points
        _render_condition_panel(
            ax=ax,
            case=case,
            context=context,
            metrics=metrics,
            trail=trail,
            reference_trail=reference_trail,
            label_colors=label_colors,
            ui_snapshot=None if ui_snapshots_by_id is None else ui_snapshots_by_id.get(context.condition.condition_id),
        )

    for ax in axes_flat[len(contexts):]:
        ax.axis("off")

    legend_handles = [
        Line2D([0], [0], color=REFERENCE_TRAIL_COLOR, linewidth=2.0, linestyle=(0, (4.0, 3.0)), label="Reference trail"),
        Line2D([0], [0], color=CONDITION_TRAIL_COLOR, linewidth=2.4, label="Condition trail"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.96), fontsize=10)
    fig.suptitle(
        "Seeds Visual Stability Under Grid, Interpolation, and Mask Changes",
        fontsize=16,
        fontweight="bold",
        color="#1f1b17",
        y=0.985,
        fontfamily=TEXT_FONT_FAMILY,
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def render_individual_condition_figures(
    *,
    case: SeedsDisplayCase,
    contexts: list[GridConditionContext],
    metrics_by_id: dict[str, ConditionMetrics],
    trails_by_id: dict[str, StaticTrailResult],
    ui_snapshots_by_id: dict[str, np.ndarray] | None,
    output_dir: Path,
) -> tuple[list[Path], list[Path]]:
    label_colors = build_label_color_map(case.point_labels)
    reference_trail = trails_by_id["reference"].points
    output_dir.mkdir(parents=True, exist_ok=True)

    png_paths: list[Path] = []
    pdf_paths: list[Path] = []
    for context in contexts:
        metrics = metrics_by_id[context.condition.condition_id]
        trail = trails_by_id[context.condition.condition_id].points
        fig, ax = plt.subplots(1, 1, figsize=(5.4, 4.7), constrained_layout=False)
        fig.patch.set_facecolor("white")
        fig.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.98)
        _render_condition_panel(
            ax=ax,
            case=case,
            context=context,
            metrics=metrics,
            trail=trail,
            reference_trail=reference_trail,
            label_colors=label_colors,
            ui_snapshot=None if ui_snapshots_by_id is None else ui_snapshots_by_id.get(context.condition.condition_id),
        )
        stem = f"seeds_visual_stability_{context.condition.condition_id}"
        png_path = output_dir / f"{stem}.png"
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(png_path, dpi=220, bbox_inches="tight")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        png_paths.append(png_path)
        pdf_paths.append(pdf_path)

    return png_paths, pdf_paths


def write_metrics_csv(metrics: list[ConditionMetrics], output_path: Path) -> None:
    rows = [
        {
            "condition_id": metric.condition_id,
            "condition_label": metric.condition_label,
            "grid_res": metric.grid_res,
            "interpolation_method": metric.interpolation_method,
            "mask_radius": metric.mask_radius,
            "is_reference": int(metric.is_reference),
            "point_index": metric.point_index,
            "support_iou": f"{metric.support_iou:.6f}",
            "field_cosine_mean": "" if not np.isfinite(metric.field_cosine_mean) else f"{metric.field_cosine_mean:.6f}",
            "trail_mean_deviation": f"{metric.trail_mean_deviation:.6f}",
            "trail_endpoint_shift": f"{metric.trail_endpoint_shift:.6f}",
            "trail_stop_reason": metric.trail_stop_reason,
            "trail_point_count": metric.trail_point_count,
        }
        for metric in metrics
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(
    *,
    case: SeedsDisplayCase,
    feature_name: str,
    point_index: int,
    metrics: list[ConditionMetrics],
    output_path: Path,
) -> None:
    non_reference = [metric for metric in metrics if not metric.is_reference]
    moderate = [
        metric
        for metric in non_reference
        if metric.condition_id in {"grid_20", "grid_30", "radius_0", "radius_2"}
    ]
    extended = [
        metric
        for metric in non_reference
        if metric.condition_id in {"grid_15", "radius_3"}
    ]
    nearest = next((metric for metric in non_reference if metric.condition_id == "nearest"), None)
    largest_trail_shift = max(non_reference, key=lambda metric: metric.trail_mean_deviation)

    moderate_support = [metric.support_iou for metric in moderate]
    moderate_cosine = [metric.field_cosine_mean for metric in moderate]
    moderate_trail = [metric.trail_mean_deviation for metric in moderate]
    extended_support = [metric.support_iou for metric in extended]
    extended_cosine = [metric.field_cosine_mean for metric in extended]
    extended_trail = [metric.trail_mean_deviation for metric in extended]
    support_min, support_max = _value_range(moderate_support)
    cosine_min, cosine_max = _value_range(moderate_cosine)
    trail_min, trail_max = _value_range(moderate_trail)
    extended_support_min, extended_support_max = _value_range(extended_support)
    extended_cosine_min, extended_cosine_max = _value_range(extended_cosine)
    extended_trail_min, extended_trail_max = _value_range(extended_trail)

    if nearest is not None and largest_trail_shift.condition_id == "nearest":
        nearest_clause = (
            f"Nearest-neighbor interpolation produced the largest normalized trail deviation "
            f"({nearest.trail_mean_deviation:.3f}) and endpoint shift ({nearest.trail_endpoint_shift:.3f}), "
            "but the overall left-to-right curved story remained visually interpretable."
        )
    elif nearest is not None:
        nearest_clause = (
            f"Nearest-neighbor interpolation remained the harshest interpolation check, with normalized trail deviation "
            f"{nearest.trail_mean_deviation:.3f} and endpoint shift {nearest.trail_endpoint_shift:.3f}; "
            f"the largest trail change overall came from {largest_trail_shift.condition_label}."
        )
    else:
        nearest_clause = (
            f"The largest trail change in the exported set came from {largest_trail_shift.condition_label}, "
            "while the overall curved trajectory remained visually interpretable."
        )

    moderate_clause = (
        f"Across moderate grid-size and mask-radius perturbations, support IoU stayed between "
        f"{support_min:.3f} and {support_max:.3f}, field cosine between "
        f"{cosine_min:.3f} and {cosine_max:.3f}, and normalized trail deviation between "
        f"{trail_min:.3f} and {trail_max:.3f}. "
        if moderate
        else ""
    )
    extended_clause = (
        f"Under the two stronger checks (grid 15 and radius 3), support IoU ranged from "
        f"{extended_support_min:.3f} to {extended_support_max:.3f}, field cosine from "
        f"{extended_cosine_min:.3f} to {extended_cosine_max:.3f}, and normalized trail deviation from "
        f"{extended_trail_min:.3f} to {extended_trail_max:.3f}. "
        if extended
        else ""
    )

    lines = [
        "# Seeds Visual Stability Sensitivity",
        "",
        f"- Dataset: `Seeds`",
        f"- Tmap: `{case.tmap_path}`",
        f"- Feature: `{feature_name}`",
        f"- Chosen point index: `{int(point_index)}`",
        "- Reference condition: `grid_res = 25`, `interpolation = linear`, `mask_radius = 1`, `mask_include_interpolation_hull = True`",
        "",
        "| Condition | Grid | Interpolation | Radius | Support IoU | Field Cosine | Trail Dev | Endpoint Shift |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for metric in metrics:
        lines.append(
            "| "
            f"{metric.condition_label} | {metric.grid_res} | {metric.interpolation_method} | {metric.mask_radius} | "
            f"{metric.support_iou:.3f} | "
            f"{'NA' if not np.isfinite(metric.field_cosine_mean) else f'{metric.field_cosine_mean:.3f}'} | "
            f"{metric.trail_mean_deviation:.3f} | {metric.trail_endpoint_shift:.3f} |"
        )

    lines.extend(
        [
            "",
            (
                "Discussion-ready summary: "
                f"{moderate_clause}"
                f"{extended_clause}"
                + nearest_clause
            ),
            "",
            "This is a case-study robustness check only; it does not claim global stability across datasets or embeddings.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_seeds_visual_sensitivity(
    *,
    tmap_path: str | Path,
    output_dir: str | Path,
    feature_name: str = "kernel_length",
    point_index: int = 35,
    reference_grid_res: int = 25,
    reference_interpolation: str = "linear",
    reference_mask_radius: int = 1,
    common_lattice_res: int = COMMON_LATTICE_RES,
    trail_sample_count: int = TRAIL_SAMPLE_COUNT,
    canvas_px: int = DEFAULT_CANVAS_PX,
    speed_scale: float = DEFAULT_SPEED_SCALE,
    ui_snapshot_dir: str | Path | None = None,
    max_figures: int = DEFAULT_MAX_FIGURES,
    render_combined: bool = False,
) -> AnalysisResult:
    case = _load_featurewind_case(tmap_path)
    feature_idx = resolve_feature_index(case.feature_names, feature_name)
    conditions = build_oat_conditions(
        reference_grid_res=reference_grid_res,
        reference_interpolation=reference_interpolation,
        reference_mask_radius=reference_mask_radius,
    )

    contexts = [
        build_condition_context(
            case,
            feature_idx=feature_idx,
            condition=condition,
            canvas_px=canvas_px,
            speed_scale=speed_scale,
        )
        for condition in conditions
    ]

    chosen_point_index, trails_by_id = choose_seed_point(
        case,
        contexts,
        primary_point_index=int(point_index),
        secondary_point_index=197,
    )

    lattice = build_common_lattice(case.bbox, resolution=common_lattice_res)
    reference_context = next(context for context in contexts if context.condition.is_reference)
    reference_trail = trails_by_id["reference"]

    metrics = [
        compute_condition_metrics(
            reference_context=reference_context,
            reference_trail=reference_trail,
            context=context,
            trail=trails_by_id[context.condition.condition_id],
            lattice=lattice,
            sample_count=trail_sample_count,
        )
        for context in contexts
    ]

    output_root = Path(output_dir).resolve()
    max_figures = max(1, min(int(max_figures), DEFAULT_MAX_FIGURES))
    exported_conditions = conditions[:max_figures]
    exported_contexts = contexts[:max_figures]
    exported_ids = [context.condition.condition_id for context in exported_contexts]
    exported_metrics = [
        next(metric for metric in metrics if metric.condition_id == condition_id)
        for condition_id in exported_ids
    ]

    figure_png = output_root / "seeds_visual_stability.png" if render_combined else None
    figure_pdf = output_root / "seeds_visual_stability.pdf" if render_combined else None
    metrics_csv = output_root / "seeds_visual_stability_metrics.csv"
    summary_md = output_root / "seeds_visual_stability_summary.md"
    ui_snapshots_by_id = load_ui_snapshots(ui_snapshot_dir, conditions)

    if render_combined:
        assert figure_png is not None and figure_pdf is not None
        render_visual_stability_figure(
            case=case,
            contexts=exported_contexts,
            metrics_by_id={metric.condition_id: metric for metric in exported_metrics},
            trails_by_id=trails_by_id,
            ui_snapshots_by_id=ui_snapshots_by_id,
            output_png=figure_png,
            output_pdf=figure_pdf,
        )
    panel_pngs, panel_pdfs = render_individual_condition_figures(
        case=case,
        contexts=exported_contexts,
        metrics_by_id={metric.condition_id: metric for metric in exported_metrics},
        trails_by_id=trails_by_id,
        ui_snapshots_by_id=ui_snapshots_by_id,
        output_dir=output_root,
    )
    write_metrics_csv(exported_metrics, metrics_csv)
    write_summary_markdown(
        case=case,
        feature_name=str(case.feature_names[feature_idx]),
        point_index=int(chosen_point_index),
        metrics=exported_metrics,
        output_path=summary_md,
    )

    return AnalysisResult(
        output_dir=output_root,
        feature_name=str(case.feature_names[feature_idx]),
        point_index=int(chosen_point_index),
        conditions=exported_conditions,
        metrics=exported_metrics,
        figure_png=figure_png,
        figure_pdf=figure_pdf,
        panel_pngs=panel_pngs,
        panel_pdfs=panel_pdfs,
        metrics_csv=metrics_csv,
        summary_md=summary_md,
    )
