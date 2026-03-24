from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Iterable, Sequence

import numpy as np


def standardize_features(points: np.ndarray) -> np.ndarray:
    """Column-wise standardization so step sizes are expressed in std units."""
    arr = np.asarray(points, dtype=float)
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return (arr - mean) / std


@dataclass(frozen=True)
class GlobalUMAPCaseSpec:
    name: str
    title: str
    description: str
    feature_names: tuple[str, ...]
    X: np.ndarray
    eval_feature_indices: tuple[int, ...]
    dr_backend: str = "umap"
    oracle_kind: str = "fixed_embedding_oracle_trail"
    local_vector_method: str = "centered_finite_difference_transform"


@dataclass
class FittedGlobalUMAPCase:
    name: str
    title: str
    description: str
    feature_names: tuple[str, ...]
    X: np.ndarray
    Y: np.ndarray
    grad_vectors: np.ndarray
    eval_feature_indices: tuple[int, ...]
    feature_std: np.ndarray
    dr_backend: str
    oracle_kind: str
    local_vector_method: str
    fd_epsilon_frac: float
    transform_model: Any

    def step_delta(self, feature_idx: int, delta_frac: float) -> float:
        return float(delta_frac) * float(self.feature_std[int(feature_idx)])

    def fd_epsilon(self, feature_idx: int) -> float:
        return float(self.fd_epsilon_frac) * float(self.feature_std[int(feature_idx)])

    def oracle_trail(self, point_idx: int, feature_idx: int, *, delta: float, steps: int) -> np.ndarray:
        point_idx = int(point_idx)
        feature_idx = int(feature_idx)
        steps = int(steps)
        delta = float(delta)
        base = np.asarray(self.X[point_idx], dtype=float)

        if steps < 1:
            raise ValueError("steps must be >= 1")

        perturbed = np.repeat(base[None, :], steps, axis=0)
        perturbed[:, feature_idx] += delta * np.arange(1, steps + 1, dtype=float)
        transformed = np.asarray(self.transform_model.transform(perturbed), dtype=float)

        path = np.empty((steps + 1, 2), dtype=float)
        # Step 0 uses the fitted embedding exactly, not transform(X[i]).
        path[0] = np.asarray(self.Y[point_idx], dtype=float)
        path[1:] = transformed
        return path


def _grid_points(nx: int, ny: int, *, x_span: tuple[float, float], y_span: tuple[float, float]) -> np.ndarray:
    xs = np.linspace(float(x_span[0]), float(x_span[1]), int(nx), dtype=float)
    ys = np.linspace(float(y_span[0]), float(y_span[1]), int(ny), dtype=float)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack([grid_x.reshape(-1), grid_y.reshape(-1)])


def _arc_manifold_points() -> np.ndarray:
    latent = _grid_points(55, 13, x_span=(-2.3, 2.3), y_span=(-0.32, 0.32))
    theta = latent[:, 0]
    radial = latent[:, 1]
    raw = np.column_stack(
        [
            theta,
            radial,
            np.cos(theta) * (1.0 + 0.30 * radial),
            np.sin(theta) * (1.0 + 0.30 * radial),
            0.45 * theta + 0.12 * radial,
            0.20 * np.sin(2.0 * theta) + 0.06 * radial,
        ]
    )
    return standardize_features(raw)


def _ribbon_points() -> np.ndarray:
    latent = _grid_points(40, 16, x_span=(-3.2, 3.2), y_span=(-0.40, 0.40))
    t = latent[:, 0]
    h = latent[:, 1]
    raw = np.column_stack(
        [
            t,
            h,
            0.32 * t + 0.10 * h,
            -0.18 * t + 0.18 * h,
            0.07 * t + 0.04 * h,
            0.03 * np.sin(0.5 * t) + 0.05 * h,
        ]
    )
    return standardize_features(raw)


def build_curved_global_umap_case(*, seed: int = 0) -> GlobalUMAPCaseSpec:
    _ = int(seed)
    X = _arc_manifold_points()
    return GlobalUMAPCaseSpec(
        name="curved_global_umap",
        title="Curved Global UMAP Transport",
        description=(
            "A deterministic arc-like synthetic dataset in standardized feature space. "
            "The fitted UMAP embedding yields a clearly curved observed oracle transform trail "
            "when feature_0 is perturbed across multiple finite steps."
        ),
        feature_names=("feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5"),
        X=X,
        eval_feature_indices=(0,),
    )


def build_near_linear_global_umap_case(*, seed: int = 0) -> GlobalUMAPCaseSpec:
    _ = int(seed)
    X = _ribbon_points()
    return GlobalUMAPCaseSpec(
        name="near_linear_global_umap",
        title="Near-Linear Global UMAP Transport",
        description=(
            "A deterministic ribbon-like synthetic dataset whose fitted UMAP embedding shows "
            "low observed curvature under feature_0 transport. This is a sanity-check case where "
            "the static trail and straight local baseline may remain similar."
        ),
        feature_names=("feature_0", "feature_1", "feature_2", "feature_3", "feature_4", "feature_5"),
        X=X,
        eval_feature_indices=(0,),
    )


def build_global_synthetic_cases(*, datasets: Iterable[str] | None = None, seed: int = 0) -> Sequence[GlobalUMAPCaseSpec]:
    requested = None if datasets is None else {str(name).strip() for name in datasets if str(name).strip()}
    all_cases = (
        build_curved_global_umap_case(seed=seed),
        build_near_linear_global_umap_case(seed=seed),
    )
    if not requested:
        return all_cases
    return tuple(case for case in all_cases if case.name in requested)


def _import_dr_backend(dr_backend: str) -> tuple[Any, str]:
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")
    backend = str(dr_backend)
    if backend == "umap":
        import umap  # type: ignore

        return umap.UMAP, "umap"
    if backend == "parametric_umap":
        try:
            from umap.parametric_umap import ParametricUMAP  # type: ignore
        except Exception as exc:  # pragma: no cover - optional path
            raise RuntimeError("parametric_umap requested but not available in this environment") from exc
        return ParametricUMAP, "parametric_umap"
    raise ValueError(f"Unsupported dr_backend: {backend}")


def fit_global_umap_case(
    spec: GlobalUMAPCaseSpec,
    *,
    seed: int,
    dr_backend: str = "umap",
    fd_epsilon_frac: float = 0.02,
) -> FittedGlobalUMAPCase:
    estimator_cls, backend_name = _import_dr_backend(dr_backend)
    common_kwargs = {
        "n_components": 2,
        "n_neighbors": 24,
        "min_dist": 0.06,
        "metric": "euclidean",
        "random_state": int(seed),
    }
    if backend_name == "umap":
        estimator = estimator_cls(
            **common_kwargs,
            transform_seed=int(seed),
            init="spectral",
            low_memory=True,
        )
    else:  # pragma: no cover - optional path
        estimator = estimator_cls(**common_kwargs)

    X = np.asarray(spec.X, dtype=float)
    Y = np.asarray(estimator.fit_transform(X), dtype=float)
    feature_std = np.std(X, axis=0, ddof=0)
    feature_std = np.where(feature_std > 1e-12, feature_std, 1.0)

    grad_vectors = np.zeros((X.shape[0], X.shape[1], 2), dtype=float)
    for feature_idx in spec.eval_feature_indices:
        epsilon = float(fd_epsilon_frac) * float(feature_std[int(feature_idx)])
        plus = X.copy()
        minus = X.copy()
        plus[:, int(feature_idx)] += epsilon
        minus[:, int(feature_idx)] -= epsilon
        y_plus = np.asarray(estimator.transform(plus), dtype=float)
        y_minus = np.asarray(estimator.transform(minus), dtype=float)
        grad_vectors[:, int(feature_idx), :] = (y_plus - y_minus) / max(2.0 * epsilon, 1e-12)

    return FittedGlobalUMAPCase(
        name=spec.name,
        title=spec.title,
        description=spec.description,
        feature_names=tuple(spec.feature_names),
        X=X,
        Y=Y,
        grad_vectors=grad_vectors,
        eval_feature_indices=tuple(int(idx) for idx in spec.eval_feature_indices),
        feature_std=np.asarray(feature_std, dtype=float),
        dr_backend=backend_name,
        oracle_kind=spec.oracle_kind,
        local_vector_method=spec.local_vector_method,
        fd_epsilon_frac=float(fd_epsilon_frac),
        transform_model=estimator,
    )
