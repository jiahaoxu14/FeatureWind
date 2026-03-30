from __future__ import annotations

import math

import numpy as np


SEEDS_FEATURE_COLUMNS = (
    "area",
    "perimeter",
    "compactness",
    "kernel_length",
    "kernel_width",
    "asymmetry_coefficient",
    "groove_length",
)


def is_seeds_dataset(col_labels) -> bool:
    return tuple(col_labels or []) == SEEDS_FEATURE_COLUMNS


def _normalize_angle(angle_radians: float) -> float:
    wrapped = (float(angle_radians) + math.pi) % (2.0 * math.pi) - math.pi
    # Keep pi represented as -pi for stable "smallest absolute angle" comparisons.
    if wrapped == math.pi:
        return -math.pi
    return wrapped


def compute_horizontal_alignment_rotation(positions) -> float:
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        return 0.0

    centered = points - points.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = math.atan2(float(principal[1]), float(principal[0]))

    candidates = (
        _normalize_angle(-angle),
        _normalize_angle(math.pi - angle),
    )
    return min(candidates, key=abs)


def rotate_positions_and_gradients(positions, grad_vectors, angle_radians: float):
    theta = float(angle_radians)
    if abs(theta) < 1e-9:
        return np.asarray(positions, dtype=float), np.asarray(grad_vectors, dtype=float)

    rotation = np.array(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ],
        dtype=float,
    )

    rotated_positions = np.asarray(positions, dtype=float) @ rotation.T
    rotated_grads = np.asarray(grad_vectors, dtype=float) @ rotation.T
    return rotated_positions, rotated_grads


def orient_dataset_for_display(col_labels, positions, grad_vectors):
    points = np.asarray(positions, dtype=float)
    grads = np.asarray(grad_vectors, dtype=float)
    if not is_seeds_dataset(col_labels):
        return points, grads, None

    rotation_radians = compute_horizontal_alignment_rotation(points)
    rotated_points, rotated_grads = rotate_positions_and_gradients(points, grads, rotation_radians)
    return rotated_points, rotated_grads, {
        "dataset": "seeds",
        "target_axis": "horizontal",
        "rotation_radians": rotation_radians,
        "rotation_degrees": math.degrees(rotation_radians),
    }
