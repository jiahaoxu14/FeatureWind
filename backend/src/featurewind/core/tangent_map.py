"""
Calculate the tangent map of a set of input data points using the DimReader method.

Input: numeric and tabular dataset
Output: the results in a JSON format.
"""

import json
import sys

import numpy as np

from . import dim_reader as DimReader

projections = DimReader.projections
projectionParamOpts = DimReader.projectionParamOpts
projectionClasses = DimReader.projectionClasses


def check_and_normalize_features(points):
    """
    Check if features are normalized to [0, 1] range, and normalize if needed.

    Args:
        points (list): List of data points, each point is a list of feature values

    Returns:
        list: Normalized points
    """
    if not points:
        return points

    points_array = np.asarray(points, dtype=float)
    n_points, n_features = points_array.shape

    print(f"Checking normalization for {n_features} features across {n_points} points...")

    feature_mins = points_array.min(axis=0)
    feature_maxs = points_array.max(axis=0)

    print("Current feature ranges:")
    needs_normalization = False
    for i in range(n_features):
        print(f"  Feature {i}: [{feature_mins[i]:.3f}, {feature_maxs[i]:.3f}]")
        if feature_mins[i] < -0.001 or feature_maxs[i] > 1.001:
            needs_normalization = True

    if not needs_normalization:
        print("Features are already normalized to [0, 1] range.")
        return points_array.tolist()

    print("Features are not normalized to [0, 1] range. Normalizing...")
    normalized_points = np.zeros_like(points_array, dtype=float)
    for i in range(n_features):
        feature_min = feature_mins[i]
        feature_max = feature_maxs[i]
        if feature_max > feature_min:
            normalized_points[:, i] = (points_array[:, i] - feature_min) / (feature_max - feature_min)
        else:
            normalized_points[:, i] = 0.0

    print("Features normalized to [0, 1] range:")
    norm_mins = normalized_points.min(axis=0)
    norm_maxs = normalized_points.max(axis=0)
    for i in range(n_features):
        print(f"  Feature {i}: [{norm_mins[i]:.3f}, {norm_maxs[i]:.3f}]")

    return normalized_points.tolist()


def _resolve_projection_class(projection):
    if projection in projectionClasses:
        return projection
    if isinstance(projection, str):
        projection_name = projection.lower()
        if projection_name not in map(str.lower, projections):
            raise ValueError("Invalid projection")
        proj_index = list(map(str.lower, projections)).index(projection_name)
        return projectionClasses[proj_index]
    raise ValueError("Invalid projection")


def assemble_tangent_entries(points, base_proj, grads):
    points_array = np.asarray(points, dtype=float)
    positions = np.asarray(base_proj, dtype=float)
    gradients = np.asarray(grads, dtype=float)

    if gradients.ndim != 3 or gradients.shape[1] != 2:
        raise ValueError("Expected gradients with shape (n_points, 2, n_features)")
    if points_array.shape[0] != positions.shape[0] or points_array.shape[0] != gradients.shape[0]:
        raise ValueError("Point, position, and gradient arrays must share the first dimension")

    tangent_map = []
    for point, position, tangent in zip(points_array, positions, gradients):
        tangent_map.append(
            {
                "domain": point.astype(float).tolist(),
                "range": position.astype(float).tolist(),
                "tangent": tangent.astype(float).tolist(),
            }
        )
    return tangent_map


def compute_tangent_map_data(points, projection, params=None, normalize=True):
    normalized_points = check_and_normalize_features(points) if normalize else np.asarray(points, dtype=float).tolist()
    projection_class = _resolve_projection_class(projection)
    runner = DimReader.ProjectionRunner(projection_class, params)
    runner.firstRun = True
    runner.calculateValues(normalized_points)

    try:
        base_proj = runner.outPoints.detach().cpu().numpy()
    except Exception:
        base_proj = np.asarray(runner.outPoints, dtype=float)

    return {
        "points": np.asarray(normalized_points, dtype=float),
        "positions": np.asarray(base_proj, dtype=float),
        "grads": np.asarray(runner.grads, dtype=float),
        "runner": runner,
    }


def calcTangentMap(points, projection, params):
    data = compute_tangent_map_data(points, projection, params=params)
    return assemble_tangent_entries(data["points"], data["positions"], data["grads"])


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        inputFile = sys.argv[1]
        projection = sys.argv[2]

        if str.lower(projection) not in map(str.lower, DimReader.projections) and str.lower(projection) != "tangent-map":
            print("Invalid Projection")
            print("Projection Options:")
            for opt in DimReader.projections:
                if opt != "Tangent-Map":
                    print("\t" + opt)
            exit(0)

        projInd = list(map(str.lower, DimReader.projections)).index(str.lower(projection))
        inputPts = DimReader.readFile(inputFile)

        if len(sys.argv) > 3:
            params = []
            for i in range(3, len(sys.argv)):
                params.append(sys.argv[i])
        else:
            params = []

        tMap = calcTangentMap(inputPts, projectionClasses[projInd], params)
        fName = inputFile[: inputFile.rfind(".")] + "_TangentMap_" + projections[projInd] + ".tmap"

        with open(fName, "w") as f:
            f.write(json.dumps(tMap))
    else:
        print("DimReaderScript [input file] [Projection] [optional parameters]")
        print("For all dimension perturbations, perturbation file = all")
        print("Projection Options:")
        for opt in projections:
            if opt != "Tangent-Map":
                print("\t" + opt)

        exit(0)
