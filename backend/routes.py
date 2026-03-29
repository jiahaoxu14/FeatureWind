import io
import json
import os
import sys
import uuid
import csv
from pathlib import Path
from typing import List, Tuple

from flask import Blueprint, request, jsonify

# Ensure core package is reachable when running inside backend/
BACKEND_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BACKEND_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from featurewind import config as fw_config
from featurewind.preprocessing import data_processing as fw_data
from featurewind.core import dim_reader as fw_dim
from featurewind.core.tangent_map import check_and_normalize_features
from featurewind.physics import grid_computation as fw_grid


api_bp = Blueprint("api", __name__)


# In-memory dataset registry for dev (replace with DB/cache in prod)
DATASETS = {}
UPLOAD_DIR = BACKEND_ROOT / "var" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _gen_dataset_id() -> str:
    return uuid.uuid4().hex


def _detect_csv_headers(path: str) -> Tuple[List[str], int]:
    """Return (headers, num_columns). If no header, synthesize names."""
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        first = next(reader)
        # Try to parse as floats; if fails, treat as header
        is_header = False
        for cell in first:
            try:
                float(cell)
            except Exception:
                is_header = True
                break
        if is_header:
            headers = first
            num_cols = len(first)
        else:
            num_cols = len(first)
            headers = [f"Feature {i}" for i in range(num_cols)]
    return headers, num_cols


def _read_csv_points_with_headers(path: str) -> Tuple[List[List[float]], List[str]]:
    points: List[List[float]] = []
    headers: List[str] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration:
            return points, headers
        # Detect header
        is_header = False
        row_vals: List[float] = []
        for cell in first:
            try:
                row_vals.append(float(cell))
            except Exception:
                is_header = True
                break
        if is_header:
            headers = first
        else:
            points.append(row_vals)
        # Rest rows
        for row in reader:
            vals: List[float] = []
            for cell in row:
                vals.append(float(cell))
            points.append(vals)
    if not headers and points:
        headers = [f"Feature {i}" for i in range(len(points[0]))]
    return points, headers


def _maybe_humanize_point_labels(col_labels, point_labels):
    if not isinstance(col_labels, list) or not isinstance(point_labels, list) or not point_labels:
        return point_labels

    wine_recognition_features = [
        "alcohol",
        "malic_acid",
        "ash",
        "alcalinity_of_ash",
        "magnesium",
        "total_phenols",
        "flavanoids",
        "nonflavanoid_phenols",
        "proanthocyanins",
        "color_intensity",
        "hue",
        "od280_od315",
        "proline",
    ]

    if col_labels == wine_recognition_features:
        label_map = {
            "1": "Barolo",
            "2": "Grignolino",
            "3": "Barbera",
        }
        return [label_map.get(str(label), str(label)) for label in point_labels]

    return point_labels


@api_bp.post("/upload")
def upload():
    """Upload a dataset (.tmap JSON or .csv). Returns dataset_id and basic metadata.

    - multipart/form-data with field name 'file'
    - or JSON {"tmap": {...}} for direct .tmap content
    """
    dataset_id = _gen_dataset_id()

    if request.content_type and request.content_type.startswith("application/json"):
        payload = request.get_json(silent=True) or {}
        if "tmap" in payload:
            # Persist to disk for consistency
            path = UPLOAD_DIR / f"{dataset_id}.tmap"
            with open(path, "w") as f:
                json.dump(payload["tmap"], f)
            # Extract labels
            try:
                col_labels = payload["tmap"].get("Col_labels", [])
            except Exception:
                col_labels = []
            DATASETS[dataset_id] = {"type": "tmap", "path": path, "labels": col_labels}
            return jsonify({"datasetId": dataset_id, "type": "tmap", "col_labels": col_labels})
        return jsonify({"error": "Unsupported JSON upload. Provide 'tmap'."}), 400

    # Multipart upload
    if "file" not in request.files:
        return jsonify({"error": "No file part 'file'"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".tmap", ".json", ".csv"):
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    server_name = f"{dataset_id}{ext}"
    save_path = UPLOAD_DIR / server_name
    file.save(save_path)

    if ext in (".tmap", ".json"):
        # Quick metadata read
        try:
            with open(save_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                col_labels = []
            elif isinstance(data, dict) and "tmap" in data:
                col_labels = data.get("Col_labels", [])
            else:
                try:
                    save_path.unlink(missing_ok=True)
                except Exception:
                    pass
                return jsonify({
                    "error": "Invalid tangent map file. Upload a .tmap/.json containing a 'tmap' array, "
                             "not a summary JSON such as '*_summary.json'."
                }), 400
        except Exception:
            try:
                save_path.unlink(missing_ok=True)
            except Exception:
                pass
            return jsonify({
                "error": "Invalid tangent map file. The uploaded .tmap/.json could not be parsed as tangent-map JSON."
            }), 400
        DATASETS[dataset_id] = {"type": "tmap", "path": save_path, "labels": col_labels}
        return jsonify({"datasetId": dataset_id, "type": "tmap", "col_labels": col_labels})
    else:
        headers, _ = _detect_csv_headers(save_path)
        DATASETS[dataset_id] = {"type": "csv", "path": save_path, "labels": headers}
        return jsonify({"datasetId": dataset_id, "type": "csv", "col_labels": headers})


@api_bp.get("/features")
def list_features():
    dataset_id = request.args.get("dataset_id")
    if not dataset_id or dataset_id not in DATASETS:
        return jsonify({"error": "Unknown dataset_id"}), 404
    entry = DATASETS[dataset_id]
    labels = entry.get("labels") or []
    # For CSV without headers, provide synthetic names
    if not labels and entry["type"] == "csv":
        labels, _ = _detect_csv_headers(entry["path"])
    return jsonify({"datasetId": dataset_id, "col_labels": labels})


@api_bp.post("/compute")
def compute():
    """Run the pipeline and return send-all grids and metadata.

    Body JSON:
    {
      dataset_id: str,
      algorithm: 'tsne'|'mds',
      topK?: number,
      featureIndex?: number,
      gridRes?: number,
      interpolationMethod?: string,
      includeRawGradients?: boolean
    }
    """
    body = request.get_json(silent=True) or {}
    dataset_id = body.get("dataset_id")
    algorithm = (body.get("algorithm") or "tsne").lower()
    top_k = body.get("topK")
    feature_index = body.get("featureIndex")
    grid_res = int(body.get("gridRes") or getattr(fw_config, "DEFAULT_GRID_RES", 25))
    interpolation_method_raw = body.get("interpolationMethod")
    include_raw = bool(body.get("includeRawGradients", False))
    cfg_overrides = body.get("config") or {}

    try:
        interpolation_method = fw_grid.normalize_interpolation_method(interpolation_method_raw)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not dataset_id or dataset_id not in DATASETS:
        return jsonify({"error": "Unknown dataset_id"}), 404

    entry = DATASETS[dataset_id]
    dtype = entry["type"]
    path = entry["path"]

    # Load and construct base arrays
    feature_values = None
    if dtype == "tmap":
        valid_points, all_grad_vectors, all_positions, col_labels = fw_data.preprocess_tangent_map(path)
        # Extract per-point labels from tmap entries for frontend marker shapes
        try:
            point_labels = [p.tmap_label for p in valid_points]
        except Exception:
            point_labels = None
        # Extract per-point raw feature values when available
        try:
            _vals = []
            for p in valid_points:
                row = getattr(p, 'domain', None)
                if row is None:
                    _vals = None; break
                _vals.append([float(x) for x in row])
            feature_values = _vals
        except Exception:
            feature_values = None
    elif dtype == "csv":
        # Read CSV with potential headers, normalize columns, then compute projection & grads
        points, col_labels = _read_csv_points_with_headers(path)
        points = check_and_normalize_features(points)

        # Run projection + gradient extraction
        if algorithm not in ("tsne", "mds"):
            return jsonify({"error": "Unsupported algorithm for CSV. Use 'tsne' or 'mds'."}), 400

        projection = fw_dim.tsne if algorithm == "tsne" else fw_dim.mds
        runner = fw_dim.ProjectionRunner(projection)
        runner.calculateValues(points)

        # Positions
        try:
            Y = runner.outPoints.detach().cpu().numpy()
        except Exception:
            # If already numpy
            Y = getattr(runner, "outPoints", None)
            if hasattr(Y, "numpy"):
                Y = Y.numpy()
        all_positions = Y

        # CSV path typically lacks per-point labels; default to a single label
        try:
            n_pts = int(getattr(all_positions, "shape", [0])[0])
            point_labels = [0] * n_pts
        except Exception:
            point_labels = None

        # Grads: (N, 2, M) -> (N, M, 2)
        import numpy as _np
        grads = _np.array(runner.grads)
        if grads.ndim != 3 or grads.shape[1] != 2:
            return jsonify({"error": "Unexpected gradient shape from ProjectionRunner"}), 500
        all_grad_vectors = _np.transpose(grads, (0, 2, 1))

        # Synthesize labels if missing
        if not col_labels:
            col_labels = [f"Feature {i}" for i in range(all_grad_vectors.shape[1])]
        # Provide normalized feature values for point coloring
        try:
            feature_values = points
        except Exception:
            feature_values = None
    else:
        return jsonify({"error": f"Unsupported dataset type: {dtype}"}), 400

    point_labels = _maybe_humanize_point_labels(col_labels, point_labels)

    # Optional: validation (skip strict check to support CSV path)
    # fw_data.validate_data(valid_points, all_grad_vectors, all_positions, col_labels)

    import numpy as _np
    all_positions = _np.asarray(all_positions, dtype=float)
    if all_positions.ndim != 2 or all_positions.shape[1] != 2 or all_positions.shape[0] == 0:
        return jsonify({
            "error": "Dataset did not produce a valid 2D position array. "
                     "This usually means the uploaded tangent map is empty or malformed."
        }), 400

    # Config state
    fw_config.initialize_global_state()
    fw_config.set_bounding_box(all_positions)

    # Selection (topK or single feature)
    n_features = len(col_labels)
    try:
        feature_ranking_arr, avg_magnitudes = fw_data.pick_top_k_features(all_grad_vectors, k=n_features)
        feature_ranking = [int(x) for x in _np.array(feature_ranking_arr).tolist()]
    except Exception:
        feature_ranking = list(range(n_features))
        avg_magnitudes = _np.zeros((n_features,), dtype=float)
    default_feature_index = int(feature_ranking[0]) if feature_ranking else 0
    if feature_index is not None:
        try:
            fi = int(feature_index)
        except Exception:
            return jsonify({"error": "featureIndex must be an integer"}), 400
        if not (0 <= fi < n_features):
            return jsonify({"error": f"featureIndex out of range 0..{n_features-1}"}), 400
        top_k_indices = [fi]
        selection_obj = {"featureIndex": fi}
    else:
        # Default: use all features when topK not provided or set to 'all'
        if top_k is None or (isinstance(top_k, str) and str(top_k).lower() == 'all'):
            top_k_indices = list(range(n_features))
            selection_obj = {"topKIndices": top_k_indices}
        else:
            try:
                top_k = int(top_k)
            except Exception:
                return jsonify({"error": "topK must be an integer or 'all'"}), 400
            top_k = max(1, min(top_k, n_features))
            top_k_indices = feature_ranking[:top_k]
            selection_obj = {"topKIndices": top_k_indices}

    # Apply per-request config overrides safely
    orig_MASK_DILATE_RADIUS_CELLS = getattr(fw_config, "MASK_DILATE_RADIUS_CELLS", 1)
    orig_MASK_INCLUDE_INTERPOLATION_HULL = getattr(fw_config, "MASK_INCLUDE_INTERPOLATION_HULL", True)
    try:
        if isinstance(cfg_overrides, dict):
            if "maskDilateRadiusCells" in cfg_overrides:
                try:
                    fw_config.MASK_DILATE_RADIUS_CELLS = max(0, int(round(float(cfg_overrides["maskDilateRadiusCells"]))))  # type: ignore[attr-defined]
                except Exception:
                    pass
            elif "maskBufferFactor" in cfg_overrides:
                try:
                    fw_config.MASK_DILATE_RADIUS_CELLS = max(0, int(round(float(cfg_overrides["maskBufferFactor"]))))  # type: ignore[attr-defined]
                except Exception:
                    pass
            if "maskIncludeInterpolationHull" in cfg_overrides:
                try:
                    fw_config.MASK_INCLUDE_INTERPOLATION_HULL = bool(cfg_overrides["maskIncludeInterpolationHull"])  # type: ignore[attr-defined]
                except Exception:
                    pass

        # Build grids (send-all includes per-feature grids regardless of selection)
        (
            interp_u_sum,
            interp_v_sum,
            interp_argmax,
            grid_x,
            grid_y,
            grid_u_feats,
            grid_v_feats,
            cell_dominant_features,
            grid_u_all_feats,
            grid_v_all_feats,
            cell_centers_x,
            cell_centers_y,
            final_mask,
        ) = fw_grid.build_grids(
            all_positions,
            grid_res,
            top_k_indices,
            all_grad_vectors,
            col_labels,
            output_dir=os.path.dirname(path),
            interpolation_method=interpolation_method,
        )
    finally:
        # Restore global to avoid cross-request bleed
        try:
            fw_config.MASK_DILATE_RADIUS_CELLS = orig_MASK_DILATE_RADIUS_CELLS  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            fw_config.MASK_INCLUDE_INTERPOLATION_HULL = orig_MASK_INCLUDE_INTERPOLATION_HULL  # type: ignore[attr-defined]
        except Exception:
            pass

    # Family-based per-feature colors for frontend and backend consistency.
    try:
        if bool(getattr(fw_config, "USE_PER_FEATURE_COLORS", False)):
            palette = list(getattr(fw_config, "GLASBEY_COLORS", [])) or ["#1f77b4"]
            colors = [palette[i % len(palette)] for i in range(n_features)]
            family_assignments = list(range(n_features))
        else:
            from featurewind.analysis import feature_clustering
            from featurewind.visualization import color_system

            n_families = max(1, min(int(getattr(fw_config, "MAX_FEATURE_FAMILIES", 4)), n_features))
            family_assignments, _, _ = feature_clustering.cluster_features_by_direction(
                grid_u_all_feats,
                grid_v_all_feats,
                n_families=n_families,
            )
            colors = color_system.assign_family_colors(family_assignments)
    except Exception:
        family_assignments = list(range(n_features))
        palette = list(getattr(fw_config, "GLASBEY_COLORS", [])) or ["#1f77b4"]
        colors = [palette[i % len(palette)] for i in range(n_features)]

    bbox = list(getattr(fw_config, "bounding_box", [0, 1, 0, 1]))

    # Convert to serializable lists
    def tolist(x):
        try:
            return x.tolist()
        except Exception:
            return x

    # Compute global max magnitude of the summed field across selected features (for vane dot alpha)
    try:
        import numpy as _np2
        _sum_u = _np2.sum(grid_u_feats, axis=0) if isinstance(grid_u_feats, _np2.ndarray) else None
        _sum_v = _np2.sum(grid_v_feats, axis=0) if isinstance(grid_v_feats, _np2.ndarray) else None
        if _sum_u is not None and _sum_v is not None:
            _sum_mag = _np2.sqrt(_sum_u**2 + _sum_v**2)
            global_sum_magnitude_max = float(_np2.nanmax(_sum_mag))
        else:
            global_sum_magnitude_max = 0.0
    except Exception:
        global_sum_magnitude_max = 0.0

    response = {
        "datasetId": dataset_id,
        "positions": tolist(all_positions),
        "col_labels": col_labels,
        "bbox": bbox,
        "grid_res": int(grid_res),
        "uAll": tolist(grid_u_all_feats),
        "vAll": tolist(grid_v_all_feats),
        "dominant": tolist(cell_dominant_features),
        "colors": colors,
        "familyAssignments": tolist(family_assignments),
        "featureRanking": feature_ranking,
        "defaultFeatureIndex": default_feature_index,
        "featureMagnitudes": tolist(avg_magnitudes),
        # Per-point labels (for marker shapes). Strings or numbers as provided.
        **({"point_labels": tolist(point_labels)} if point_labels is not None else {}),
        "global_sum_magnitude_max": global_sum_magnitude_max,
        "interpolationMethod": interpolation_method,
        "selection": selection_obj,
        "meta": {"dtypeHint": "float32", "order": "row-major"},
    }

    # Include per-point feature values when available for point coloring in frontend
    if feature_values is not None:
        response["feature_values"] = tolist(feature_values)

    # Provide unmasked grid so the frontend can respect masking
    try:
        if final_mask is not None:
            import numpy as _np
            unmasked = _np.logical_not(final_mask).astype(_np.uint8)
            response["unmasked"] = tolist(unmasked)
    except Exception:
        pass

    if include_raw:
        response["gradVectors"] = tolist(all_grad_vectors)

    return jsonify(response)


@api_bp.post("/colors")
def recolor():
    """Deprecated: feature-family recoloring has been removed from the web app flow."""
    return jsonify({"error": "Feature-family recoloring is deprecated in the web app API."}), 410
