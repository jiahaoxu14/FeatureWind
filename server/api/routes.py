import io
import json
import os
import uuid
import csv
from typing import List, Tuple

from flask import Blueprint, request, jsonify

from featurewind import config as fw_config
from featurewind.preprocessing import data_processing as fw_data
from featurewind.core import dim_reader as fw_dim
from featurewind.core.tangent_map import check_and_normalize_features
from featurewind.physics import grid_computation as fw_grid


api_bp = Blueprint("api", __name__)


# In-memory dataset registry for dev (replace with DB/cache in prod)
DATASETS = {}
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "..", "uploads")
UPLOAD_DIR = os.path.abspath(UPLOAD_DIR)
os.makedirs(UPLOAD_DIR, exist_ok=True)


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
            path = os.path.join(UPLOAD_DIR, f"{dataset_id}.tmap")
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
    save_path = os.path.join(UPLOAD_DIR, server_name)
    file.save(save_path)

    if ext in (".tmap", ".json"):
        # Quick metadata read
        try:
            with open(save_path, "r") as f:
                data = json.load(f)
            col_labels = data.get("Col_labels", [])
        except Exception:
            col_labels = []
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
      includeRawGradients?: boolean
    }
    """
    body = request.get_json(silent=True) or {}
    dataset_id = body.get("dataset_id")
    algorithm = (body.get("algorithm") or "tsne").lower()
    top_k = body.get("topK")
    feature_index = body.get("featureIndex")
    grid_res = int(body.get("gridRes") or getattr(fw_config, "DEFAULT_GRID_RES", 25))
    include_raw = bool(body.get("includeRawGradients", False))

    if not dataset_id or dataset_id not in DATASETS:
        return jsonify({"error": "Unknown dataset_id"}), 404

    entry = DATASETS[dataset_id]
    dtype = entry["type"]
    path = entry["path"]

    # Load and construct base arrays
    if dtype == "tmap":
        valid_points, all_grad_vectors, all_positions, col_labels = fw_data.preprocess_tangent_map(path)
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

        # Grads: (N, 2, M) -> (N, M, 2)
        import numpy as _np
        grads = _np.array(runner.grads)
        if grads.ndim != 3 or grads.shape[1] != 2:
            return jsonify({"error": "Unexpected gradient shape from ProjectionRunner"}), 500
        all_grad_vectors = _np.transpose(grads, (0, 2, 1))

        # Synthesize labels if missing
        if not col_labels:
            col_labels = [f"Feature {i}" for i in range(all_grad_vectors.shape[1])]
    else:
        return jsonify({"error": f"Unsupported dataset type: {dtype}"}), 400

    # Optional: validation (skip strict check to support CSV path)
    # fw_data.validate_data(valid_points, all_grad_vectors, all_positions, col_labels)

    # Config state
    fw_config.initialize_global_state()
    fw_config.set_bounding_box(all_positions)

    # Selection (topK or single feature)
    import numpy as _np
    n_features = len(col_labels)
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
        if top_k is None:
            top_k = min(5, n_features)
        try:
            top_k = int(top_k)
        except Exception:
            return jsonify({"error": "topK must be an integer"}), 400
        top_k = max(1, min(top_k, n_features))
        tk_indices, _ = fw_data.pick_top_k_features(all_grad_vectors, k=top_k)
        top_k_indices = _np.array(tk_indices).tolist()
        selection_obj = {"topKIndices": top_k_indices}

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
    ) = fw_grid.build_grids(all_positions, grid_res, top_k_indices, all_grad_vectors, col_labels, output_dir=os.path.dirname(path))

    # Colors: simple per-feature palette
    palette = list(getattr(fw_config, "GLASBEY_COLORS", []))
    if not palette:
        palette = ["#1f77b4"]
    colors = [palette[i % len(palette)] for i in range(n_features)]

    bbox = list(getattr(fw_config, "bounding_box", [0, 1, 0, 1]))

    # Convert to serializable lists
    def tolist(x):
        try:
            return x.tolist()
        except Exception:
            return x

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
        "selection": selection_obj,
        "meta": {"dtypeHint": "float32", "order": "row-major"},
    }

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
