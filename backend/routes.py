import csv
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from flask import Blueprint, jsonify, request

# Ensure core package is reachable when running inside backend/
BACKEND_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BACKEND_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from featurewind import config as fw_config
from featurewind.core import dim_reader as fw_dim
from featurewind.core.tangent_map import check_and_normalize_features
from featurewind.physics import grid_computation as fw_grid
from featurewind.preprocessing import data_processing as fw_data


api_bp = Blueprint("api", __name__)


# In-memory dataset registry for dev; manifests allow basic reload after restart.
DATASETS: Dict[str, Dict[str, Any]] = {}
UPLOAD_DIR = BACKEND_ROOT / "var" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _gen_dataset_id() -> str:
    return uuid.uuid4().hex


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json_load(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _safe_json_dump(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _detect_csv_headers(path: str) -> Tuple[List[str], int]:
    """Return (headers, num_columns). If no header, synthesize names."""
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        first = next(reader)
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
        for row in reader:
            vals: List[float] = []
            for cell in row:
                vals.append(float(cell))
            points.append(vals)
    if not headers and points:
        headers = [f"Feature {i}" for i in range(len(points[0]))]
    return points, headers


def _count_csv_rows(path: Path) -> int:
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return 0
        first = rows[0]
        try:
            for cell in first:
                float(cell)
            has_header = False
        except Exception:
            has_header = True
        return max(0, len(rows) - (1 if has_header else 0))
    except Exception:
        return 0


def _normalize_projection_meta(meta: Any) -> Dict[str, Any]:
    if not isinstance(meta, dict):
        return {}
    projection = meta.get("projection")
    if isinstance(projection, dict):
        out = dict(projection)
    else:
        out = {}
    method = out.get("method")
    if method is not None:
        out["method"] = str(method).lower()
    return out


def _validation_mode_from_projection(projection_meta: Dict[str, Any], has_domain_values: bool) -> str:
    method = str(projection_meta.get("method") or "").lower()
    if not has_domain_values:
        return "domain-values-missing"
    if method == "tsne":
        return "exact-tsne-rerun"
    if method:
        return "projection-not-yet-supported"
    return "projection-metadata-missing"


def _extract_tmap_metadata(tmap_payload: Dict[str, Any]) -> Dict[str, Any]:
    tmap_entries = tmap_payload.get("tmap") if isinstance(tmap_payload.get("tmap"), list) else []
    col_labels = tmap_payload.get("Col_labels") if isinstance(tmap_payload.get("Col_labels"), list) else []
    meta = tmap_payload.get("meta") if isinstance(tmap_payload.get("meta"), dict) else {}
    projection_meta = _normalize_projection_meta(meta)
    sample = tmap_entries[: min(len(tmap_entries), 8)]
    has_domain_values = bool(sample) and all(isinstance(entry, dict) and isinstance(entry.get("domain"), list) for entry in sample)
    validation_mode = _validation_mode_from_projection(projection_meta, has_domain_values)
    return {
        "col_labels": col_labels,
        "meta": meta,
        "projection_meta": projection_meta,
        "point_count": len(tmap_entries),
        "has_domain_values": has_domain_values,
        "supports_validation": validation_mode == "exact-tsne-rerun",
        "validation_mode": validation_mode,
    }


def _dataset_response(entry: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "datasetId": entry["datasetId"],
        "type": entry["type"],
        "datasetType": entry["type"],
        "col_labels": entry.get("labels") or [],
        "sourceName": entry.get("sourceName"),
        "fileHash": entry.get("fileHash"),
        "projectionMeta": entry.get("projectionMeta") or {},
        "supportsValidation": bool(entry.get("supportsValidation", False)),
        "validationMode": entry.get("validationMode") or "projection-metadata-missing",
        "pointCount": entry.get("pointCount"),
    }


def _manifest_path(dataset_id: str) -> Path:
    return UPLOAD_DIR / f"{dataset_id}.manifest.json"


def _write_manifest(entry: Dict[str, Any]) -> None:
    payload = {
        **_dataset_response(entry),
        "storedPath": str(entry["path"]),
        "manifestPath": str(_manifest_path(entry["datasetId"])),
        "meta": entry.get("meta") or {},
        "hasDomainValues": bool(entry.get("hasDomainValues", False)),
    }
    _safe_json_dump(_manifest_path(entry["datasetId"]), payload)


def _register_dataset(
    dataset_id: str,
    dtype: str,
    path: Path,
    labels: List[str],
    *,
    source_name: str,
    file_hash: str,
    meta: Optional[Dict[str, Any]] = None,
    projection_meta: Optional[Dict[str, Any]] = None,
    point_count: Optional[int] = None,
    supports_validation: bool = False,
    validation_mode: str = "projection-metadata-missing",
    has_domain_values: bool = False,
) -> Dict[str, Any]:
    entry = {
        "datasetId": dataset_id,
        "type": dtype,
        "path": path,
        "labels": labels,
        "sourceName": source_name,
        "fileHash": file_hash,
        "meta": meta or {},
        "projectionMeta": projection_meta or {},
        "pointCount": point_count,
        "supportsValidation": supports_validation,
        "validationMode": validation_mode,
        "hasDomainValues": has_domain_values,
    }
    DATASETS[dataset_id] = entry
    _write_manifest(entry)
    return entry


def _load_dataset_entry(dataset_id: str) -> Optional[Dict[str, Any]]:
    if dataset_id in DATASETS:
        return DATASETS[dataset_id]
    manifest = _manifest_path(dataset_id)
    if not manifest.exists():
        return None
    try:
        data = _safe_json_load(manifest)
        stored_path = Path(data["storedPath"])
        if not stored_path.exists():
            return None
        entry = {
            "datasetId": dataset_id,
            "type": data.get("datasetType") or data.get("type"),
            "path": stored_path,
            "labels": data.get("col_labels") or [],
            "sourceName": data.get("sourceName"),
            "fileHash": data.get("fileHash"),
            "meta": data.get("meta") or {},
            "projectionMeta": data.get("projectionMeta") or {},
            "pointCount": data.get("pointCount"),
            "supportsValidation": bool(data.get("supportsValidation", False)),
            "validationMode": data.get("validationMode") or "projection-metadata-missing",
            "hasDomainValues": bool(data.get("hasDomainValues", False)),
        }
    except Exception:
        return None
    DATASETS[dataset_id] = entry
    return entry


def _projection_params_from_meta(projection_meta: Dict[str, Any]) -> List[str]:
    params: List[str] = []
    method = str(projection_meta.get("method") or "").lower()
    if method == "tsne" and projection_meta.get("perplexity") is not None:
        try:
            params.append(f"perplexity={float(projection_meta['perplexity'])}")
        except Exception:
            pass
    return params


def _load_dataset_arrays(entry: Dict[str, Any], algorithm: str = "tsne") -> Dict[str, Any]:
    dtype = entry["type"]
    path = entry["path"]

    if dtype == "tmap":
        valid_points, all_grad_vectors, all_positions, col_labels = fw_data.preprocess_tangent_map(path)
        try:
            point_labels = [p.tmap_label for p in valid_points]
        except Exception:
            point_labels = None
        try:
            feature_values = np.array([[float(x) for x in getattr(p, "domain", [])] for p in valid_points], dtype=float)
        except Exception:
            feature_values = None
        return {
            "all_grad_vectors": all_grad_vectors,
            "all_positions": all_positions,
            "col_labels": col_labels,
            "feature_values": feature_values,
            "point_labels": point_labels,
        }

    points, col_labels = _read_csv_points_with_headers(path)
    points = check_and_normalize_features(points)

    if algorithm not in ("tsne", "mds"):
        raise ValueError("Unsupported algorithm for CSV. Use 'tsne' or 'mds'.")

    projection = fw_dim.tsne if algorithm == "tsne" else fw_dim.mds
    runner = fw_dim.ProjectionRunner(projection)
    runner.calculateValues(points)

    try:
        all_positions = runner.outPoints.detach().cpu().numpy()
    except Exception:
        Y = getattr(runner, "outPoints", None)
        all_positions = Y.numpy() if hasattr(Y, "numpy") else Y

    grads = np.array(runner.grads)
    if grads.ndim != 3 or grads.shape[1] != 2:
        raise ValueError("Unexpected gradient shape from ProjectionRunner")

    all_grad_vectors = np.transpose(grads, (0, 2, 1))
    if not col_labels:
        col_labels = [f"Feature {i}" for i in range(all_grad_vectors.shape[1])]

    point_labels = [0] * int(getattr(all_positions, "shape", [0])[0]) if all_positions is not None else None
    feature_values = np.asarray(points, dtype=float)
    return {
        "all_grad_vectors": all_grad_vectors,
        "all_positions": all_positions,
        "col_labels": col_labels,
        "feature_values": feature_values,
        "point_labels": point_labels,
    }


def _estimate_similarity_transform(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean
    denom = float(np.sum(src_centered ** 2))
    if denom <= 1e-12:
        return 1.0, np.eye(src.shape[1]), dst_mean - src_mean

    cov = src_centered.T @ dst_centered
    u, singular_values, vt = np.linalg.svd(cov)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    scale = float(np.sum(singular_values) / denom)
    translation = dst_mean - scale * (src_mean @ rotation)
    return scale, rotation, translation


def _apply_similarity_transform(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return scale * (points @ rotation) + translation


def _vector_metrics(predicted: np.ndarray, actual: np.ndarray) -> Dict[str, Any]:
    predicted = np.asarray(predicted, dtype=float)
    actual = np.asarray(actual, dtype=float)
    pred_norm = float(np.linalg.norm(predicted))
    actual_norm = float(np.linalg.norm(actual))
    error_norm = float(np.linalg.norm(actual - predicted))
    cosine_similarity = None
    if pred_norm > 1e-9 and actual_norm > 1e-9:
        cosine_similarity = float(np.dot(predicted, actual) / (pred_norm * actual_norm))
    magnitude_ratio = None if pred_norm <= 1e-9 else float(actual_norm / pred_norm)
    return {
        "predictedNorm": pred_norm,
        "actualNorm": actual_norm,
        "errorNorm": error_norm,
        "cosineSimilarity": cosine_similarity,
        "magnitudeRatio": magnitude_ratio,
    }


def _mean_defined(values: List[Any]) -> Optional[float]:
    nums = [float(v) for v in values if v is not None]
    return float(sum(nums) / len(nums)) if nums else None


@api_bp.post("/upload")
def upload():
    """Upload a dataset (.tmap JSON or .csv). Returns dataset_id and metadata."""
    dataset_id = _gen_dataset_id()

    if request.content_type and request.content_type.startswith("application/json"):
        payload = request.get_json(silent=True) or {}
        if "tmap" not in payload or not isinstance(payload["tmap"], dict):
            return jsonify({"error": "Unsupported JSON upload. Provide 'tmap'."}), 400

        tmap_payload = payload["tmap"]
        save_path = UPLOAD_DIR / f"{dataset_id}.tmap"
        _safe_json_dump(save_path, tmap_payload)
        metadata = _extract_tmap_metadata(tmap_payload)
        entry = _register_dataset(
            dataset_id,
            "tmap",
            save_path,
            metadata["col_labels"],
            source_name=payload.get("filename") or f"{dataset_id}.tmap",
            file_hash=_file_sha256(save_path),
            meta=metadata["meta"],
            projection_meta=metadata["projection_meta"],
            point_count=metadata["point_count"],
            supports_validation=metadata["supports_validation"],
            validation_mode=metadata["validation_mode"],
            has_domain_values=metadata["has_domain_values"],
        )
        return jsonify(_dataset_response(entry))

    if "file" not in request.files:
        return jsonify({"error": "No file part 'file'"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".tmap", ".json", ".csv"):
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    save_path = UPLOAD_DIR / f"{dataset_id}{ext}"
    file.save(save_path)
    file_hash = _file_sha256(save_path)

    if ext in (".tmap", ".json"):
        try:
            tmap_payload = _safe_json_load(save_path)
            metadata = _extract_tmap_metadata(tmap_payload)
        except Exception:
            metadata = {
                "col_labels": [],
                "meta": {},
                "projection_meta": {},
                "point_count": None,
                "supports_validation": False,
                "validation_mode": "projection-metadata-missing",
                "has_domain_values": False,
            }
        entry = _register_dataset(
            dataset_id,
            "tmap",
            save_path,
            metadata["col_labels"],
            source_name=file.filename,
            file_hash=file_hash,
            meta=metadata["meta"],
            projection_meta=metadata["projection_meta"],
            point_count=metadata["point_count"],
            supports_validation=metadata["supports_validation"],
            validation_mode=metadata["validation_mode"],
            has_domain_values=metadata["has_domain_values"],
        )
        return jsonify(_dataset_response(entry))

    headers, _ = _detect_csv_headers(str(save_path))
    entry = _register_dataset(
        dataset_id,
        "csv",
        save_path,
        headers,
        source_name=file.filename,
        file_hash=file_hash,
        meta={},
        projection_meta={},
        point_count=_count_csv_rows(save_path),
        supports_validation=False,
        validation_mode="projection-metadata-missing",
        has_domain_values=False,
    )
    return jsonify(_dataset_response(entry))


@api_bp.get("/features")
def list_features():
    dataset_id = request.args.get("dataset_id")
    entry = _load_dataset_entry(dataset_id) if dataset_id else None
    if not entry:
        return jsonify({"error": "Unknown dataset_id"}), 404
    labels = entry.get("labels") or []
    if not labels and entry["type"] == "csv":
        labels, _ = _detect_csv_headers(str(entry["path"]))
    return jsonify({"datasetId": dataset_id, "col_labels": labels, "datasetType": entry["type"]})


@api_bp.post("/compute")
def compute():
    """Run the compute pipeline and return grids, metadata, and optional raw gradients."""
    body = request.get_json(silent=True) or {}
    dataset_id = body.get("dataset_id")
    algorithm = (body.get("algorithm") or "tsne").lower()
    top_k = body.get("topK")
    feature_index = body.get("featureIndex")
    grid_res = int(body.get("gridRes") or getattr(fw_config, "DEFAULT_GRID_RES", 25))
    include_raw = bool(body.get("includeRawGradients", False))
    cfg_overrides = body.get("config") or {}
    manual_families = None
    try:
        mf = cfg_overrides.get("familyAssignments") if isinstance(cfg_overrides, dict) else None
        if isinstance(mf, list):
            manual_families = [int(x) for x in mf]
    except Exception:
        manual_families = None

    entry = _load_dataset_entry(dataset_id) if dataset_id else None
    if not entry:
        return jsonify({"error": "Unknown dataset_id"}), 404

    try:
        dataset_arrays = _load_dataset_arrays(entry, algorithm=algorithm)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    all_grad_vectors = dataset_arrays["all_grad_vectors"]
    all_positions = dataset_arrays["all_positions"]
    col_labels = dataset_arrays["col_labels"]
    feature_values = dataset_arrays["feature_values"]
    point_labels = dataset_arrays["point_labels"]

    fw_config.initialize_global_state()
    fw_config.set_bounding_box(all_positions)

    n_features = len(col_labels)
    if feature_index is not None:
        try:
            fi = int(feature_index)
        except Exception:
            return jsonify({"error": "featureIndex must be an integer"}), 400
        if not (0 <= fi < n_features):
            return jsonify({"error": f"featureIndex out of range 0..{n_features-1}"}), 400
        feature_index = fi
        top_k_indices = [fi]
        selection_obj = {"featureIndex": fi}
    else:
        if top_k is None or (isinstance(top_k, str) and str(top_k).lower() == "all"):
            top_k_indices = list(range(n_features))
            selection_obj = {"topKIndices": top_k_indices}
        else:
            try:
                top_k = int(top_k)
            except Exception:
                return jsonify({"error": "topK must be an integer or 'all'"}), 400
            top_k = max(1, min(top_k, n_features))
            tk_indices, _ = fw_data.pick_top_k_features(all_grad_vectors, k=top_k)
            top_k_indices = np.asarray(tk_indices).tolist()
            selection_obj = {"topKIndices": top_k_indices}

    orig_mask_buffer_factor = getattr(fw_config, "MASK_BUFFER_FACTOR", 0.2)
    try:
        if isinstance(cfg_overrides, dict) and "maskBufferFactor" in cfg_overrides:
            try:
                fw_config.MASK_BUFFER_FACTOR = float(cfg_overrides["maskBufferFactor"])  # type: ignore[attr-defined]
            except Exception:
                pass

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
            output_dir=os.path.dirname(str(entry["path"])),
        )
    finally:
        try:
            fw_config.MASK_BUFFER_FACTOR = orig_mask_buffer_factor  # type: ignore[attr-defined]
        except Exception:
            pass

    family_assignments = None
    try:
        use_per_feature = bool(getattr(fw_config, "USE_PER_FEATURE_COLORS", False))
        if use_per_feature:
            palette = list(getattr(fw_config, "GLASBEY_COLORS", [])) or ["#1f77b4"]
            colors = [palette[i % len(palette)] for i in range(n_features)]
            family_assignments = list(range(n_features))
        else:
            from featurewind.analysis import feature_clustering
            from featurewind.visualization import color_system

            n_families = min(n_features, int(getattr(fw_config, "MAX_FEATURE_FAMILIES", 4)))
            if manual_families and len(manual_families) == n_features:
                family_assignments = manual_families
            else:
                family_assignments, _, _ = feature_clustering.cluster_features_by_direction(
                    grid_u_all_feats,
                    grid_v_all_feats,
                    n_families=n_families,
                )
            colors = color_system.assign_family_colors(family_assignments)
            try:
                n_selected = len(top_k_indices) if feature_index is None else 1
                if (
                    bool(getattr(fw_config, "COLOR_BY_FEATURE_WHEN_FEW", True))
                    and n_selected <= int(getattr(fw_config, "FEATURE_COLOR_DISTINCT_THRESHOLD", 5))
                ):
                    palette = list(getattr(fw_config, "GLASBEY_COLORS", [])) or ["#1f77b4"]
                    sel_feats = top_k_indices if feature_index is None else [feature_index]
                    fam_to_color: Dict[int, str] = {}
                    for feat_idx in sel_feats:
                        fam_id = int(family_assignments[feat_idx]) if family_assignments is not None else feat_idx
                        if fam_id not in fam_to_color:
                            fam_to_color[fam_id] = palette[len(fam_to_color) % len(palette)]
                    for idx, fam_id in enumerate(family_assignments):
                        if fam_id in fam_to_color:
                            colors[idx] = fam_to_color[fam_id]
            except Exception:
                pass
        try:
            if feature_index is not None and 0 <= feature_index < len(colors):
                colors[feature_index] = getattr(fw_config, "SINGLE_FEATURE_COLOR", "#EE6677")
        except Exception:
            pass
    except Exception:
        palette = list(getattr(fw_config, "GLASBEY_COLORS", [])) or ["#1f77b4"]
        colors = [palette[i % len(palette)] for i in range(n_features)]

    feature_magnitudes = np.linalg.norm(all_grad_vectors, axis=2)
    avg_feature_magnitudes = feature_magnitudes.mean(axis=0)
    global_feature_ranking = np.argsort(-avg_feature_magnitudes).tolist()
    bbox = list(getattr(fw_config, "bounding_box", [0, 1, 0, 1]))

    def tolist(x: Any):
        try:
            return x.tolist()
        except Exception:
            return x

    try:
        sum_u = np.sum(grid_u_feats, axis=0) if isinstance(grid_u_feats, np.ndarray) else None
        sum_v = np.sum(grid_v_feats, axis=0) if isinstance(grid_v_feats, np.ndarray) else None
        if sum_u is not None and sum_v is not None:
            sum_mag = np.sqrt(sum_u ** 2 + sum_v ** 2)
            global_sum_magnitude_max = float(np.nanmax(sum_mag))
        else:
            global_sum_magnitude_max = 0.0
    except Exception:
        global_sum_magnitude_max = 0.0

    response: Dict[str, Any] = {
        **_dataset_response(entry),
        "positions": tolist(all_positions),
        "col_labels": col_labels,
        "bbox": bbox,
        "grid_res": int(grid_res),
        "uAll": tolist(grid_u_all_feats),
        "vAll": tolist(grid_v_all_feats),
        "dominant": tolist(cell_dominant_features),
        "colors": colors,
        **({"point_labels": tolist(point_labels)} if point_labels is not None else {}),
        **({"family_assignments": tolist(family_assignments)} if family_assignments is not None else {}),
        "global_sum_magnitude_max": global_sum_magnitude_max,
        "selection": selection_obj,
        "feature_stats": {
            "avgMagnitude": avg_feature_magnitudes.tolist(),
            "globalRanking": global_feature_ranking,
        },
        "meta": {
            "dtypeHint": "float32",
            "order": "row-major",
            "projectionMeta": entry.get("projectionMeta") or ({"method": algorithm} if entry["type"] == "csv" else {}),
        },
    }

    if feature_values is not None:
        response["feature_values"] = tolist(feature_values)

    try:
        if final_mask is not None:
            response["unmasked"] = tolist(np.logical_not(final_mask).astype(np.uint8))
    except Exception:
        pass

    if include_raw:
        response["gradVectors"] = tolist(all_grad_vectors)

    return jsonify(response)


@api_bp.post("/validate")
def validate():
    """Validate a single-feature local movement prediction with a rerun projection."""
    body = request.get_json(silent=True) or {}
    dataset_id = body.get("dataset_id")
    entry = _load_dataset_entry(dataset_id) if dataset_id else None
    if not entry:
        return jsonify({"error": "Unknown dataset_id"}), 404

    projection_meta = dict(entry.get("projectionMeta") or {})
    validation_mode = entry.get("validationMode") or "projection-metadata-missing"
    method = str(projection_meta.get("method") or "").lower()
    if method != "tsne":
        return jsonify({
            "error": "Validation currently supports only t-SNE sessions with projection metadata.",
            "validationMode": validation_mode,
        }), 400

    try:
        feature_index = int(body.get("featureIndex"))
    except Exception:
        return jsonify({"error": "featureIndex must be an integer"}), 400

    try:
        delta = float(body.get("delta"))
    except Exception:
        return jsonify({"error": "delta must be numeric"}), 400

    point_indices = body.get("pointIndices")
    if not isinstance(point_indices, list) or not point_indices:
        return jsonify({"error": "pointIndices must be a non-empty list"}), 400

    try:
        dataset_arrays = _load_dataset_arrays(entry, algorithm=method)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    all_grad_vectors = np.asarray(dataset_arrays["all_grad_vectors"], dtype=float)
    all_positions = np.asarray(dataset_arrays["all_positions"], dtype=float)
    feature_values = dataset_arrays["feature_values"]
    col_labels = dataset_arrays["col_labels"]
    if feature_values is None:
        return jsonify({"error": "Validation requires domain values in the dataset."}), 400

    feature_values = np.asarray(feature_values, dtype=float)
    n_points, n_features = feature_values.shape

    if not (0 <= feature_index < n_features):
        return jsonify({"error": f"featureIndex out of range 0..{n_features-1}"}), 400

    try:
        unique_point_indices = sorted({int(idx) for idx in point_indices})
    except Exception:
        return jsonify({"error": "pointIndices must contain integers"}), 400
    if not unique_point_indices:
        return jsonify({"error": "pointIndices must contain at least one point"}), 400
    if unique_point_indices[0] < 0 or unique_point_indices[-1] >= n_points:
        return jsonify({"error": f"pointIndices out of range 0..{n_points-1}"}), 400

    predicted_vectors = all_grad_vectors[unique_point_indices, feature_index, :] * float(delta)
    perturbed_values = feature_values.copy()
    perturbed_values[unique_point_indices, feature_index] = np.clip(
        perturbed_values[unique_point_indices, feature_index] + float(delta),
        0.0,
        1.0,
    )

    projection = fw_dim.tsne
    params = _projection_params_from_meta(projection_meta)
    baseline_projection = np.asarray(fw_dim.project_points(feature_values.tolist(), projection, params=params, seed=0), dtype=float)
    perturbed_projection = np.asarray(fw_dim.project_points(perturbed_values.tolist(), projection, params=params, seed=0), dtype=float)

    scale, rotation, translation = _estimate_similarity_transform(baseline_projection, all_positions)
    baseline_aligned = _apply_similarity_transform(baseline_projection, scale, rotation, translation)
    perturbed_aligned = _apply_similarity_transform(perturbed_projection, scale, rotation, translation)

    actual_vectors = perturbed_aligned[unique_point_indices] - baseline_aligned[unique_point_indices]
    anchors = all_positions[unique_point_indices]
    baseline_alignment_rmse = float(np.sqrt(np.mean(np.sum((baseline_aligned - all_positions) ** 2, axis=1))))

    point_results = []
    point_metrics = []
    for local_idx, point_idx in enumerate(unique_point_indices):
        predicted = predicted_vectors[local_idx]
        actual = actual_vectors[local_idx]
        metrics = _vector_metrics(predicted, actual)
        point_metrics.append(metrics)
        point_results.append({
            "index": point_idx,
            "anchor": anchors[local_idx].tolist(),
            "predicted": predicted.tolist(),
            "actual": actual.tolist(),
            "predictedEndpoint": (anchors[local_idx] + predicted).tolist(),
            "actualEndpoint": (anchors[local_idx] + actual).tolist(),
            "metrics": metrics,
        })

    centroid_anchor = anchors.mean(axis=0)
    centroid_predicted = predicted_vectors.mean(axis=0)
    centroid_actual = actual_vectors.mean(axis=0)
    centroid_metrics = _vector_metrics(centroid_predicted, centroid_actual)

    return jsonify({
        "status": "ok",
        "validationMode": "exact-tsne-rerun",
        "featureIndex": feature_index,
        "featureLabel": col_labels[feature_index],
        "delta": float(delta),
        "pointIndices": unique_point_indices,
        "centroid": {
            "anchor": centroid_anchor.tolist(),
            "predicted": centroid_predicted.tolist(),
            "actual": centroid_actual.tolist(),
            "predictedEndpoint": (centroid_anchor + centroid_predicted).tolist(),
            "actualEndpoint": (centroid_anchor + centroid_actual).tolist(),
            "metrics": centroid_metrics,
        },
        "points": point_results,
        "metrics": {
            "baselineAlignmentRmse": baseline_alignment_rmse,
            "meanErrorNorm": _mean_defined([item["errorNorm"] for item in point_metrics]),
            "meanCosineSimilarity": _mean_defined([item["cosineSimilarity"] for item in point_metrics]),
            "meanMagnitudeRatio": _mean_defined([item["magnitudeRatio"] for item in point_metrics]),
            "selectedPointCount": len(unique_point_indices),
        },
    })


@api_bp.post("/colors")
def recolor():
    """Assign colors from provided family assignments without recomputing grids."""
    body = request.get_json(silent=True) or {}
    dataset_id = body.get("dataset_id")
    fams = body.get("familyAssignments")
    entry = _load_dataset_entry(dataset_id) if dataset_id else None
    if not entry:
        return jsonify({"error": "Unknown dataset_id"}), 404
    labels = entry.get("labels") or []
    if not isinstance(fams, list) or len(fams) != len(labels):
        return jsonify({"error": "familyAssignments must be a list of length equal to number of features"}), 400
    try:
        fams_int = [int(x) for x in fams]
        from featurewind.visualization import color_system

        colors = color_system.assign_family_colors(fams_int)
        return jsonify({"colors": colors, "family_assignments": fams_int})
    except Exception as exc:
        return jsonify({"error": f"Failed to assign colors: {exc}"}), 500
