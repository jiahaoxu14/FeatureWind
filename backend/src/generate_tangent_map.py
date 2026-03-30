#!/usr/bin/env python3
"""
Generate tangent maps from CSV datasets with automatic label handling.

This script processes CSV files by:
1. Extracting labels from the dataset
2. Running tangent map generation on features only
3. Adding labels back to the final tangent map
4. Creating a complete .tmap file for FeatureWind visualization
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BACKEND_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from featurewind.core.tangent_map import assemble_tangent_entries, check_and_normalize_features, compute_tangent_map_data
from featurewind.preprocessing.csv_label_utils import identify_label_column


TMAP_CACHE_CODE_VERSION = "tmap_accel_v1"
DEFAULT_TMAP_CACHE_DIR = BACKEND_ROOT / "var" / "cache" / "tmap"


def extract_features_and_labels(csv_file):
    print(f"Loading dataset: {csv_file}")
    df = pd.read_csv(csv_file)
    label_column = identify_label_column(df)

    if label_column:
        print(f"Found label column: '{label_column}'")
        labels = df[label_column].tolist()
        feature_columns = [col for col in df.columns if col != label_column]
        feature_df = df[feature_columns]
    else:
        print("No label column found, treating all columns as features")
        labels = [0] * len(df)
        feature_columns = df.columns.tolist()
        feature_df = df
        label_column = None

    print(f"Features: {feature_columns}")
    print(f"Dataset shape: {df.shape}, Features: {feature_df.shape}")
    return feature_df, labels, label_column, feature_columns


def _normalize_projection_name(projection):
    name = str(projection).strip().lower()
    if name not in {"tsne", "mds"}:
        raise ValueError("Projection must be 'tsne' or 'mds'")
    return name


def _normalize_quality_name(quality):
    name = str(quality or "balanced").strip().lower()
    if name not in {"draft", "balanced", "final"}:
        raise ValueError("quality must be one of: draft, balanced, final")
    return name


def _normalize_cache_policy(cache):
    name = str(cache or "auto").strip().lower()
    if name not in {"auto", "off", "refresh"}:
        raise ValueError("cache must be one of: auto, off, refresh")
    return name


def _coerce_feature_frame(points, feature_columns=None):
    if isinstance(points, pd.DataFrame):
        frame = points.copy()
    else:
        array = np.asarray(points, dtype=float)
        if array.ndim != 2:
            raise ValueError("points must be a 2D array or DataFrame")
        columns = list(feature_columns) if feature_columns is not None else [f"Feature {i}" for i in range(array.shape[1])]
        frame = pd.DataFrame(array, columns=columns)

    frame = frame.apply(pd.to_numeric, errors="raise")
    if feature_columns is not None:
        if len(feature_columns) != frame.shape[1]:
            raise ValueError("feature_columns length must match the number of columns")
        frame.columns = list(feature_columns)
    return frame


def _default_cache_dir(cache_dir=None):
    if cache_dir is not None:
        return Path(cache_dir)
    return DEFAULT_TMAP_CACHE_DIR


def _build_cache_key(normalized_points, projection, perplexity, quality, seed):
    points_array = np.asarray(normalized_points, dtype=np.float32, order="C")
    hasher = hashlib.sha256()
    hasher.update(np.asarray(points_array.shape, dtype=np.int64).tobytes())
    hasher.update(points_array.tobytes(order="C"))
    hasher.update(str(projection).strip().lower().encode("utf-8"))
    perplexity_value = "none" if perplexity is None else f"{float(perplexity):.12g}"
    hasher.update(perplexity_value.encode("utf-8"))
    hasher.update(str(quality).strip().lower().encode("utf-8"))
    hasher.update(str(int(seed)).encode("utf-8"))
    hasher.update(TMAP_CACHE_CODE_VERSION.encode("utf-8"))
    return hasher.hexdigest()


def _cache_paths(cache_dir, cache_key):
    cache_root = Path(cache_dir)
    return {
        "root": cache_root,
        "base": cache_root / f"{cache_key}.base.npz",
        "full": cache_root / f"{cache_key}.full.npz",
    }


def _build_final_payload(points, positions, grads, feature_columns, labels):
    entries = assemble_tangent_entries(points, positions, grads)
    normalized_labels = list(labels) if labels is not None else [0] * len(entries)
    if len(normalized_labels) < len(entries):
        normalized_labels.extend([0] * (len(entries) - len(normalized_labels)))

    for idx, tangent_entry in enumerate(entries):
        tangent_entry["class"] = normalized_labels[idx]
        tangent_entry["label"] = False

    return {"tmap": entries, "Col_labels": list(feature_columns)}


def _save_full_cache(cache_file, points, positions, grads, feature_columns, labels, projection, quality, perplexity, seed):
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "feature_columns": list(feature_columns),
            "labels": list(labels) if labels is not None else None,
            "projection": str(projection),
            "quality": str(quality),
            "perplexity": None if perplexity is None else float(perplexity),
            "seed": int(seed),
        }
        np.savez_compressed(
            cache_file,
            points=np.asarray(points, dtype=np.float32),
            positions=np.asarray(positions, dtype=np.float32),
            grads=np.asarray(grads, dtype=np.float32),
            meta_json=np.asarray(json.dumps(meta)),
        )
    except Exception as exc:
        print(f"Failed to save full cache ({exc}).")


def _load_full_cache(cache_file):
    try:
        with np.load(cache_file, allow_pickle=False) as cached:
            points = np.asarray(cached["points"], dtype=float)
            positions = np.asarray(cached["positions"], dtype=float)
            grads = np.asarray(cached["grads"], dtype=float)
            meta = json.loads(str(cached["meta_json"].tolist()))
        return {
            "points": points,
            "positions": positions,
            "grads": grads,
            "meta": meta,
        }
    except Exception as exc:
        print(f"Failed to load full cache ({exc}); recomputing.")
        return None


def generate_tmap(
    points,
    projection,
    *,
    perplexity=None,
    quality="balanced",
    cache="auto",
    seed=0,
    labels=None,
    feature_columns=None,
    cache_dir=None,
):
    projection_name = _normalize_projection_name(projection)
    quality_name = _normalize_quality_name(quality)
    cache_policy = _normalize_cache_policy(cache)
    seed_value = int(seed)

    feature_df = _coerce_feature_frame(points, feature_columns=feature_columns)
    inferred_feature_columns = list(feature_df.columns)
    normalized_points = np.asarray(
        check_and_normalize_features(feature_df.to_numpy(dtype=float).tolist()),
        dtype=np.float32,
    )

    cache_key = _build_cache_key(normalized_points, projection_name, perplexity, quality_name, seed_value)
    cache_paths = _cache_paths(_default_cache_dir(cache_dir), cache_key)

    if cache_policy == "auto" and cache_paths["full"].exists():
        cached = _load_full_cache(cache_paths["full"])
        if cached is not None:
            print(f"Loaded tangent-map payload from cache: {cache_paths['full']}")
            cached_meta = cached["meta"]
            cached_columns = inferred_feature_columns or cached_meta.get("feature_columns") or []
            cached_labels = labels if labels is not None else cached_meta.get("labels")
            return _build_final_payload(
                cached["points"],
                cached["positions"],
                cached["grads"],
                cached_columns,
                cached_labels,
            )

    params = {
        "quality": quality_name,
        "cache": cache_policy,
        "seed": seed_value,
    }
    if perplexity is not None:
        params["perplexity"] = float(perplexity)
    if cache_policy != "off":
        params["base_state_cache_path"] = str(cache_paths["base"])

    tangent_data = compute_tangent_map_data(
        normalized_points.tolist(),
        projection_name,
        params=params,
        normalize=False,
    )
    final_payload = _build_final_payload(
        tangent_data["points"],
        tangent_data["positions"],
        tangent_data["grads"],
        inferred_feature_columns,
        labels,
    )

    if cache_policy != "off":
        _save_full_cache(
            cache_paths["full"],
            tangent_data["points"],
            tangent_data["positions"],
            tangent_data["grads"],
            inferred_feature_columns,
            labels,
            projection_name,
            quality_name,
            perplexity,
            seed_value,
        )

    return final_payload


def save_tmap_payload(final_data, output_name=None, input_csv_path=None):
    if input_csv_path:
        input_dir = Path(input_csv_path).parent
        if output_name:
            output_file = input_dir / f"{output_name}.tmap"
        else:
            output_file = input_dir / f"{Path(input_csv_path).stem}.tmap"
    else:
        output_file = Path(f"{output_name}.tmap" if output_name else "output.tmap")

    with open(output_file, "w") as f:
        json.dump(final_data, f)

    print(f"Enhanced tangent map saved to: {output_file}")
    print(f"Total entries: {len(final_data.get('tmap', []))}")
    print(f"Feature columns: {final_data.get('Col_labels', [])}")
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Generate tangent maps from CSV datasets with automatic label handling"
    )
    parser.add_argument("csv_file", help="Input CSV file")
    parser.add_argument("projection", choices=["tsne", "mds"], help="Projection method")
    parser.add_argument("output_name", nargs="?", help="Output filename (without extension)")
    parser.add_argument(
        "--perplexity",
        type=float,
        default=None,
        help="t-SNE perplexity. Applies when projection=tsne.",
    )
    parser.add_argument(
        "--quality",
        choices=["draft", "balanced", "final"],
        default="balanced",
        help="Generation quality preset.",
    )
    parser.add_argument(
        "--cache",
        choices=["auto", "off", "refresh"],
        default="auto",
        help="Cache mode for base t-SNE state and full tangent-map payload.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for deterministic t-SNE generation.",
    )

    args = parser.parse_args()

    try:
        feature_df, labels, label_column, feature_columns = extract_features_and_labels(args.csv_file)
        del label_column
        final_data = generate_tmap(
            feature_df,
            args.projection,
            perplexity=args.perplexity,
            quality=args.quality,
            cache=args.cache,
            seed=args.seed,
            labels=labels,
            feature_columns=feature_columns,
        )
        output_file = save_tmap_payload(final_data, args.output_name, args.csv_file)
        print(f"\n✓ Successfully generated tangent map: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
