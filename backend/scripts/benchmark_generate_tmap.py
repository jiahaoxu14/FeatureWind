#!/usr/bin/env python3

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

import numpy as np

import sys

BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BACKEND_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from generate_tangent_map import extract_features_and_labels, generate_tmap


DEFAULT_DATASETS = {
    "wine": BACKEND_ROOT / "datasets" / "examples" / "wine_recognition" / "wine_recognition.csv",
    "penguin": BACKEND_ROOT / "datasets" / "examples" / "penguin" / "penguin.csv",
}


def _payload_arrays(payload):
    positions = np.asarray([entry["range"] for entry in payload["tmap"]], dtype=float)
    grads = np.asarray([entry["tangent"] for entry in payload["tmap"]], dtype=float)
    return positions, grads


def _aligned_rmse(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a_centered = a - a.mean(axis=0, keepdims=True)
    b_centered = b - b.mean(axis=0, keepdims=True)
    a_scale = max(np.linalg.norm(a_centered), 1e-12)
    b_scale = max(np.linalg.norm(b_centered), 1e-12)
    a_norm = a_centered / a_scale
    b_norm = b_centered / b_scale
    u, _, vt = np.linalg.svd(b_norm.T @ a_norm, full_matrices=False)
    rot = u @ vt
    aligned = b_norm @ rot
    return float(np.sqrt(np.mean((a_norm - aligned) ** 2)))


def _median_gradient_cosine(a, b):
    grad_a = np.asarray(a, dtype=float)
    grad_b = np.asarray(b, dtype=float)
    flat_a = grad_a.reshape(grad_a.shape[0], -1)
    flat_b = grad_b.reshape(grad_b.shape[0], -1)
    dots = np.sum(flat_a * flat_b, axis=1)
    denom = np.linalg.norm(flat_a, axis=1) * np.linalg.norm(flat_b, axis=1)
    cosine = np.divide(dots, np.maximum(denom, 1e-12))
    return float(np.median(cosine))


def _timed_generation(feature_df, labels, feature_columns, projection, quality, cache_mode, seed, perplexity, cache_dir):
    start = time.perf_counter()
    payload = generate_tmap(
        feature_df,
        projection,
        perplexity=perplexity,
        quality=quality,
        cache=cache_mode,
        seed=seed,
        labels=labels,
        feature_columns=feature_columns,
        cache_dir=cache_dir,
    )
    elapsed = time.perf_counter() - start
    return elapsed, payload


def main():
    parser = argparse.ArgumentParser(description="Benchmark .tmap generation presets")
    parser.add_argument("--projection", choices=["tsne", "mds"], default="tsne")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--perplexity", type=float, default=None)
    parser.add_argument("--cache-dir", type=str, default=None)
    args = parser.parse_args()

    if args.cache_dir:
        cache_root = Path(args.cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory()
        cache_root = Path(temp_ctx.name)

    try:
        for dataset_name, csv_path in DEFAULT_DATASETS.items():
            feature_df, labels, _, feature_columns = extract_features_and_labels(str(csv_path))
            final_time, final_payload = _timed_generation(
                feature_df,
                labels,
                feature_columns,
                args.projection,
                "final",
                "off",
                args.seed,
                args.perplexity,
                cache_root / dataset_name,
            )
            balanced_time, balanced_payload = _timed_generation(
                feature_df,
                labels,
                feature_columns,
                args.projection,
                "balanced",
                "off",
                args.seed,
                args.perplexity,
                cache_root / dataset_name,
            )
            refresh_time, _ = _timed_generation(
                feature_df,
                labels,
                feature_columns,
                args.projection,
                "balanced",
                "refresh",
                args.seed,
                args.perplexity,
                cache_root / dataset_name,
            )
            cached_time, cached_payload = _timed_generation(
                feature_df,
                labels,
                feature_columns,
                args.projection,
                "balanced",
                "auto",
                args.seed,
                args.perplexity,
                cache_root / dataset_name,
            )

            final_positions, final_grads = _payload_arrays(final_payload)
            balanced_positions, balanced_grads = _payload_arrays(balanced_payload)
            cached_positions, cached_grads = _payload_arrays(cached_payload)
            speedup = final_time / max(balanced_time, 1e-12)
            print(f"\n[{dataset_name}]")
            print(f"  final/off        : {final_time:.3f}s")
            print(f"  balanced/off     : {balanced_time:.3f}s  speedup={speedup:.2f}x")
            print(f"  balanced/refresh : {refresh_time:.3f}s")
            print(f"  balanced/auto    : {cached_time:.3f}s")
            print(f"  procrustes rmse  : {_aligned_rmse(final_positions, balanced_positions):.6f}")
            print(f"  grad cosine med  : {_median_gradient_cosine(final_grads, balanced_grads):.6f}")
            print(f"  cache parity rmse: {_aligned_rmse(balanced_positions, cached_positions):.6f}")
            print(f"  cache grad cos   : {_median_gradient_cosine(balanced_grads, cached_grads):.6f}")
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    main()
