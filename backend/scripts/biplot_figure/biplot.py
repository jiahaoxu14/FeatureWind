#!/usr/bin/env python3
"""
Generate a PCA biplot for the Iris dataset.

The script:
- reads the local Iris CSV (no downloads)
- standardizes the four numeric features
- projects them to 2D with PCA
- saves the 2D coordinates and a biplot PNG showing samples and loadings

Usage:
    python backend/scripts/biplot_figure/biplot.py
    python backend/scripts/biplot_figure/biplot.py --output backend/scripts/biplot_figure/iris_pca_biplot.png --show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_FIG_PATH = SCRIPT_DIR / "iris_pca_biplot.png"
DEFAULT_PROJ_PATH = SCRIPT_DIR / "iris_pca_projection.csv"
DEFAULT_DATASET_PATH = BACKEND_ROOT / "datasets" / "examples" / "iris" / "iris.csv"
DEFAULT_LOCAL_PNG = SCRIPT_DIR / "iris_local_biplots.png"


def load_iris_dataset(csv_path: Path):
    """Load Iris features/labels from a local CSV (no network)."""
    df = pd.read_csv(csv_path)
    label_col = "species" if "species" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != label_col]

    feature_df = df[feature_cols]
    species = df[label_col].astype(str)
    ordered_species = pd.unique(species)
    categories = pd.Categorical(species, categories=ordered_species)
    labels = categories.codes
    label_names = list(categories.categories)
    return feature_df, labels, feature_cols, label_names


def run_pca(feature_df, n_components=2):
    """Standardize features and compute PCA."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_df)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(scaled)
    # Loadings scaled by sqrt of explained variance for clearer arrows
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    return scores, loadings, pca


def _loading_scale(scores: np.ndarray, loadings: np.ndarray) -> float:
    """
    Compute a scale factor so loading arrows fit nicely within the score cloud.
    """
    score_span = np.max(np.abs(scores), axis=0)
    loading_span = np.max(np.abs(loadings), axis=0)
    loading_span[loading_span == 0] = 1.0
    ratios = score_span / loading_span
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if ratios.size == 0:
        return 1.0
    return 0.6 * float(ratios.min())


def make_biplot(scores, loadings, labels, feature_names, label_names, explained):
    """Create a matplotlib Figure with a PCA biplot."""
    fig, ax = plt.subplots(figsize=(8.0, 6.5))

    palette = ["#4C78A8", "#F58518", "#54A24B"]
    for idx, name in enumerate(label_names):
        mask = labels == idx
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            label=name,
            color=palette[idx % len(palette)],
            alpha=0.78,
            edgecolors="white",
            linewidth=0.6,
            s=46,
        )

    ax.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")

    scale = _loading_scale(scores, loadings)
    for i, feat in enumerate(feature_names):
        x = loadings[i, 0] * scale
        y = loadings[i, 1] * scale
        ax.arrow(
            0,
            0,
            x,
            y,
            color="#555555",
            alpha=0.9,
            width=0.003,
            head_width=0.08,
            length_includes_head=True,
        )
        ax.text(
            x * 1.06,
            y * 1.06,
            feat,
            color="#222222",
            fontsize=9,
            ha="center",
            va="center",
        )

    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title("Iris PCA biplot")
    ax.legend(frameon=True, title="Species")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    return fig


def save_projections(scores, labels, label_names, path: Path):
    """Persist 2D PCA coordinates with species labels."""
    df = pd.DataFrame(scores, columns=["PC1", "PC2"])
    df["species_id"] = labels
    df["species"] = [label_names[i] for i in labels]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_local_biplots(
    scores: np.ndarray,
    loadings: np.ndarray,
    labels: np.ndarray,
    feature_names,
    label_names,
    path: Path,
    max_cols: int = 10,
):
    """
    Save a single-view projection where each point gets its own local axes.

    For every point we draw loading arrows that originate at that point,
    so you can see per-point “local” axes overlaid on one projection.
    """
    fig, ax = plt.subplots(figsize=(8.5, 7.0))

    palette = ["#4C78A8", "#F58518", "#54A24B"]
    scale = _loading_scale(scores, loadings)

    # Base scatter
    for idx, name in enumerate(label_names):
        mask = labels == idx
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            label=name,
            color=palette[idx % len(palette)],
            alpha=0.7,
            edgecolors="white",
            linewidth=0.6,
            s=30,
        )

    # Local axes per point
    for idx, (px, py) in enumerate(scores):
        label_idx = labels[idx] if idx < len(labels) else 0
        for f_idx, _ in enumerate(feature_names):
            dx = loadings[f_idx, 0] * scale
            dy = loadings[f_idx, 1] * scale
            ax.arrow(
                px,
                py,
                dx,
                dy,
                color="#555555",
                alpha=0.25,
                width=0.001,
                head_width=0.025,
                length_includes_head=True,
                zorder=1,
            )
        ax.axhline(py, color="#CCCCCC", linewidth=0.4, linestyle="--", alpha=0.35)
        ax.axvline(px, color="#CCCCCC", linewidth=0.4, linestyle="--", alpha=0.35)

    ax.axhline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="#AAAAAA", linewidth=0.8, linestyle="--")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Local PCA biplots (per-point axes overlay)")
    ax.legend(frameon=True, title="Species")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout(pad=0.9)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved local biplots to: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a PCA biplot and 2D projections for the Iris dataset."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_FIG_PATH,
        help=f"Path to save the biplot PNG (default: {DEFAULT_FIG_PATH})",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help=f"Path to the Iris CSV (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--projections-out",
        type=Path,
        default=DEFAULT_PROJ_PATH,
        help=(
            "CSV path for 2D PCA scores with species labels "
            f"(default: {DEFAULT_PROJ_PATH})"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="DPI for the saved plot (default: 220).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window after saving.",
    )
    parser.add_argument(
        "--local-biplots",
        type=Path,
        default=None,
        help=(
            "Optional path to save a grid of local biplots (each sample gets its own axes). "
            f"Default is {DEFAULT_LOCAL_PNG} if the flag is supplied without a value."
        ),
        nargs="?",
        const=DEFAULT_LOCAL_PNG,
    )
    parser.add_argument(
        "--local-max-cols",
        type=int,
        default=10,
        help="Maximum columns for the local biplot grid (default: 10).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    feature_df, labels, feature_names, label_names = load_iris_dataset(args.dataset)
    scores, loadings, pca = run_pca(feature_df)

    fig = make_biplot(scores, loadings, labels, feature_names, label_names, pca.explained_variance_ratio_)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved biplot to: {args.output}")

    projection_path = save_projections(scores, labels, label_names, args.projections_out)
    print(f"Saved 2D projections to: {projection_path}")

    if args.local_biplots:
        save_local_biplots(
            scores,
            loadings,
            labels,
            feature_names,
            label_names,
            args.local_biplots,
            max_cols=args.local_max_cols,
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
