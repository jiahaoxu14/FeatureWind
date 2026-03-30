"""Generate a scatterplot matrix for size-related Seeds features."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from feature_kde_plot import gaussian_kde_1d


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

FEATURES = [
    "kernel_width",
    "kernel_length",
    "area",
    "perimeter",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Seeds scatterplot matrix for size-related features."
    )
    parser.add_argument(
        "--csv",
        default="backend/datasets/examples/seeds/seeds.csv",
        help="Path to the Seeds CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/paper_figures",
        help="Directory for generated figure files.",
    )
    parser.add_argument(
        "--output-stem",
        default="seeds_relationship_pairplot",
        help="Base filename for the output figure.",
    )
    return parser.parse_args()


def _pretty_label(name: str) -> str:
    return name.replace("_", " ").strip().title()


def resolve_label_colors(labels: list[object]) -> dict[str, str]:
    unique = sorted({str(label) for label in labels})
    return {
        label: LABEL_PALETTE[idx % len(LABEL_PALETTE)]
        for idx, label in enumerate(unique)
    }


def draw_pairplot(
    df: pd.DataFrame,
    output_dir: Path,
    output_stem: str,
) -> list[Path]:
    label_colors = resolve_label_colors(df["label"].tolist())
    labels = sorted(label_colors)

    fig, axes = plt.subplots(
        len(FEATURES),
        len(FEATURES),
        figsize=(11.5, 11.0),
        sharex="col",
    )
    fig.patch.set_facecolor("white")

    feature_ranges: dict[str, tuple[float, float, np.ndarray]] = {}
    for feature in FEATURES:
        values = pd.to_numeric(df[feature], errors="raise").to_numpy(dtype=float)
        value_min = float(np.min(values))
        value_max = float(np.max(values))
        span = max(value_max - value_min, 1e-3)
        pad = span * 0.08
        grid = np.linspace(value_min - pad, value_max + pad, 512)
        feature_ranges[feature] = (value_min - pad, value_max + pad, grid)

    for row_idx, y_feature in enumerate(FEATURES):
        for col_idx, x_feature in enumerate(FEATURES):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#fbfaf7")
            ax.grid(color="#e5ded1", linewidth=0.65, alpha=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#9a9488")
            ax.spines["bottom"].set_color("#9a9488")
            ax.tick_params(colors="#3a342e", labelsize=9)

            x_min, x_max, x_grid = feature_ranges[x_feature]
            ax.set_xlim(x_min, x_max)

            if row_idx == col_idx:
                max_density = 0.0
                for label in labels:
                    label_mask = df["label"].astype(str) == label
                    values = pd.to_numeric(
                        df.loc[label_mask, x_feature],
                        errors="raise",
                    ).to_numpy(dtype=float)
                    density, _ = gaussian_kde_1d(values, x_grid)
                    color = label_colors[label]
                    max_density = max(max_density, float(np.max(density)))
                    ax.fill_between(x_grid, density, color=color, alpha=0.22, zorder=2)
                    ax.plot(x_grid, density, color=color, linewidth=2.0, zorder=3)

                ax.set_ylim(0.0, max_density * 1.12 if max_density > 0 else 1.0)
                ax.set_yticks([])
            else:
                y_min, y_max, _ = feature_ranges[y_feature]
                ax.set_ylim(y_min, y_max)
                for label in labels:
                    label_mask = df["label"].astype(str) == label
                    ax.scatter(
                        pd.to_numeric(df.loc[label_mask, x_feature], errors="raise"),
                        pd.to_numeric(df.loc[label_mask, y_feature], errors="raise"),
                        s=28,
                        color=label_colors[label],
                        alpha=0.82,
                        edgecolors="white",
                        linewidths=0.45,
                        zorder=3,
                    )

            if row_idx == len(FEATURES) - 1:
                ax.set_xlabel(_pretty_label(x_feature), fontsize=10, color="#1f1b17")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if col_idx == 0 and row_idx != col_idx:
                ax.set_ylabel(_pretty_label(y_feature), fontsize=10, color="#1f1b17")
            elif col_idx == 0:
                ax.set_ylabel("Density", fontsize=10, color="#1f1b17")
            else:
                ax.set_ylabel("")
                if row_idx != col_idx:
                    ax.tick_params(labelleft=False)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            markersize=8,
        )
        for label, color in label_colors.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.975),
        ncol=len(handles),
        frameon=False,
        fontsize=11,
    )
    fig.suptitle(
        "Seeds: Relationships Among Kernel Size and Shape Features",
        fontsize=16,
        fontweight="bold",
        color="#1f1b17",
        y=0.995,
    )
    fig.text(
        0.5,
        0.963,
        "Scatterplots show pairwise structure by species; diagonal panels show per-species KDE.",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#5d554d",
    )
    fig.tight_layout(rect=[0.03, 0.04, 0.99, 0.93])

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)

    df = pd.read_csv(csv_path)
    missing = [feature for feature in FEATURES if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if "label" not in df.columns:
        raise ValueError("Missing required label column: label")

    outputs = draw_pairplot(df=df, output_dir=output_dir, output_stem=args.output_stem)
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
