"""Generate a 1D KDE plot for a feature in a CSV dataset."""

from __future__ import annotations

import argparse
import os
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PALETTE = [
    "#7A1E48",
    "#2B6CB0",
    "#2F855A",
    "#C05621",
    "#6B46C1",
    "#B83280",
]


def _pretty_label(name: str) -> str:
    return name.replace("_", " ").strip().title()


def _sort_labels(values: Iterable[object]) -> list[object]:
    unique = list(dict.fromkeys(values))
    try:
        return sorted(unique, key=float)
    except (TypeError, ValueError):
        return sorted(unique, key=str)


def silverman_bandwidth(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n < 2:
        return 0.1

    std = float(np.std(values, ddof=1))
    q75, q25 = np.percentile(values, [75, 25])
    iqr = float(q75 - q25)
    scale = min(std, iqr / 1.34) if iqr > 0 else std
    if not np.isfinite(scale) or scale <= 0:
        scale = std if std > 0 else max(np.ptp(values) / 6.0, 1e-3)
    if scale <= 0:
        scale = 1e-3
    return max(0.9 * scale * (n ** (-1.0 / 5.0)), 1e-3)


def gaussian_kde_1d(values: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=float)
    grid = np.asarray(grid, dtype=float)
    bandwidth = silverman_bandwidth(values)
    scaled = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * scaled * scaled).sum(axis=1)
    density /= values.size * bandwidth * np.sqrt(2.0 * np.pi)
    return density, bandwidth


def plot_feature_kde(
    csv_path: str,
    feature: str,
    label_column: str,
    output_dir: str,
    output_stem: str | None = None,
    title: str | None = None,
) -> list[str]:
    df = pd.read_csv(csv_path)
    if feature not in df.columns:
        raise ValueError(f"Missing feature column: {feature}")
    if label_column not in df.columns:
        raise ValueError(f"Missing label column: {label_column}")

    plot_df = df[[feature, label_column]].dropna().copy()
    plot_df[feature] = pd.to_numeric(plot_df[feature], errors="raise")

    x = plot_df[feature].to_numpy(dtype=float)
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x_span = max(x_max - x_min, 1e-3)
    x_pad = x_span * 0.12
    grid = np.linspace(x_min - x_pad, x_max + x_pad, 512)

    labels = _sort_labels(plot_df[label_column].tolist())
    curves = []
    for idx, label in enumerate(labels):
        values = plot_df.loc[plot_df[label_column] == label, feature].to_numpy(dtype=float)
        density, bandwidth = gaussian_kde_1d(values, grid)
        curves.append(
            {
                "label": label,
                "values": values,
                "density": density,
                "bandwidth": bandwidth,
                "color": PALETTE[idx % len(PALETTE)],
            }
        )

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfaf7")

    max_density = 0.0
    for curve in curves:
        label_text = f"Class {curve['label']}"
        color = curve["color"]
        density = curve["density"]
        max_density = max(max_density, float(np.max(density)))

        ax.fill_between(grid, density, color=color, alpha=0.18, zorder=2)
        ax.plot(grid, density, color=color, linewidth=2.4, label=label_text, zorder=3)

        rug_y0 = -0.018 - 0.01 * (labels.index(curve["label"]) % 2)
        rug_y1 = rug_y0 + 0.008
        for value in curve["values"]:
            ax.plot([value, value], [rug_y0, rug_y1], color=color, alpha=0.55, linewidth=0.9, zorder=1)

    ax.grid(axis="y", color="#d9d3c8", linewidth=0.8, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#7b7467")
    ax.spines["bottom"].set_color("#7b7467")
    ax.tick_params(colors="#342f28", labelsize=11)

    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-0.04, max_density * 1.12 if max_density > 0 else 1.0)
    ax.set_xlabel(_pretty_label(feature), fontsize=12, color="#1f1b17")
    ax.set_ylabel("Density", fontsize=12, color="#1f1b17")

    if title is None:
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
        title = f"{_pretty_label(dataset_name)}: {_pretty_label(feature)} KDE"
    ax.set_title(title, fontsize=14, fontweight="bold", color="#1f1b17", pad=12)

    legend = ax.legend(frameon=False, fontsize=11, loc="upper right")
    for text in legend.get_texts():
        text.set_color("#1f1b17")

    output_stem = output_stem or f"{feature}_kde"
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{output_stem}.png")
    pdf_path = os.path.join(output_dir, f"{output_stem}.pdf")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a feature KDE plot from a CSV dataset.")
    parser.add_argument("csv_path", help="Path to the dataset CSV")
    parser.add_argument("--feature", required=True, help="Feature column to plot")
    parser.add_argument("--label-column", default="class", help="Column used to split KDE curves")
    parser.add_argument("--output-dir", default="output/paper_figures", help="Directory for generated files")
    parser.add_argument("--output-stem", default=None, help="Base filename for the output files")
    parser.add_argument("--title", default=None, help="Optional plot title override")
    args = parser.parse_args()

    outputs = plot_feature_kde(
        csv_path=args.csv_path,
        feature=args.feature,
        label_column=args.label_column,
        output_dir=args.output_dir,
        output_stem=args.output_stem,
        title=args.title,
    )
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
