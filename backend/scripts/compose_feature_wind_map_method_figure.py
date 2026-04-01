#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Georgia", "DejaVu Serif"]
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIL_DIR = REPO_ROOT / "output/paper_figures/trail_generation"
OUTPUT_DIR = REPO_ROOT / "output/paper_figures/method"

PANEL_SPECS = [
    (
        "A",
        "Static Trail (Single-Feature Mode)",
        TRAIL_DIR / "simple_2d_tsne_static_trail_generation_stage_4.png",
    ),
    (
        "B",
        "Static Trail (Compare Mode)",
        TRAIL_DIR / "simple_2d_tsne_static_trail_feature_comparison.png",
    ),
    (
        "C",
        "Animated Trail (Single-Feature Mode)",
        TRAIL_DIR / "simple_2d_tsne_animated_trail_generation_stage_4.png",
    ),
    (
        "D",
        "Animated Trail (Compare Mode)",
        TRAIL_DIR / "simple_2d_tsne_animated_trail_feature_comparison.png",
    ),
]


def _panel_title(label: str, title: str) -> str:
    return f"({label}) {title}"


def _add_static_callouts(ax) -> None:
    ax.annotate(
        "Seed marker",
        xy=(0.33, 0.73),
        xycoords="axes fraction",
        xytext=(0.08, 0.94),
        textcoords="axes fraction",
        fontsize=12.5,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=1.4, color="#111827"),
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.22", fc=(1, 1, 1, 0.92), ec="none"),
    )
    ax.annotate(
        "Persistent trail",
        xy=(0.34, 0.44),
        xycoords="axes fraction",
        xytext=(0.06, 0.11),
        textcoords="axes fraction",
        fontsize=12.5,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=1.4, color="#111827"),
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.22", fc=(1, 1, 1, 0.92), ec="none"),
    )
    ax.annotate(
        "Terminal arrowhead",
        xy=(0.30, 0.30),
        xycoords="axes fraction",
        xytext=(0.56, 0.08),
        textcoords="axes fraction",
        fontsize=12.5,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=1.4, color="#111827"),
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.22", fc=(1, 1, 1, 0.92), ec="none"),
    )


def _add_compare_static_callout(ax) -> None:
    ax.text(
        0.03,
        0.04,
        "One trail per selected feature\nfrom the same seed",
        transform=ax.transAxes,
        fontsize=12.3,
        ha="left",
        va="bottom",
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.24", fc=(1, 1, 1, 0.92), ec="none"),
    )


def _add_animated_callouts(ax) -> None:
    ax.annotate(
        "Particle heads",
        xy=(0.84, 0.46),
        xycoords="axes fraction",
        xytext=(0.58, 0.10),
        textcoords="axes fraction",
        fontsize=12.5,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=1.4, color="#111827"),
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.22", fc=(1, 1, 1, 0.92), ec="none"),
    )
    ax.annotate(
        "Short fading histories",
        xy=(0.30, 0.60),
        xycoords="axes fraction",
        xytext=(0.07, 0.93),
        textcoords="axes fraction",
        fontsize=12.5,
        ha="left",
        va="center",
        arrowprops=dict(arrowstyle="-", lw=1.4, color="#111827"),
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.22", fc=(1, 1, 1, 0.92), ec="none"),
    )


def _add_compare_animated_callout(ax) -> None:
    ax.text(
        0.03,
        0.04,
        "Feature-colored particle flows\nshown in one shared view",
        transform=ax.transAxes,
        fontsize=12.3,
        ha="left",
        va="bottom",
        color="#111827",
        bbox=dict(boxstyle="round,pad=0.24", fc=(1, 1, 1, 0.92), ec="none"),
    )


def compose_figure() -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 11.2), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.03, right=0.985, top=0.965, bottom=0.035, wspace=0.05, hspace=0.14)

    for ax, (label, title, image_path) in zip(axes.flat, PANEL_SPECS):
        image = mpimg.imread(image_path)
        ax.imshow(image)
        ax.set_axis_off()
        ax.text(
            0.0,
            1.04,
            _panel_title(label, title),
            transform=ax.transAxes,
            fontsize=15.5,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="#111827",
        )

    _add_static_callouts(axes[0, 0])
    _add_compare_static_callout(axes[0, 1])
    _add_animated_callouts(axes[1, 0])
    _add_compare_animated_callout(axes[1, 1])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "feature_wind_map_method_overview.png"
    pdf_path = OUTPUT_DIR / "feature_wind_map_method_overview.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    png_path, pdf_path = compose_figure()
    print(f"Method overview figure PNG: {png_path}")
    print(f"Method overview figure PDF: {pdf_path}")


if __name__ == "__main__":
    main()
