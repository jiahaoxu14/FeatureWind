#!/usr/bin/env python3
"""Generate a standalone legend for the frontend Point Color By colormap."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


# Match the viridis anchor stops currently interpolated in the frontend.
VIRIDIS_STOPS = [
    (0.000, (68 / 255.0, 1 / 255.0, 84 / 255.0)),
    (0.125, (71 / 255.0, 45 / 255.0, 123 / 255.0)),
    (0.250, (59 / 255.0, 82 / 255.0, 139 / 255.0)),
    (0.375, (44 / 255.0, 114 / 255.0, 142 / 255.0)),
    (0.500, (33 / 255.0, 145 / 255.0, 140 / 255.0)),
    (0.625, (40 / 255.0, 174 / 255.0, 128 / 255.0)),
    (0.750, (94 / 255.0, 201 / 255.0, 98 / 255.0)),
    (0.875, (173 / 255.0, 220 / 255.0, 48 / 255.0)),
    (1.000, (253 / 255.0, 231 / 255.0, 37 / 255.0)),
]


def build_colormap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("featurewind_point_color_by", VIRIDIS_STOPS)


def generate_point_color_legend(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmap = build_colormap()
    gradient = np.linspace(0.0, 1.0, 512, dtype=float)[:, None]

    fig = plt.figure(figsize=(0.85, 7.2), dpi=220)
    ax = fig.add_axes([0.40, 0.03, 0.20, 0.94])
    ax.imshow(gradient, aspect="auto", cmap=cmap, origin="lower", extent=(0.0, 1.0, 0.0, 1.0))
    ax.set_xticks([])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["0.0", "0.5", "1.0"], fontsize=10)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#111827")

    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Point Color By legend PNG.")
    parser.add_argument(
        "--output",
        default="output/paper_figures/point_color_by_viridis_legend.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = generate_point_color_legend(Path(args.output))
    print(f"Saved point-color legend to {output_path}")


if __name__ == "__main__":
    main()
