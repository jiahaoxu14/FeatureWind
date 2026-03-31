#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Georgia"]
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "output/paper_figures/seeds_visual_sensitivity"
DEFAULT_OUTPUT_PNG = DEFAULT_INPUT_DIR / "seeds_visual_stability_final_figure.png"
DEFAULT_OUTPUT_PDF = DEFAULT_INPUT_DIR / "seeds_visual_stability_final_figure.pdf"
DEFAULT_METRICS_CSV = DEFAULT_INPUT_DIR / "seeds_visual_stability_metrics.csv"
PANEL_ORDER = [
    "reference",
    "grid_15",
    "grid_20",
    "grid_30",
    "nearest",
    "radius_0",
    "radius_2",
    "radius_3",
]
PANEL_CAPTIONS = {
    "reference": "(A) Reference",
    "grid_15": "(B) Grid 15",
    "grid_20": "(C) Grid 20",
    "grid_30": "(D) Grid 30",
    "nearest": "(E) Nearest",
    "radius_0": "(F) Radius 0",
    "radius_2": "(G) Radius 2",
    "radius_3": "(H) Radius 3",
}
TEXT_FONT_FAMILY = "Georgia"
TITLE_CROP_PX = 96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate the Seeds visual-sensitivity panels into a final paper figure.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing the standalone panel PNGs.",
    )
    parser.add_argument(
        "--output-png",
        default=str(DEFAULT_OUTPUT_PNG),
        help="Path for the concatenated PNG output.",
    )
    parser.add_argument(
        "--output-pdf",
        default=str(DEFAULT_OUTPUT_PDF),
        help="Path for the concatenated PDF output.",
    )
    parser.add_argument(
        "--metrics-csv",
        default=str(DEFAULT_METRICS_CSV),
        help="Path to the sensitivity metrics CSV used to label each panel.",
    )
    return parser.parse_args()


def _load_metrics(metrics_csv: Path) -> dict[str, dict[str, str]]:
    with metrics_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["condition_id"]): row for row in rows}


def _metric_line(metric_row: dict[str, str]) -> str:
    support_iou = float(metric_row["support_iou"])
    field_cosine = float(metric_row["field_cosine_mean"])
    trail_dev = float(metric_row["trail_mean_deviation"])
    return f"IoU {support_iou:.2f}  |  cos {field_cosine:.2f}  |  dev {trail_dev:.3f}"


def compose_figure(*, input_dir: Path, metrics_csv: Path, output_png: Path, output_pdf: Path) -> None:
    image_paths = [
        input_dir / f"seeds_visual_stability_{condition_id}.png"
        for condition_id in PANEL_ORDER
    ]
    missing = [path for path in image_paths if not path.exists()]
    if missing:
        names = ", ".join(str(path.name) for path in missing)
        raise FileNotFoundError(f"Missing panel PNGs: {names}")

    metrics_by_id = _load_metrics(metrics_csv)
    missing_metrics = [condition_id for condition_id in PANEL_ORDER if condition_id not in metrics_by_id]
    if missing_metrics:
        names = ", ".join(missing_metrics)
        raise FileNotFoundError(f"Missing metrics rows for: {names}")

    images = [mpimg.imread(path)[TITLE_CROP_PX:, ...] for path in image_paths]
    fig, axes = plt.subplots(2, 4, figsize=(15.8, 9.0), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.012, right=0.988, top=0.965, bottom=0.055, wspace=0.035, hspace=0.30)

    for ax, image, condition_id in zip(axes.ravel(), images, PANEL_ORDER):
        ax.imshow(image)
        ax.set_axis_off()
        ax.text(
            0.5,
            1.095,
            PANEL_CAPTIONS[condition_id],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=13.0,
            color="#1f1b17",
            fontfamily=TEXT_FONT_FAMILY,
            fontweight="normal",
        )
        ax.text(
            0.5,
            1.025,
            _metric_line(metrics_by_id[condition_id]),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=11.2,
            color="#1f1b17",
            fontfamily=TEXT_FONT_FAMILY,
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(output_pdf, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    compose_figure(
        input_dir=Path(args.input_dir).resolve(),
        metrics_csv=Path(args.metrics_csv).resolve(),
        output_png=Path(args.output_png).resolve(),
        output_pdf=Path(args.output_pdf).resolve(),
    )
    print(f"Final PNG: {Path(args.output_png).resolve()}")
    print(f"Final PDF: {Path(args.output_pdf).resolve()}")


if __name__ == "__main__":
    main()
