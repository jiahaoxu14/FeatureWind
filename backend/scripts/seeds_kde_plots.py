"""Generate KDE plots for every feature in the Seeds dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from feature_kde_plot import plot_feature_kde

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one KDE figure per Seeds feature.")
    parser.add_argument(
        "--csv",
        default="backend/datasets/examples/seeds/seeds.csv",
        help="Path to the Seeds CSV file.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/paper_figures/seeds_kde",
        help="Directory for generated KDE figures.",
    )
    return parser.parse_args()


def resolve_label_colors(labels) -> dict[str, str]:
    unique = sorted({str(label) for label in labels})
    return {
        label: LABEL_PALETTE[idx % len(LABEL_PALETTE)]
        for idx, label in enumerate(unique)
    }


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)

    df = pd.read_csv(csv_path)
    features = [column for column in df.columns if column != "label"]
    label_colors = resolve_label_colors(df["label"].tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    for feature in features:
        outputs = plot_feature_kde(
            csv_path=str(csv_path),
            feature=feature,
            label_column="label",
            output_dir=str(output_dir),
            output_stem=f"{feature}_kde",
            title=f"Seeds: {feature.replace('_', ' ').title()} KDE",
            feature_color=None,
            label_colors=label_colors,
        )
        for output in outputs:
            print(output)


if __name__ == "__main__":
    main()
