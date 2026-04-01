#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = BACKEND_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from featurewind.analysis.seeds_visual_sensitivity import run_seeds_visual_sensitivity


def parse_args() -> argparse.Namespace:
    repo_root = BACKEND_ROOT.parent
    parser = argparse.ArgumentParser(
        description="Generate a compact Seeds visual-stability sensitivity figure and summary artifacts."
    )
    parser.add_argument(
        "--tmap",
        default=str(repo_root / "backend/datasets/examples/seeds/seeds_tsne.tmap"),
        help="Path to the fixed Seeds tangent-map file.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo_root / "output/paper_figures/seeds_visual_sensitivity"),
        help="Directory where the sensitivity artifacts will be written.",
    )
    parser.add_argument(
        "--feature-name",
        default="kernel_length",
        help="Feature name to analyze. Defaults to the paper case, kernel_length.",
    )
    parser.add_argument(
        "--point-index",
        type=int,
        default=35,
        help="Primary seed point index to try before deterministic fallback.",
    )
    parser.add_argument(
        "--reference-grid-res",
        type=int,
        default=25,
        help="Reference grid resolution for the baseline condition.",
    )
    parser.add_argument(
        "--reference-interpolation",
        default="linear",
        help="Reference interpolation method for the baseline condition.",
    )
    parser.add_argument(
        "--reference-mask-radius",
        type=int,
        default=1,
        help="Reference mask dilation radius, in grid cells.",
    )
    parser.add_argument(
        "--ui-snapshot-dir",
        default=None,
        help=(
            "Optional directory of per-condition UI snapshot images named "
            "`reference.png`, `grid_15.png`, `grid_20.png`, `grid_30.png`, "
            "`linear_nearest.png`, `nearest.png`, `radius_0.png`, `radius_2.png`, "
            "and `radius_3.png`."
        ),
    )
    parser.add_argument(
        "--max-figures",
        type=int,
        default=9,
        help="Maximum number of separate condition figures to export. Clamped to 9.",
    )
    parser.add_argument(
        "--include-combined",
        action="store_true",
        help="Also render the combined multi-panel figure. Separate figures are always exported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_seeds_visual_sensitivity(
        tmap_path=Path(args.tmap),
        output_dir=Path(args.output_dir),
        feature_name=str(args.feature_name),
        point_index=int(args.point_index),
        reference_grid_res=int(args.reference_grid_res),
        reference_interpolation=str(args.reference_interpolation),
        reference_mask_radius=int(args.reference_mask_radius),
        ui_snapshot_dir=None if args.ui_snapshot_dir in (None, "") else Path(args.ui_snapshot_dir),
        max_figures=int(args.max_figures),
        render_combined=bool(args.include_combined),
    )

    print("Seeds visual sensitivity analysis completed.")
    print(f"Chosen point index: {result.point_index}")
    if result.figure_png is not None and result.figure_pdf is not None:
        print(f"Figure PNG: {result.figure_png}")
        print(f"Figure PDF: {result.figure_pdf}")
    else:
        print("Combined figure: skipped")
    print(f"Standalone panel PNGs: {len(result.panel_pngs)}")
    print(f"Standalone panel PDFs: {len(result.panel_pdfs)}")
    print(f"Metrics CSV: {result.metrics_csv}")
    print(f"Summary MD: {result.summary_md}")


if __name__ == "__main__":
    main()
