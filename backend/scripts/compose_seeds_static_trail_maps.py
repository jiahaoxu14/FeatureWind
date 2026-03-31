#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = REPO_ROOT / "output/paper_figures/static_trails/seeds_tsne"
DEFAULT_OUTPUT_PNG = DEFAULT_INPUT_DIR / "kernel_length_sampled_trails_combined.png"
DEFAULT_OUTPUT_PDF = DEFAULT_INPUT_DIR / "kernel_length_sampled_trails_combined.pdf"
DEFAULT_OVERLAY_OUTPUT_PNG = DEFAULT_INPUT_DIR / "kernel_length_sampled_trails_overlay.png"
DEFAULT_OVERLAY_OUTPUT_PDF = DEFAULT_INPUT_DIR / "kernel_length_sampled_trails_overlay.pdf"
SOURCE_FILES = [
    "point_035_kernel_length.png",
    "point_197_kernel_length.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine the left map panels from the Seeds kernel_length static-trail exports.",
    )
    parser.add_argument(
        "--mode",
        choices=["concat", "overlay"],
        default="concat",
        help="`concat` places the two map panels side by side; `overlay` merges both trails into one map.",
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="Directory containing the existing point_035/point_197 kernel_length PNGs.",
    )
    parser.add_argument(
        "--output-png",
        default=str(DEFAULT_OUTPUT_PNG),
        help="Path for the combined PNG output.",
    )
    parser.add_argument(
        "--output-pdf",
        default=str(DEFAULT_OUTPUT_PDF),
        help="Path for the combined PDF output.",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=36,
        help="Horizontal white gap, in pixels, between the two cropped map panels.",
    )
    parser.add_argument(
        "--diff-threshold",
        type=int,
        default=6,
        help="Per-channel difference threshold for overlay mode when extracting the second trail.",
    )
    return parser.parse_args()


def _find_left_panel_right_edge(image: np.ndarray) -> int:
    if image.ndim != 3:
        raise ValueError("Expected an H x W x C image array.")
    height, width = image.shape[:2]
    row_start = int(round(height * 0.14))
    row_end = int(round(height * 0.88))
    window = image[row_start:row_end, :, :3]
    nonwhite_counts = np.any(window < 245, axis=2).sum(axis=0)
    low_columns = nonwhite_counts < max(10, int(window.shape[0] * 0.01))

    candidate_runs: list[tuple[int, int]] = []
    run_start: int | None = None
    search_start = int(width * 0.25)
    search_end = int(width * 0.60)
    for idx in range(search_start, search_end):
        if bool(low_columns[idx]) and run_start is None:
            run_start = idx
        elif not bool(low_columns[idx]) and run_start is not None:
            candidate_runs.append((run_start, idx - 1))
            run_start = None
    if run_start is not None:
        candidate_runs.append((run_start, search_end - 1))

    wide_runs = [run for run in candidate_runs if run[1] - run[0] + 1 >= 18]
    if not wide_runs:
        raise ValueError("Could not detect the left-panel boundary from the source figure.")

    gap_start, gap_end = max(wide_runs, key=lambda run: run[1] - run[0])
    return int((gap_start + gap_end) // 2)


def compose_maps(*, input_dir: Path, output_png: Path, output_pdf: Path, gap: int) -> None:
    source_paths = [input_dir / name for name in SOURCE_FILES]
    missing = [path for path in source_paths if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing source PNGs: {names}")

    source_images = [Image.open(path).convert("RGB") for path in source_paths]
    image_arrays = [np.asarray(image) for image in source_images]
    crop_edges = [_find_left_panel_right_edge(array) for array in image_arrays]
    cropped_images = [
        image.crop((0, 0, crop_right, image.height))
        for image, crop_right in zip(source_images, crop_edges)
    ]

    canvas_width = sum(image.width for image in cropped_images) + max(0, int(gap))
    canvas_height = max(image.height for image in cropped_images)
    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")

    x_offset = 0
    for image in cropped_images:
        y_offset = (canvas_height - image.height) // 2
        canvas.paste(image, (x_offset, y_offset))
        x_offset += image.width + max(0, int(gap))

    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, format="PNG")
    canvas.save(output_pdf, format="PDF", resolution=300.0)


def compose_overlay(
    *,
    input_dir: Path,
    output_png: Path,
    output_pdf: Path,
    diff_threshold: int,
) -> None:
    source_paths = [input_dir / name for name in SOURCE_FILES]
    missing = [path for path in source_paths if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing source PNGs: {names}")

    source_images = [Image.open(path).convert("RGB") for path in source_paths]
    image_arrays = [np.asarray(image) for image in source_images]
    crop_edges = [_find_left_panel_right_edge(array) for array in image_arrays]
    cropped_images = [
        image.crop((0, 0, crop_right, image.height))
        for image, crop_right in zip(source_images, crop_edges)
    ]

    target_width = min(image.width for image in cropped_images)
    target_height = min(image.height for image in cropped_images)
    aligned_arrays = [
        np.asarray(image.crop((0, 0, target_width, target_height)), dtype=np.uint8)
        for image in cropped_images
    ]

    image_a = aligned_arrays[0]
    image_b = aligned_arrays[1]
    delta = np.abs(image_b.astype(np.int16) - image_a.astype(np.int16))
    mask = np.any(delta >= int(diff_threshold), axis=2)

    corner_samples = np.vstack([
        image_a[0:24, 0:24].reshape(-1, 3),
        image_a[0:24, -24:].reshape(-1, 3),
        image_a[-24:, 0:24].reshape(-1, 3),
        image_a[-24:, -24:].reshape(-1, 3),
    ]).astype(np.float32)
    background_rgb = np.median(corner_samples, axis=0)

    dist_a = np.linalg.norm(image_a.astype(np.float32) - background_rgb[None, None, :], axis=2)
    dist_b = np.linalg.norm(image_b.astype(np.float32) - background_rgb[None, None, :], axis=2)
    pick_b = mask & (dist_b > dist_a)

    merged = image_a.copy()
    merged[pick_b] = image_b[pick_b]

    canvas = Image.fromarray(merged, mode="RGB")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_png, format="PNG")
    canvas.save(output_pdf, format="PDF", resolution=300.0)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    if args.mode == "overlay":
        output_png = Path(args.output_png).resolve()
        output_pdf = Path(args.output_pdf).resolve()
        if str(args.output_png) == str(DEFAULT_OUTPUT_PNG):
            output_png = DEFAULT_OVERLAY_OUTPUT_PNG.resolve()
        if str(args.output_pdf) == str(DEFAULT_OUTPUT_PDF):
            output_pdf = DEFAULT_OVERLAY_OUTPUT_PDF.resolve()
        compose_overlay(
            input_dir=input_dir,
            output_png=output_png,
            output_pdf=output_pdf,
            diff_threshold=int(args.diff_threshold),
        )
    else:
        output_png = Path(args.output_png).resolve()
        output_pdf = Path(args.output_pdf).resolve()
        compose_maps(
            input_dir=input_dir,
            output_png=output_png,
            output_pdf=output_pdf,
            gap=int(args.gap),
        )
    print(f"Combined PNG: {output_png}")
    print(f"Combined PDF: {output_pdf}")


if __name__ == "__main__":
    main()
