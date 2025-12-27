#!/usr/bin/env python3
"""
Simple 3D CSV scatter plotter.

Usage examples:
  python scripts/plot_3d_scatter.py examples/singlehelix/single_helix_300.csv
  python scripts/plot_3d_scatter.py data.csv --cols x y z --color-by label --out out.png
  python scripts/plot_3d_scatter.py data.csv --cols 0 1 2 --alpha 0.6 --size 6

Notes:
- Auto-detects a header row. If present and --cols are names, they are matched.
- If no --cols provided, uses 'x y z' when available, otherwise first three columns.
- Optional --color-by can be a column name or index.
"""

import argparse
import csv
import os
from typing import List, Tuple, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np


def try_parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def detect_header(first_row: List[str]) -> bool:
    """Return True if the first row looks like a header (contains non-numeric)."""
    for cell in first_row:
        if try_parse_float(cell) is None:
            return True
    return False


def read_csv_columns(path: str,
                     cols: Optional[List[str]] = None,
                     color_by: Optional[str] = None,
                     delimiter: str = ',') -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    """
    Read selected columns from a CSV file.

    Args:
        path: CSV file path
        cols: list of 3 column specifiers (names or zero-based indices as strings)
        color_by: optional column specifier (name or index) for coloring
        delimiter: CSV delimiter

    Returns:
        (x, y, z, c, headers)
        where c is None if no color column is provided/found
    """
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            first_row = next(reader)
        except StopIteration:
            raise ValueError("Empty CSV file")

        has_header = detect_header(first_row)
        headers = first_row if has_header else []

        # If no header, reprocess first row as data
        rows_iter = reader if has_header else (row for row in [first_row] + list(reader))

        # Resolve column indices
        def resolve_col_idx(spec: str, default_idx: Optional[int] = None) -> Optional[int]:
            if spec is None:
                return default_idx
            # Try index
            try:
                return int(spec)
            except Exception:
                pass
            # Try header name
            if headers:
                low = [h.strip().lower() for h in headers]
                try:
                    return low.index(spec.strip().lower())
                except ValueError:
                    return None
            return None

        # Default columns: x y z if present, else 0 1 2
        if cols is None or len(cols) == 0:
            if headers:
                default_names = ['x', 'y', 'z']
                indices = []
                for name in default_names:
                    try:
                        indices.append(headers.index(name))
                    except ValueError:
                        indices = []
                        break
                if len(indices) == 3:
                    col_indices = indices
                else:
                    col_indices = [0, 1, 2]
            else:
                col_indices = [0, 1, 2]
        else:
            if len(cols) != 3:
                raise ValueError("--cols requires exactly three entries for x y z")
            col_indices = [resolve_col_idx(c) for c in cols]
            if any(ci is None for ci in col_indices):
                raise ValueError(f"Could not resolve one or more columns from {cols} with headers {headers}")

        color_idx = resolve_col_idx(color_by) if color_by is not None else None

        xs: List[float] = []
        ys: List[float] = []
        zs: List[float] = []
        cs: List[float] = [] if color_idx is not None else None

        for row in rows_iter:
            # Skip empty lines
            if not row:
                continue
            # Guard against short rows
            if max(col_indices + ([color_idx] if color_idx is not None else [])) >= len(row):
                continue
            x = try_parse_float(row[col_indices[0]])
            y = try_parse_float(row[col_indices[1]])
            z = try_parse_float(row[col_indices[2]])
            if x is None or y is None or z is None:
                continue
            xs.append(x)
            ys.append(y)
            zs.append(z)
            if color_idx is not None:
                cval = try_parse_float(row[color_idx])
                # If non-numeric label, try mapping to int category
                if cval is None:
                    try:
                        cval = float(abs(hash(row[color_idx])) % 1000)
                    except Exception:
                        cval = 0.0
                cs.append(cval)

        x_arr = np.asarray(xs, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        z_arr = np.asarray(zs, dtype=float)
        c_arr = np.asarray(cs, dtype=float) if cs is not None else None

        return x_arr, y_arr, z_arr, c_arr, headers


def set_axes_equal(ax):
    """Set 3D plot axes to equal scale for proper aspect."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def main():
    parser = argparse.ArgumentParser(description="Plot a 3D scatter from a CSV file")
    parser.add_argument('csv', help='Path to CSV file')
    parser.add_argument('--cols', nargs=3, metavar=('X', 'Y', 'Z'),
                        help='Columns for x y z (names or zero-based indices)')
    parser.add_argument('--color-by', help='Optional column (name or index) to color by')
    parser.add_argument('--delimiter', default=',', help='CSV delimiter (default: ,)')
    parser.add_argument('--alpha', type=float, default=0.8, help='Point alpha (default: 0.8)')
    parser.add_argument('--size', type=float, default=10.0, help='Marker size (default: 10)')
    parser.add_argument('--marker', default='o', help='Marker style (default: o)')
    parser.add_argument('--cmap', default='viridis', help='Colormap for coloring (default: viridis)')
    parser.add_argument('--out', help='Output image path (PNG). Defaults next to CSV.')
    parser.add_argument('--show', action='store_true', help='Show the plot window')

    args = parser.parse_args()

    x, y, z, c, headers = read_csv_columns(args.csv, args.cols, args.color_by, args.delimiter)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#ffffff')
    # Show grid lines on 3D axes
    ax.grid(True, alpha=0.3, linewidth=0.5)

    scatter_kwargs = dict(s=args.size, marker=args.marker, alpha=args.alpha, depthshade=False)
    # Use gray fill with black border for points
    fill_color = '#777777'
    edge_color = '#000000'
    ax.scatter(x, y, z, facecolors=fill_color, edgecolors=edge_color, linewidths=0.5, **scatter_kwargs)

    # Titles and labels
    title = os.path.basename(args.csv)
    ax.set_title(f"3D Scatter: {title}")
    if headers and args.cols is None:
        # If we auto-picked x y z from headers, use them
        for label, setter in zip(['x', 'y', 'z'], [ax.set_xlabel, ax.set_ylabel, ax.set_zlabel]):
            setter(label)
    else:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

    # Nudge Z-axis label outward so it doesn't get occluded/clipped
    try:
        current_zlabel = ax.get_zlabel() or 'z'
        # Disable auto-rotation of z label for clearer rendering
        try:
            ax.zaxis.set_rotate_label(False)
        except Exception:
            pass
        ax.set_zlabel(current_zlabel, labelpad=18)
    except Exception:
        pass

    # Improve aspect
    set_axes_equal(ax)

    # Hide tick labels (keep ticks so grid lines remain visible)
    try:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    except Exception:
        pass

    # Save
    out_path = args.out
    if not out_path:
        base, _ = os.path.splitext(args.csv)
        out_path = base + "_3d.png"
    # Add generous margins to avoid clipping 3D axis labels
    try:
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.06, top=0.94)
    except Exception:
        pass
    fig.savefig(out_path)
    print(f"Saved 3D scatter to {out_path}")

    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
