#!/usr/bin/env python3
"""
Generate a synthetic dataset of two separated 3D rings:
- Ring A: lies in the XY plane (z = z0)
- Ring B: lies in the XZ plane (y = y0)

Rings are spatially separated (not interlocked) by offsets greater than their radii.
Saves a CSV with columns: x, y, z, label
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path


def generate_two_rings(n_per_ring=300, radius=1.0, xy_z0=0.0, xz_y0=3.0,
                       center_xy=(0.0, 0.0), center_xz=(0.0, 0.0), noise_scale=0.0,
                       random_seed=42):
    """Create two separated 3D rings.

    Args:
        n_per_ring: Number of points per ring
        radius: Circle radius for both rings
        xy_z0: Z coordinate of the XY-plane ring
        xz_y0: Y coordinate of the XZ-plane ring (separation; set > radius to avoid interlock)
        center_xy: (cx, cy) center of the XY ring in the XY plane
        center_xz: (cx, cz) center of the XZ ring in the XZ plane
        noise_scale: Optional Gaussian noise on coordinates
        random_seed: RNG seed for reproducibility

    Returns:
        pandas.DataFrame with columns [x, y, z, label]
    """
    rng = np.random.default_rng(random_seed)

    # Parameterizations
    t_a = np.linspace(0, 2 * np.pi, n_per_ring, endpoint=False)
    t_b = np.linspace(0, 2 * np.pi, n_per_ring, endpoint=False)

    # Ring A (XY plane): (x, y) circle, z fixed
    cx_a, cy_a = center_xy
    x_a = cx_a + radius * np.cos(t_a)
    y_a = cy_a + radius * np.sin(t_a)
    z_a = np.full_like(x_a, xy_z0)

    # Ring B (XZ plane): (x, z) circle, y fixed
    cx_b, cz_b = center_xz
    x_b = cx_b + radius * np.cos(t_b)
    z_b = cz_b + radius * np.sin(t_b)
    y_b = np.full_like(x_b, xz_y0)

    if noise_scale and noise_scale > 0:
        x_a += rng.normal(0, noise_scale, x_a.shape)
        y_a += rng.normal(0, noise_scale, y_a.shape)
        z_a += rng.normal(0, noise_scale, z_a.shape)
        x_b += rng.normal(0, noise_scale, x_b.shape)
        y_b += rng.normal(0, noise_scale, y_b.shape)
        z_b += rng.normal(0, noise_scale, z_b.shape)

    # Labels: 0 for XY ring, 1 for XZ ring
    lab_a = np.zeros_like(x_a, dtype=int)
    lab_b = np.ones_like(x_b, dtype=int)

    x = np.concatenate([x_a, x_b])
    y = np.concatenate([y_a, y_b])
    z = np.concatenate([z_a, z_b])
    label = np.concatenate([lab_a, lab_b])

    df = pd.DataFrame({
        'x': x,
        'y': y,
        'z': z,
        'label': label,
    })
    return df


def main():
    # Defaults
    n_per_ring = 300
    radius = 1.0
    xy_z0 = 0.0
    xz_y0 = 3.0  # Separation along y; > radius ensures non-interlocking
    noise_scale = 0.0

    # CLI parsing (positional: n_per_ring, optional flags kept minimal)
    if len(sys.argv) > 1:
        try:
            n_per_ring = int(sys.argv[1])
        except ValueError:
            print(f"Invalid n_per_ring '{sys.argv[1]}', using default {n_per_ring}")
    if len(sys.argv) > 2:
        try:
            radius = float(sys.argv[2])
        except ValueError:
            print(f"Invalid radius '{sys.argv[2]}', using default {radius}")
    if len(sys.argv) > 3:
        try:
            noise_scale = float(sys.argv[3])
        except ValueError:
            print(f"Invalid noise_scale '{sys.argv[3]}', using default {noise_scale}")

    print(f"Generating two rings: n_per_ring={n_per_ring}, radius={radius}, noise={noise_scale}")
    df = generate_two_rings(n_per_ring=n_per_ring, radius=radius, noise_scale=noise_scale)

    out_dir = Path('examples/tworings')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"two_rings_n{n_per_ring}_r{radius:.2f}.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved dataset to: {out_path}")
    print(f"Shape: {df.shape}")
    print("Label counts:", df['label'].value_counts().sort_index().to_dict())
    print("Head:\n", df.head())


if __name__ == '__main__':
    main()

