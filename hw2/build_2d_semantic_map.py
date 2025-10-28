#!/usr/bin/env python3
"""
2D semantic map construction for apartment_0 (first floor).

Steps implemented:
- Load 3D semantic point cloud from `semantic_3d_pointcloud/point.npy` and colors
  from `semantic_3d_pointcloud/color0255.npy` (preferred) or `color01.npy`.
- Remove floor and ceiling points based on Y coordinate thresholds.
- Save filtered coordinates and colors (x,z,r,g,b) to CSV and NPZ files.
- Render a 2D scatter map (x vs z) and save as `map.png`.
- Export a JSON file with the pixel↔Habitat-coordinate transform parameters.

Notes:
- Scale relationship: apartment_0 coords = points array * 10000.0 / 255.0
- Colors: `color01.npy` is in [0,1]; `color0255.npy` is in [0,255].
- If `pandas`/`openpyxl` are available and `--verify-colors-with-excel` is used,
  the script will check color consistency with `color_coding_semantic_segmentation_classes.xlsx`.

Usage (from repo root):
  python build_2d_semantic_map.py \
    --data-dir semantic_3d_pointcloud \
    --out-dir . \
    --target-longer-side 2000 \
    --floor-frac 0.01 --ceiling-frac 0.01

Outputs (written to --out-dir):
- map.png                      : rendered 2D semantic map
- map_points.csv               : filtered (x,z,r,g,b) rows
- map_points.npz               : filtered arrays (x,y,z,rgb255)
- map_transform.json           : pixel↔Habitat-coordinate mapping metadata

"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np


def load_points_and_colors(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load point coordinates and colors from npy files.

    Returns:
        points_apartment (N,3) float64: apartment_0 scaled coordinates.
        colors_255 (N,3) uint8: RGB in [0,255].
    """
    point_path = os.path.join(data_dir, "point.npy")
    color255_path = os.path.join(data_dir, "color0255.npy")
    color01_path = os.path.join(data_dir, "color01.npy")

    if not os.path.isfile(point_path):
        raise FileNotFoundError(f"Missing points file: {point_path}")

    points = np.load(point_path)  # shape (N,3), expected float in [0,255]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Unexpected points shape {points.shape}, expected (N,3)")

    # Scale to apartment_0 coordinates
    scale_factor = 10000.0 / 255.0
    points_apartment = points.astype(np.float64) * scale_factor

    # Load colors, prefer 0..255 version
    if os.path.isfile(color255_path):
        colors = np.load(color255_path)
    elif os.path.isfile(color01_path):
        colors = np.load(color01_path)
        # convert [0,1] -> [0,255]
        colors = np.clip(np.rint(colors * 255.0), 0, 255)
    else:
        raise FileNotFoundError(
            f"Missing color files in {data_dir}. Expect one of 'color0255.npy' or 'color01.npy'"
        )

    if colors.ndim != 2 or colors.shape[1] != 3 or colors.shape[0] != points.shape[0]:
        raise ValueError(
            f"Unexpected colors shape {colors.shape}, expected (N,3) matching points (N={points.shape[0]})"
        )

    # Normalize to uint8 0..255
    if np.issubdtype(colors.dtype, np.floating):
        # Could be already in [0,255] float
        cmin, cmax = float(colors.min()), float(colors.max())
        if cmax <= 1.0 + 1e-6:  # safety
            colors = np.clip(np.rint(colors * 255.0), 0, 255)
        colors_255 = np.clip(np.rint(colors), 0, 255).astype(np.uint8)
    else:
        # ints
        colors_255 = np.clip(colors, 0, 255).astype(np.uint8)

    return points_apartment, colors_255


def filter_floor_ceiling(
    points_apartment: np.ndarray,
    colors_255: np.ndarray,
    floor_frac: float = 0.01,
    ceiling_frac: float = 0.01,
    floor_abs: Optional[float] = None,
    ceiling_abs: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Remove floor and ceiling points using Y thresholds.

    Two strategies:
    - Fractional (percentile-based): remove bottom/top fractions of Y range.
    - Absolute (unit-based): remove points within a fixed margin from min/max Y.

    Args:
        points_apartment: (N,3) float array (x,y,z).
        colors_255: (N,3) uint8 RGB.
        floor_frac: bottom fraction (0..1) to remove; ignored if floor_abs provided.
        ceiling_frac: top fraction (0..1) to remove; ignored if ceiling_abs provided.
        floor_abs: remove points with y <= y_min + floor_abs (apartment units).
        ceiling_abs: remove points with y >= y_max - ceiling_abs (apartment units).

    Returns:
        points_f: filtered (M,3)
        colors_f: filtered (M,3)
        info: thresholds and stats
    """
    y = points_apartment[:, 1]
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_rng = float(y_max - y_min) if float(y_max - y_min) > 0 else 1.0

    if floor_abs is not None:
        y_floor_thr = y_min + float(floor_abs)
    else:
        y_floor_thr = float(np.quantile(y, min(max(floor_frac, 0.0), 1.0)))

    if ceiling_abs is not None:
        y_ceil_thr = y_max - float(ceiling_abs)
    else:
        y_ceil_thr = float(np.quantile(y, 1.0 - min(max(ceiling_frac, 0.0), 1.0)))

    keep_mask = (y > y_floor_thr) & (y < y_ceil_thr)
    points_f = points_apartment[keep_mask]
    colors_f = colors_255[keep_mask]

    info = {
        "y_min": y_min,
        "y_max": y_max,
        "y_range": y_rng,
        "floor_threshold": y_floor_thr,
        "ceiling_threshold": y_ceil_thr,
        "kept_points": int(points_f.shape[0]),
        "removed_points": int(points_apartment.shape[0] - points_f.shape[0]),
    }

    return points_f, colors_f, info


def save_points(out_dir: str, points: np.ndarray, colors_255: np.ndarray) -> Tuple[str, str]:
    """Save filtered points and colors to NPZ and CSV.

    CSV contains x,z,r,g,b columns.
    NPZ contains x,y,z and rgb255 arrays.
    """
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "map_points.npz")
    csv_path = os.path.join(out_dir, "map_points.csv")

    np.savez_compressed(
        npz_path,
        x=points[:, 0].astype(np.float32),
        y=points[:, 1].astype(np.float32),
        z=points[:, 2].astype(np.float32),
        rgb255=colors_255.astype(np.uint8),
    )

    # CSV: x,z,r,g,b
    xz = points[:, [0, 2]]
    rgb = colors_255.astype(np.uint8)
    csv_mat = np.concatenate([xz, rgb.astype(np.float32)], axis=1)
    header = "x,z,r,g,b"
    np.savetxt(csv_path, csv_mat, fmt=["%.3f", "%.3f", "%d", "%d", "%d"], delimiter=",", header=header, comments="")

    return npz_path, csv_path


def render_scatter_map(
    out_dir: str,
    points: np.ndarray,
    colors_255: np.ndarray,
    target_longer_side: int = 2000,
    margin_px: int = 20,
    dpi: int = 100,
    point_size: float = 0.3,
) -> Tuple[str, Dict[str, float]]:
    """Render a 2D scatter (x vs z) into map.png and return transform params.

    Image pixel mapping:
      width  = 2*margin + (x_max - x_min) * scale
      height = 2*margin + (z_max - z_min) * scale
      px_x   = margin + (x - x_min) * scale
      px_y   = margin + (z_max - z) * scale   # top-left origin (y down)
    """
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt

    x = points[:, 0]
    z = points[:, 2]
    x_min, x_max = float(x.min()), float(x.max())
    z_min, z_max = float(z.min()), float(z.max())
    range_x = max(x_max - x_min, 1e-6)
    range_z = max(z_max - z_min, 1e-6)

    # Determine scale so that longer side fits target_longer_side
    work_w = range_x
    work_h = range_z
    longer_world = max(work_w, work_h)
    scale = (float(target_longer_side) - 2.0 * float(margin_px)) / float(longer_world)
    width = int(round(2 * margin_px + work_w * scale))
    height = int(round(2 * margin_px + work_h * scale))

    # Pixel coordinates (flip z to image y)
    px_x = margin_px + (x - x_min) * scale
    px_y = margin_px + (z_max - z) * scale

    colors_01 = (colors_255.astype(np.float32) / 255.0).clip(0.0, 1.0)

    fig_w_in = width / float(dpi)
    fig_h_in = height / float(dpi)
    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # full canvas
    ax.scatter(px_x, px_y, c=colors_01, s=point_size, linewidths=0, marker=",", rasterized=True)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # y down
    ax.axis("off")

    out_png = os.path.join(out_dir, "map.png")
    fig.savefig(out_png, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)

    transform = {
        "x_min": x_min,
        "x_max": x_max,
        "z_min": z_min,
        "z_max": z_max,
        "scale_px_per_world": float(scale),
        "margin_px": int(margin_px),
        "width": int(width),
        "height": int(height),
        "pixel_from_world": {
            "px_x": "margin_px + (x - x_min) * scale",
            "px_y": "margin_px + (z_max - z) * scale",
            "origin_top_left": True,
        },
        "world_from_pixel": {
            "x": "x_min + (px_x - margin_px) / scale",
            "z": "z_max - (px_y - margin_px) / scale",
        },
        "notes": "Coordinates are in apartment_0 units per points*10000/255."
    }

    return out_png, transform


def maybe_verify_colors_with_excel(out_dir: str, colors_255: np.ndarray, excel_path: str, enable: bool) -> Optional[str]:
    """Optionally verify that colors appear in the allowed set from the Excel.

    Writes a short report to out_dir if performed. Returns path to report or None.
    """
    if not enable:
        return None
    if not os.path.isfile(excel_path):
        return None
    try:
        import pandas as pd  # type: ignore
    except Exception:
        # pandas/openpyxl may not be available; skip quietly
        return None

    try:
        df = pd.read_excel(excel_path)
    except Exception:
        return None

    # Try to infer RGB column names
    cols = {c.lower(): c for c in df.columns}
    r_col = cols.get("r") or cols.get("red") or cols.get("r(0-255)")
    g_col = cols.get("g") or cols.get("green") or cols.get("g(0-255)")
    b_col = cols.get("b") or cols.get("blue") or cols.get("b(0-255)")
    if not (r_col and g_col and b_col):
        return None

    allowed = df[[r_col, g_col, b_col]].dropna().astype(int).to_numpy().astype(np.uint8)
    allowed_codes = set((int(r) << 16) | (int(g) << 8) | int(b) for r, g, b in allowed)

    data = colors_255.reshape(-1, 3)
    data_codes_all = ((data[:, 0].astype(np.int64) << 16) | (data[:, 1].astype(np.int64) << 8) | data[:, 2].astype(np.int64))
    unique_codes, counts = np.unique(data_codes_all, return_counts=True)

    missing_mask = np.array([code not in allowed_codes for code in unique_codes])
    n_missing = int(missing_mask.sum())
    pct_missing = float(100.0 * counts[missing_mask].sum() / max(1, data.shape[0]))

    report = {
        "unique_colors": int(unique_codes.size),
        "unique_missing": int(n_missing),
        "pct_points_not_in_excel_palette": pct_missing,
        "note": "If >0 due to rounding, using color0255.npy should avoid issues.",
    }

    report_path = os.path.join(out_dir, "color_verification.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report_path


def save_transform(out_dir: str, transform: Dict, filter_info: Dict[str, float]) -> str:
    """Save transform and filtering metadata to JSON."""
    tf = {
        "point_to_apartment_scale": 10000.0 / 255.0,
        "filter": filter_info,
        "transform": transform,
    }
    path = os.path.join(out_dir, "map_transform.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tf, f, indent=2)
    return path


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a 2D semantic map from 3D point cloud")
    p.add_argument("--data-dir", default="semantic_3d_pointcloud", help="Directory containing point.npy and color npy files")
    p.add_argument("--out-dir", default=".", help="Directory to write outputs")

    # Filtering
    p.add_argument("--floor-frac", type=float, default=0.01, help="Bottom fraction of Y to remove (ignored if --floor-abs is set)")
    p.add_argument("--ceiling-frac", type=float, default=0.01, help="Top fraction of Y to remove (ignored if --ceiling-abs is set)")
    p.add_argument("--floor-abs", type=float, default=None, help="Remove y <= y_min + this absolute margin (apartment units)")
    p.add_argument("--ceiling-abs", type=float, default=None, help="Remove y >= y_max - this absolute margin (apartment units)")

    # Rendering
    p.add_argument("--target-longer-side", type=int, default=2000, help="Target pixel size for the longer image side")
    p.add_argument("--margin-px", type=int, default=20, help="Image pixel margin around content")
    p.add_argument("--dpi", type=int, default=100, help="Matplotlib DPI for figure sizing")
    p.add_argument("--point-size", type=float, default=0.3, help="Scatter point size (pt^2)")

    # Verification
    p.add_argument("--verify-colors-with-excel", action="store_true", help="Verify colors against Excel palette if dependencies available")
    p.add_argument("--excel-path", default="color_coding_semantic_segmentation_classes.xlsx", help="Path to Excel palette for optional verification")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    points_apartment, colors_255 = load_points_and_colors(args.data_dir)
    points_f, colors_f, filter_info = filter_floor_ceiling(
        points_apartment,
        colors_255,
        floor_frac=args.floor_frac,
        ceiling_frac=args.ceiling_frac,
        floor_abs=args.floor_abs,
        ceiling_abs=args.ceiling_abs,
    )

    # Save filtered points/colors
    npz_path, csv_path = save_points(args.out_dir, points_f, colors_f)

    # Render map and write transform
    out_png, transform = render_scatter_map(
        args.out_dir,
        points_f,
        colors_f,
        target_longer_side=args.target_longer_side,
        margin_px=args.margin_px,
        dpi=args.dpi,
        point_size=args.point_size,
    )
    tf_path = save_transform(args.out_dir, transform, filter_info)

    # Optional verification of color palette
    report_path = maybe_verify_colors_with_excel(args.out_dir, colors_f, args.excel_path, args.verify_colors_with_excel)

    # Console summary
    print("Wrote:")
    print(f"  {out_png}")
    print(f"  {npz_path}")
    print(f"  {csv_path}")
    print(f"  {tf_path}")
    if report_path:
        print(f"  {report_path}")
    print("Filter info:")
    for k, v in filter_info.items():
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

