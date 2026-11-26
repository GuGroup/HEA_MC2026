#!/usr/bin/env python3
"""Extract ML1-ML6 experimental ORR activity values from the published PNG.

The dot positions follow the same 20 x 21 masked grid used by
get_compositions_from_png.py and ML*_composition_data.csv.
For each ML, this script samples the dot color at each Row_Idx/Col_Idx and
maps that color back to a numeric activity by matching it to the ML-specific
colorbar in PtPdIrRhRu_ORR_ML1-6.png. Each output CSV contains the
linearly scaled activity magnitudes for that microlibrary.
"""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import pandas as pd
from PIL import Image

DATA_DIR = Path(__file__).resolve().parent
IMAGE_PATH = DATA_DIR / "PtPdIrRhRu_ORR_ML1-6.png"

# Activity limits extrapolated from the tick-label spacing in
# PtPdIrRhRu_ORR_ML1-6.png. Values are (bottom/yellow, top/purple), evaluated
# at the first and last colored rows of each colorbar.
ACTIVITY_LIMITS = {
    1: (-0.303111, -0.081807),
    2: (-0.287784, -0.102855),
    3: (-0.212517, -0.082173),
    4: (-0.474854, -0.086927),
    5: (-0.190812, -0.068204),
    6: (-0.129001, -0.073389),
}

# Approximate panel regions containing the circular dot grids in the 1103 x 641 PNG.
# The exact 20 x 21 dot centers are inferred from colored pixels inside these boxes.
PANEL_REGIONS = {
    1: (50, 40, 290, 310),
    2: (400, 40, 640, 310),
    3: (750, 40, 990, 310),
    4: (50, 348, 290, 620),
    5: (400, 348, 640, 620),
    6: (750, 348, 990, 620),
}

# Colorbar colored-strip bounding boxes detected from the PNG.
COLORBAR_BOXES = {
    1: (300, 31, 314, 301),
    2: (651, 31, 665, 301),
    3: (1001, 31, 1015, 301),
    4: (300, 339, 314, 609),
    5: (651, 339, 665, 609),
    6: (1001, 339, 1015, 609),
}

N_ROWS = 20
N_COLS = 21
DOT_RADIUS_PX = 4.0
CENTER_EDGE_MARGIN_PX = 3.0


def load_rgb_image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def colored_pixel_mask(rgb: np.ndarray) -> np.ndarray:
    arr = rgb.astype(float) / 255.0
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = (mx - mn) / (mx + 1e-12)
    # Keeps viridis-colored dots/colorbars and removes white background, black text,
    # and the gray circle outline.
    return (sat > 0.18) & (mx < 0.96) & (mn < 0.95)


def infer_grid_centers(mask: np.ndarray, ml_idx: int) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = PANEL_REGIONS[ml_idx]
    panel_mask = mask[y0:y1, x0:x1]
    ys, xs = np.where(panel_mask)
    if len(xs) == 0:
        raise RuntimeError(f"No colored grid pixels found for ML{ml_idx}")

    xmin = xs.min() + x0 + CENTER_EDGE_MARGIN_PX
    xmax = xs.max() + x0 - CENTER_EDGE_MARGIN_PX
    ymin = ys.min() + y0 + CENTER_EDGE_MARGIN_PX
    ymax = ys.max() + y0 - CENTER_EDGE_MARGIN_PX

    x_centers = np.linspace(xmin, xmax, N_COLS)
    y_centers = np.linspace(ymin, ymax, N_ROWS)
    return y_centers, x_centers


def sample_dot_rgb(rgb: np.ndarray, mask: np.ndarray, y: float, x: float) -> np.ndarray:
    height, width = mask.shape
    xi = int(round(x))
    yi = int(round(y))
    r = int(np.ceil(DOT_RADIUS_PX))
    xa = max(0, xi - r)
    xb = min(width, xi + r + 1)
    ya = max(0, yi - r)
    yb = min(height, yi + r + 1)

    yy, xx = np.mgrid[ya:yb, xa:xb]
    disk = (xx - x) ** 2 + (yy - y) ** 2 <= DOT_RADIUS_PX**2
    use = disk & mask[ya:yb, xa:xb]
    pixels = rgb[ya:yb, xa:xb][use]
    if len(pixels) == 0:
        pixels = rgb[ya:yb, xa:xb][disk]
    if len(pixels) == 0:
        raise RuntimeError(f"Could not sample dot color near x={x:.1f}, y={y:.1f}")
    return np.median(pixels.astype(float), axis=0)


def build_colorbar_lookup(
    rgb: np.ndarray, mask: np.ndarray, ml_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = COLORBAR_BOXES[ml_idx]
    # Avoid the black border by using only the inner colored stripe.
    inner = rgb[y0 : y1 + 1, x0 + 4 : x1 - 3]
    inner_mask = mask[y0 : y1 + 1, x0 + 4 : x1 - 3]

    rows = []
    colors = []
    for local_y in range(inner.shape[0]):
        row_mask = inner_mask[local_y]
        if row_mask.sum() < 2:
            continue
        rows.append(y0 + local_y)
        colors.append(np.median(inner[local_y][row_mask].astype(float), axis=0))

    if len(rows) < 20:
        raise RuntimeError(f"Too few colorbar rows found for ML{ml_idx}: {len(rows)}")

    rows = np.asarray(rows, dtype=float)
    colors = np.asarray(colors, dtype=float)
    bottom_value, top_value = ACTIVITY_LIMITS[ml_idx]
    frac_from_top = (rows - rows.min()) / (rows.max() - rows.min())
    values = top_value + frac_from_top * (bottom_value - top_value)
    return colors, values


def rgb_to_activity(
    dot_rgb: np.ndarray, colorbar_colors: np.ndarray, colorbar_values: np.ndarray
) -> float:
    distances = np.linalg.norm(colorbar_colors - dot_rgb[None, :], axis=1)
    return float(colorbar_values[int(np.argmin(distances))])


def extract_ml(rgb: np.ndarray, mask: np.ndarray, ml_idx: int) -> pd.DataFrame:
    comp_path = DATA_DIR / f"ML{ml_idx}_composition_data.csv"
    comp = pd.read_csv(comp_path)
    required = {"Row_Idx", "Col_Idx"}
    missing = required - set(comp.columns)
    if missing:
        raise RuntimeError(f"{comp_path} is missing columns: {sorted(missing)}")

    y_centers, x_centers = infer_grid_centers(mask, ml_idx)
    colorbar_colors, colorbar_values = build_colorbar_lookup(rgb, mask, ml_idx)

    records = []
    for _, row in comp.iterrows():
        row_idx = int(row["Row_Idx"])
        col_idx = int(row["Col_Idx"])
        if not (0 <= row_idx < N_ROWS and 0 <= col_idx < N_COLS):
            raise RuntimeError(
                f"Invalid grid index in ML{ml_idx}: "
                f"Row_Idx={row_idx}, Col_Idx={col_idx}"
            )
        dot_rgb = sample_dot_rgb(rgb, mask, y_centers[row_idx], x_centers[col_idx])
        activity = rgb_to_activity(dot_rgb, colorbar_colors, colorbar_values)
        records.append({"Row_Idx": row_idx, "Col_Idx": col_idx, "Activity": activity})

    out = pd.DataFrame.from_records(records, columns=["Row_Idx", "Col_Idx", "Activity"])
    if len(out) != len(comp):
        raise RuntimeError(f"ML{ml_idx}: expected {len(comp)} rows, got {len(out)}")
    return out


def main() -> None:
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(IMAGE_PATH)

    rgb = load_rgb_image(IMAGE_PATH)
    if rgb.shape[:2] != (641, 1103):
        print(f"Warning: image shape is {rgb.shape[:2]}, expected (641, 1103).")
        print(
            "The script will still try to infer dot centers from the configured panel boxes."
        )

    mask = colored_pixel_mask(rgb)
    datasets = {ml: extract_ml(rgb, mask, ml) for ml in range(1, 7)}
    # Preserve the original intermediate CSV precision and common scale before
    # normalizing each ML, so the distributed eight-decimal values reproduce.
    magnitudes = {
        ml: data["Activity"].map(lambda value: float(f"{value:.8f}")).abs()
        for ml, data in datasets.items()
    }
    global_min = min(values.min() for values in magnitudes.values())
    global_max = max(values.max() for values in magnitudes.values())
    if global_max <= global_min:
        raise ValueError("Distinct activity magnitudes are required for scaling")
    for ml_idx, out in datasets.items():
        magnitude = ((magnitudes[ml_idx] - global_min) / (global_max - global_min)).round(8)
        minimum, maximum = magnitude.min(), magnitude.max()
        if not maximum > minimum:
            raise ValueError(f"ML{ml_idx}: distinct activity magnitudes are required for scaling")
        out["Activity"] = -(magnitude - minimum) / (maximum - minimum)
        out_path = DATA_DIR / f"ML{ml_idx}_activity_data.csv"
        out.to_csv(
            out_path,
            index=False,
            quoting=csv.QUOTE_MINIMAL,
            float_format="%.8f",
        )
        print(
            f"ML{ml_idx}: wrote {out_path.name} "
            f"({len(out)} rows, Activity range "
            f"{out['Activity'].min():.5f} to {out['Activity'].max():.5f})"
        )


if __name__ == "__main__":
    main()
