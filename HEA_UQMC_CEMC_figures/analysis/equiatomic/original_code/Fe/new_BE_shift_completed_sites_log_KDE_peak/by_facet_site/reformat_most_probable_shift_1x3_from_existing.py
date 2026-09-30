#!/usr/bin/env python3
"""Reformat existing shifted 1x3 figures to match the parent no-shift style.

The original per-site structure inputs are no longer present.  This script
digitizes the already-rendered shifted curves and constrains every recovered
curve to the mean and integrated contribution stored in the analysis summary
CSVs before drawing it with the publication formatting.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(
    "/home/jinsookim/HEA_MC/FeCoNiPtPd_HER/equi-atomic/"
    "new_BE_shift_completed_sites_log_KDE_peak/by_facet_site"
)
ANALYSIS = "hbe_distribution_analysis_1x3"
DESTINATION = "most_probable_trial_be_shift_1x3"
CASES = (
    "fcc100_bridge",
    "fcc100_hollow",
    "fcc110_bridge",
    "fcc111_hollow",
    "fcc111_top",
)
METHODS = ("Random", "CEMC", "CEMC + layer shuffle")
DISPLAY_TITLES = {
    "Random": "Homogeneous",
    "CEMC": "CEMC",
    "CEMC + layer shuffle": "CEMC+layer shuffled",
}
ELEMENTS = ("Fe", "Co", "Ni", "Pd", "Pt")

# Colors in the existing shifted raster figures (seaborn deep palette).
SOURCE_COLORS = {
    "Fe": np.array((196, 78, 82)),
    "Co": np.array((76, 114, 176)),
    "Ni": np.array((85, 168, 104)),
    "Pd": np.array((129, 114, 179)),
    "Pt": np.array((204, 185, 116)),
}
# Requested publication colors, identical to the parent no-shift 1x3 figures.
OUTPUT_COLORS = {
    "Fe": "#E31A1C",
    "Co": "#FF9900",
    "Ni": "#2CA02C",
    "Pd": "#FFD700",
    "Pt": "#1F77B4",
}
ACTIVITY_POSITIONS = {
    "fcc100_bridge": {
        "Random": (0.96, 0.68, "right"),
        "CEMC": (0.04, 0.95, "left"),
        "CEMC + layer shuffle": (0.04, 0.95, "left"),
    },
    "fcc100_hollow": {
        "Random": (0.04, 0.68, "left"),
        "CEMC": (0.04, 0.95, "left"),
        "CEMC + layer shuffle": (0.04, 0.95, "left"),
    },
    "fcc110_bridge": {
        method: (0.96, 0.95, "right") for method in METHODS
    },
    "fcc111_hollow": {
        "Random": (0.96, 0.95, "right"),
        "CEMC": (0.04, 0.95, "left"),
        "CEMC + layer shuffle": (0.04, 0.95, "left"),
    },
    "fcc111_top": {
        "Random": (0.96, 0.95, "right"),
        "CEMC": (0.04, 0.95, "left"),
        "CEMC + layer shuffle": (0.04, 0.95, "left"),
    },
}


def collapse_runs(indices: np.ndarray) -> list[float]:
    if len(indices) == 0:
        return []
    groups: list[list[int]] = [[int(indices[0])]]
    for value in indices[1:]:
        value = int(value)
        if value <= groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [float(np.mean(group)) for group in groups]


def detect_axes(rgb: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[float], float]:
    """Return three axes boxes, x=0 pixel positions, and pixels per 0.25 eV."""
    height, width, _ = rgb.shape
    dark = np.max(rgb, axis=2) < 80

    horizontal = np.flatnonzero(dark.sum(axis=1) > 0.72 * width)
    horizontal_centers = collapse_runs(horizontal)
    if len(horizontal_centers) < 2:
        raise RuntimeError("Could not detect horizontal axes spines")
    y_top = int(round(horizontal_centers[0]))
    y_bottom = int(round(horizontal_centers[-1]))

    inner_dark_counts = dark[y_top : y_bottom + 1].sum(axis=0)
    vertical = np.flatnonzero(inner_dark_counts > 0.88 * (y_bottom - y_top + 1))
    vertical_centers = collapse_runs(vertical)
    if len(vertical_centers) != 6:
        raise RuntimeError(f"Expected six vertical spines, found {vertical_centers}")
    bounds = [int(round(value)) for value in vertical_centers]
    boxes = [
        (bounds[0], bounds[1], y_top, y_bottom),
        (bounds[2], bounds[3], y_top, y_bottom),
        (bounds[4], bounds[5], y_top, y_bottom),
    ]

    x_zero: list[float] = []
    grid_spacings: list[float] = []
    for x_left, x_right, _, _ in boxes:
        interior = dark[y_top + 3 : y_bottom - 2, x_left + 3 : x_right - 2]
        x0_local = int(np.argmax(interior.sum(axis=0)))
        x_zero.append(float(x_left + 3 + x0_local))

        # The old figure uses seaborn's RGB(239,239,239) major grid.  Detect
        # its vertical lines and use their median distance (0.25 eV).
        panel = rgb[y_top + 4 : y_bottom - 3, x_left + 2 : x_right - 1]
        is_grid = np.all(panel == 239, axis=2)
        candidates = np.flatnonzero(is_grid.sum(axis=0) > 0.72 * panel.shape[0])
        centers = collapse_runs(candidates)
        if len(centers) < 3:
            raise RuntimeError("Could not detect source x grid spacing")
        diffs = np.diff(centers)
        grid_spacings.append(float(np.median(diffs[(diffs > 0.08 * (x_right - x_left))])))

    return boxes, x_zero, float(np.median(grid_spacings))


def detect_zero_density_row(rgb: np.ndarray, box: tuple[int, int, int, int]) -> float:
    x_left, x_right, y_top, y_bottom = box
    panel = rgb[y_top + 2 : y_bottom - 1, x_left + 3 : x_right - 2]
    is_grid = np.all(panel == 239, axis=2)
    candidates = np.flatnonzero(is_grid.sum(axis=1) > 0.72 * panel.shape[1])
    centers = collapse_runs(candidates)
    if len(centers) >= 2:
        spacing = float(np.median(np.diff(centers)))
        # In the old plots the lowest visible grid line is one tick above y=0;
        # the zero-density line is hidden by the five colored zero tails.
        return float(y_top + 2 + centers[-1] + spacing)

    # Some individual-temperature panels have too few y ticks for a spacing
    # estimate.  Pt is drawn last, so its near-zero tails remain visible over
    # the widest horizontal span and give a reliable zero-density raster row.
    distance = np.sqrt(
        np.sum((panel.astype(float) - SOURCE_COLORS["Pt"].astype(float)) ** 2, axis=2)
    )
    row_counts = (distance < 24.0).sum(axis=1)
    strong_rows = np.flatnonzero(row_counts > 0.45 * row_counts.max())
    if len(strong_rows) == 0:
        raise RuntimeError("Could not detect source zero-density row")
    return float(y_top + 2 + strong_rows[-1])


def recover_curve(
    rgb: np.ndarray,
    box: tuple[int, int, int, int],
    x_zero: float,
    pixels_per_quarter_ev: float,
    y_zero: float,
    source_color: np.ndarray,
    target_mean: float,
    target_sd: float,
    target_area: float,
    bridge_all_gaps: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    x_left, x_right, y_top, y_bottom = box
    panel = rgb[y_top : y_bottom + 1, x_left : x_right + 1].astype(float)
    distance = np.sqrt(np.sum((panel - source_color.astype(float)) ** 2, axis=2))
    mask = distance < 24.0

    local_y = np.full(mask.shape[1], np.nan)
    for column in range(mask.shape[1]):
        rows = np.flatnonzero(mask[:, column])
        if len(rows):
            local_y[column] = float(np.median(rows) + y_top)
    height = np.maximum(y_zero - local_y, 0.0)
    visible = np.flatnonzero(np.isfinite(height) & (height > 0.75))
    x = ((np.arange(mask.shape[1]) + x_left) - x_zero) * (
        0.25 / pixels_per_quarter_ev
    )
    if target_area <= 0 or not np.isfinite(target_mean):
        return x, np.zeros_like(x), target_mean
    if len(visible) < 6:
        # An extremely small contribution can be completely covered by curves
        # drawn later in the source raster.  In that rare case use the exact
        # first two stored moments and area from the summary CSV.
        sigma = target_sd if np.isfinite(target_sd) and target_sd > 0 else 0.02
        density = target_area * np.exp(-0.5 * ((x - target_mean) / sigma) ** 2)
        density /= sigma * np.sqrt(2.0 * np.pi)
        return x, density, target_mean

    if bridge_all_gaps:
        # For all-temperature averages the source KDE curves are continuous.
        # Long missing raster segments are caused by later-drawn, overlapping
        # element curves—not by a physical zero in the distribution.
        recovered = np.interp(np.arange(mask.shape[1]), visible, height[visible])
        recovered[: visible[0]] = 0.0
        recovered[visible[-1] + 1 :] = 0.0
        radius = 9
        offsets = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (offsets / 3.0) ** 2)
        kernel /= kernel.sum()
        recovered = np.convolve(recovered, kernel, mode="same")
    else:
        recovered = np.zeros(mask.shape[1], dtype=float)
        recovered[visible] = height[visible]
        # Fill only short interruptions caused by curve crossings/dashed reference.
        for left, right in zip(visible[:-1], visible[1:]):
            if 1 < right - left <= 22:
                recovered[left : right + 1] = np.linspace(
                    recovered[left], recovered[right], right - left + 1
                )
        recovered = np.convolve(recovered, np.array([0.2, 0.6, 0.2]), mode="same")

    raw_area = float(np.trapezoid(recovered, x))
    if raw_area <= 0:
        raise RuntimeError("Recovered curve has zero area")
    raw_mean = float(np.trapezoid(x * recovered, x) / raw_area)
    x = x + (target_mean - raw_mean)
    density = recovered * (target_area / raw_area)
    return x, density, raw_mean


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric(value: str) -> float:
    return float(value) if value is not None and value.strip() else float("nan")


def format_sig_figs(value: float, digits: int = 2) -> str:
    """Format with exactly ``digits`` significant figures, including zeros."""
    if value == 0:
        return f"{value:.{digits - 1}f}"
    decimal_places = digits - int(np.floor(np.log10(abs(value)))) - 1
    if decimal_places > 0:
        return f"{value:.{decimal_places}f}"
    return f"{value:.0f}"


def summary_lookup(rows: list[dict[str, str]], temperature: int | None) -> dict[tuple[str, str], dict[str, str]]:
    if temperature is not None:
        rows = [row for row in rows if int(float(row["temperature_K"])) == temperature]
    return {(row["method"], row["element"]): row for row in rows}


def temperature_from_name(name: str) -> int | None:
    if name.startswith("all_temperatures_average"):
        return None
    if name.startswith("final_0298K"):
        return 298
    if name.startswith("T_"):
        return int(name[2:6])
    raise ValueError(name)


def draw_one(
    case: str,
    source: Path,
    destination: Path,
    lookup: dict,
    bridge_all_gaps: bool = False,
) -> list[dict[str, object]]:
    rgb = np.array(Image.open(source).convert("RGB"))
    boxes, zero_pixels, grid_spacing = detect_axes(rgb)
    zero_rows = [detect_zero_density_row(rgb, box) for box in boxes]

    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    audit: list[dict[str, object]] = []
    x_limits: list[tuple[float, float]] = []
    for method, box, x_zero, y_zero in zip(METHODS, boxes, zero_pixels, zero_rows):
        x_left, x_right, _, _ = box
        scale = 0.25 / grid_spacing
        x_limits.append(((x_left - x_zero) * scale, (x_right - x_zero) * scale))
        for element in ELEMENTS:
            row = lookup[method, element]
            target_mean = numeric(row["weighted_mean_deltaG_H_eV"])
            target_sd = numeric(row["weighted_sd_deltaG_H_eV"])
            target_area = numeric(row["integrated_fractional_contribution"])
            x, density, raw_mean = recover_curve(
                rgb,
                box,
                x_zero,
                grid_spacing,
                y_zero,
                SOURCE_COLORS[element],
                target_mean,
                target_sd,
                target_area,
                bridge_all_gaps=bridge_all_gaps,
            )
            curves[method, element] = (x, density)
            audit.append(
                {
                    "source_png": source.name,
                    "method": method,
                    "element": element,
                    "summary_mean_eV": target_mean,
                    "digitized_mean_before_constraint_eV": raw_mean,
                    "mean_correction_eV": target_mean - raw_mean,
                    "constrained_integrated_fraction": target_area,
                }
            )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.titlesize": 21,
            "axes.labelsize": 20,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "axes.linewidth": 1.2,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(18.9, 6.4), sharex=True, sharey=True)
    common_left = min(value[0] for value in x_limits)
    common_right = max(value[1] for value in x_limits)
    for method, ax in zip(METHODS, axes):
        for element in ELEMENTS:
            x, density = curves[method, element]
            ax.plot(x, density, lw=2.7, color=OUTPUT_COLORS[element], label=element)
        ax.axvline(0.0, ls="--", lw=2.1, color="black")
        ax.set_xlim(common_left, common_right)
        ax.set_title(DISPLAY_TITLES[method], pad=10)
        ax.set_xlabel(r"$\Delta G_{\mathrm{H}}$ (eV)")
        ax.tick_params(width=1.2, length=5)
        ax.grid(False)
        row = lookup[method, ELEMENTS[0]]
        activity = float(row["activity_no_log"])
        activity_x, activity_y, alignment = ACTIVITY_POSITIONS[case][method]
        ax.text(
            activity_x,
            activity_y,
            f"Activity = {format_sig_figs(activity)}",
            transform=ax.transAxes,
            ha=alignment,
            va="top",
            fontsize=24,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=3.0),
            zorder=90,
        )

    axes[0].set_ylabel("Probability density")
    handles, labels = axes[0].get_legend_handles_labels()
    legend = axes[0].legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=2,
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        columnspacing=1.4,
        handlelength=2.4,
        fontsize=18,
    )
    legend.set_zorder(100)
    fig.tight_layout()
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return audit


def process_case(
    case: str,
    all_average_only: bool = False,
    bridge_all_gaps: bool = False,
) -> None:
    analysis_dir = ROOT / case / ANALYSIS
    plots_dir = analysis_dir / "plots_by_temperature"
    destination_dir = plots_dir / DESTINATION
    destination_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_rows(analysis_dir / "all_temperatures_average_element_deltaG_H_summary.csv")
    temperature_rows = read_rows(analysis_dir / "element_HBE_distribution_summary.csv")
    source_files = [plots_dir / "all_temperatures_average_element_deltaG_H_distribution.png"]
    if not all_average_only:
        source_files += [plots_dir / "final_0298K_element_HBE_distribution.png"]
        source_files += [plots_dir / f"T_{temperature:04d}K_element_HBE_distribution.png" for temperature in range(300, 2001, 100)]

    audit_rows: list[dict[str, object]] = []
    for source in source_files:
        temperature = temperature_from_name(source.name)
        rows = all_rows if temperature is None else temperature_rows
        lookup = summary_lookup(rows, temperature)
        if len(lookup) != 15:
            raise RuntimeError(f"Incomplete summary lookup for {case}, {source.name}: {len(lookup)}")
        destination = destination_dir / source.name
        audit_rows.extend(
            draw_one(
                case,
                source,
                destination,
                lookup,
                bridge_all_gaps=bridge_all_gaps,
            )
        )
        print(f"DONE {case}: {destination.name}", flush=True)

    audit_name = (
        "all_temperatures_curve_digitization_constraint_audit.csv"
        if all_average_only
        else "curve_digitization_constraint_audit.csv"
    )
    audit_path = destination_dir / audit_name
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)

    readme = destination_dir / "README.txt"
    previous = readme.read_text(encoding="utf-8") if readme.exists() else ""
    readme.write_text(
        previous.rstrip()
        + "\n\nFormatting update:\n"
        + "All figures were redrawn to exactly follow the parent no-BE-shift 1x3 style.\n"
        + "Existing shifted curves were digitized because the original structure inputs are absent.\n"
        + "Each curve was constrained to its stored weighted mean and integrated elemental contribution.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--all-average-only", action="store_true")
    parser.add_argument("--bridge-all-gaps", action="store_true")
    args = parser.parse_args()
    for case in CASES:
        if args.case and case != args.case:
            continue
        process_case(
            case,
            all_average_only=args.all_average_only,
            bridge_all_gaps=args.bridge_all_gaps,
        )


if __name__ == "__main__":
    main()
