#!/usr/bin/env python3
"""Plot BE histogram distributions accumulated by uq_cemc_mpi."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import pandas as pd


ELEMENT_ORDER = ["Ir", "Pd", "Pt", "Rh", "Ru"]
ELEMENT_COLORS = {
    "Ir": "#4C78A8",
    "Pd": "#F58518",
    "Pt": "#54A24B",
    "Rh": "#B279A2",
    "Ru": "#E45756",
}


def find_histogram_csvs(results_root: Path, world_size: int | None) -> list[Path]:
    if world_size is None:
        paths = sorted(results_root.glob("shard_*/be_histograms/be_histogram_counts.csv"))
    else:
        paths = [
            results_root / f"shard_{idx:02d}" / "be_histograms" / "be_histogram_counts.csv"
            for idx in range(world_size)
        ]
    return [p for p in paths if p.exists() and p.stat().st_size > 0]


def read_histograms(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        if df.empty:
            continue
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        raise SystemExit("No non-empty BE histogram CSV files were found.")

    df = pd.concat(frames, ignore_index=True)
    required = {"method", "temperature", "element", "bin_left", "bin_right", "count"}
    missing = required.difference(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in BE histogram CSV: {sorted(missing)}")

    for col in ("temperature", "bin_left", "bin_right", "count"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["element"].isin(ELEMENT_ORDER)].copy()
    df = df.dropna(subset=["bin_left", "bin_right", "count"])
    df = df[df["bin_left"].apply(pd.notna) & df["bin_right"].apply(pd.notna)]
    df = df[df["bin_left"].apply(lambda x: pd.notna(x) and x != float("-inf"))]
    df = df[df["bin_right"].apply(lambda x: pd.notna(x) and x != float("inf"))]

    grouped = (
        df.groupby(["method", "temperature", "element", "bin_left", "bin_right"], dropna=False)["count"]
        .sum()
        .reset_index()
    )
    grouped["bin_center"] = (grouped["bin_left"] + grouped["bin_right"]) / 2.0
    grouped["bin_width"] = grouped["bin_right"] - grouped["bin_left"]
    return grouped


def transform_counts(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    out = df.copy()
    out["value"] = out["count"].astype(float)
    if mode == "count":
        return out

    if mode == "frequency":
        keys = ["method", "temperature"]
        denom = out.groupby(keys, dropna=False)["count"].transform("sum").astype(float)
        denom = denom.where(denom > 0)
        out["value"] = out["count"].astype(float) / denom
        return out

    keys = ["method", "temperature", "element"]
    denom = out.groupby(keys, dropna=False)["count"].transform("sum").astype(float)
    denom = denom.where(denom > 0)
    out["value"] = out["count"].astype(float) / denom
    if mode == "density":
        out["value"] = out["value"] / out["bin_width"].astype(float)
    return out


def plot_panel(ax, df: pd.DataFrame, title: str, alpha: float) -> None:
    for element in ELEMENT_ORDER:
        sub = df[df["element"] == element].sort_values("bin_center")
        if sub.empty:
            continue
        width = float(sub["bin_width"].median()) if not sub["bin_width"].empty else 0.01
        ax.bar(
            sub["bin_center"],
            sub["value"],
            width=width,
            color=ELEMENT_COLORS[element],
            alpha=alpha,
            linewidth=0,
            label=element,
            align="center",
        )
    ax.set_title(title)
    ax.set_xlabel("E_OH_used / BE (eV)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)


def gaussian_smooth(values: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return values
    radius = max(1, int(round(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def plot_kde_panel(ax, df: pd.DataFrame, title: str, mode: str, bandwidth: float) -> None:
    if df.empty:
        ax.set_title(title)
        ax.set_xlabel("E_OH_used / BE (eV)")
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)
        return

    bins = (
        df[["bin_left", "bin_right", "bin_center", "bin_width"]]
        .drop_duplicates()
        .sort_values("bin_center")
    )
    centers = bins["bin_center"].to_numpy(float)
    widths = bins["bin_width"].to_numpy(float)
    bin_width = float(np.nanmedian(widths)) if len(widths) else 0.01
    sigma_bins = max(float(bandwidth) / bin_width, 0.0)
    panel_total = float(df["count"].sum())

    for element in ELEMENT_ORDER:
        sub = df[df["element"] == element]
        if sub.empty:
            continue
        counts = (
            bins[["bin_left", "bin_right", "bin_center"]]
            .merge(sub[["bin_left", "bin_right", "count"]], on=["bin_left", "bin_right"], how="left")
            ["count"]
            .fillna(0.0)
            .to_numpy(float)
        )
        y = gaussian_smooth(counts, sigma_bins)
        if mode == "frequency":
            if panel_total > 0:
                y = y / panel_total
        elif mode == "probability":
            total = float(counts.sum())
            if total > 0:
                y = y / total
        elif mode == "density":
            total = float(counts.sum())
            if total > 0:
                y = y / total / bin_width
        ax.plot(centers, y, color=ELEMENT_COLORS[element], linewidth=2.0, label=element)
        ax.fill_between(centers, y, color=ELEMENT_COLORS[element], alpha=0.10, linewidth=0)

    ax.set_title(title)
    ax.set_xlabel("E_OH_used / BE (eV)")
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.7)


def plot_overview(df: pd.DataFrame, outdir: Path, mode: str, alpha: float) -> Path:
    cemc = df[df["method"] == "cemc"].copy()
    if not cemc.empty:
        cemc = (
            cemc.groupby(["element", "bin_left", "bin_right", "bin_center", "bin_width"])["count"]
            .sum()
            .reset_index()
        )
        cemc["method"] = "cemc_all_temperatures"
        cemc["temperature"] = float("nan")

    random = df[df["method"] == "random"].copy()
    plot_df = transform_counts(pd.concat([cemc, random], ignore_index=True), mode)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    plot_panel(
        axes[0],
        plot_df[plot_df["method"] == "cemc_all_temperatures"],
        "CEMC, all target temperatures",
        alpha,
    )
    plot_panel(axes[1], plot_df[plot_df["method"] == "random"], "Random initial slabs", alpha)
    axes[0].set_ylabel(mode)
    axes[1].set_ylabel(mode)

    handles = [Patch(facecolor=ELEMENT_COLORS[e], alpha=alpha, label=e) for e in ELEMENT_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out = outdir / f"be_distribution_overview_{mode}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_kde_overview(
    df: pd.DataFrame, outdir: Path, mode: str, alpha: float, bandwidth: float
) -> Path:
    cemc = df[df["method"] == "cemc"].copy()
    if not cemc.empty:
        cemc = (
            cemc.groupby(["element", "bin_left", "bin_right", "bin_center", "bin_width"])["count"]
            .sum()
            .reset_index()
        )
        cemc["method"] = "cemc_all_temperatures"
        cemc["temperature"] = float("nan")

    random = df[df["method"] == "random"].copy()
    plot_df = pd.concat([cemc, random], ignore_index=True)
    bar_df = transform_counts(plot_df, mode)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    cemc_raw = plot_df[plot_df["method"] == "cemc_all_temperatures"]
    random_raw = plot_df[plot_df["method"] == "random"]
    plot_panel(
        axes[0],
        bar_df[bar_df["method"] == "cemc_all_temperatures"],
        "CEMC, all target temperatures",
        alpha,
    )
    plot_kde_panel(axes[0], cemc_raw, "CEMC, all target temperatures", mode, bandwidth)
    plot_panel(
        axes[1],
        bar_df[bar_df["method"] == "random"],
        "Random initial slabs",
        alpha,
    )
    plot_kde_panel(axes[1], random_raw, "Random initial slabs", mode, bandwidth)
    axes[0].set_ylabel(mode)
    axes[1].set_ylabel(mode)

    handles = [Patch(facecolor=ELEMENT_COLORS[e], alpha=0.35, label=e) for e in ELEMENT_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    out = outdir / f"be_distribution_overview_{mode}_kde.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_cemc_by_temperature(df: pd.DataFrame, outdir: Path, mode: str, alpha: float) -> Path:
    cemc = df[df["method"] == "cemc"].dropna(subset=["temperature"]).copy()
    if cemc.empty:
        raise SystemExit("No CEMC histogram rows are available.")
    cemc = transform_counts(cemc, mode)
    temps = sorted(cemc["temperature"].unique(), reverse=True)

    ncols = 3
    nrows = (len(temps) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows), sharex=True)
    axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    for ax, temp in zip(axes_flat, temps):
        sub = cemc[cemc["temperature"] == temp]
        plot_panel(ax, sub, f"{int(temp)} K", alpha)
        ax.set_ylabel(mode)
    for ax in axes_flat[len(temps):]:
        ax.axis("off")

    handles = [Patch(facecolor=ELEMENT_COLORS[e], alpha=alpha, label=e) for e in ELEMENT_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = outdir / f"be_distribution_cemc_by_temperature_{mode}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_kde_cemc_by_temperature(
    df: pd.DataFrame, outdir: Path, mode: str, alpha: float, bandwidth: float
) -> Path:
    cemc = df[df["method"] == "cemc"].dropna(subset=["temperature"]).copy()
    if cemc.empty:
        raise SystemExit("No CEMC histogram rows are available.")
    temps = sorted(cemc["temperature"].unique(), reverse=True)

    ncols = 3
    nrows = (len(temps) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows), sharex=True)
    axes_flat = list(axes.ravel()) if hasattr(axes, "ravel") else [axes]

    bar_df = transform_counts(cemc, mode)
    for ax, temp in zip(axes_flat, temps):
        sub = cemc[cemc["temperature"] == temp]
        bar_sub = bar_df[bar_df["temperature"] == temp]
        plot_panel(ax, bar_sub, f"{int(temp)} K", alpha)
        plot_kde_panel(ax, sub, f"{int(temp)} K", mode, bandwidth)
        ax.set_ylabel(mode)
    for ax in axes_flat[len(temps):]:
        ax.axis("off")

    handles = [Patch(facecolor=ELEMENT_COLORS[e], alpha=0.35, label=e) for e in ELEMENT_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = outdir / f"be_distribution_cemc_by_temperature_{mode}_kde.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def write_combined_csv(df: pd.DataFrame, outdir: Path) -> Path:
    out = outdir / "be_histogram_counts_combined.csv"
    cols = ["method", "temperature", "element", "bin_left", "bin_right", "count"]
    df[cols].to_csv(out, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("results/be_distribution_plots"))
    parser.add_argument(
        "--mode",
        choices=("count", "frequency", "probability", "density"),
        default="probability",
        help=(
            "Y-axis scaling. frequency normalizes all elements together within each "
            "method/temp panel; probability normalizes each element separately."
        ),
    )
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--by-temperature", action="store_true")
    parser.add_argument("--kde", action="store_true", help="Also draw smoothed curve plots from histogram counts.")
    parser.add_argument("--kde-bandwidth", type=float, default=0.05, help="Gaussian smoothing bandwidth in eV.")
    args = parser.parse_args()

    paths = find_histogram_csvs(args.results_root, args.world_size)
    if not paths:
        raise SystemExit(f"No BE histogram CSVs found under {args.results_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = read_histograms(paths)
    combined_csv = write_combined_csv(df, args.output_dir)
    overview = plot_overview(df, args.output_dir, args.mode, args.alpha)
    outputs = [combined_csv, overview]
    if args.by_temperature:
        outputs.append(plot_cemc_by_temperature(df, args.output_dir, args.mode, args.alpha))
    if args.kde:
        outputs.append(plot_kde_overview(df, args.output_dir, args.mode, args.alpha, args.kde_bandwidth))
        if args.by_temperature:
            outputs.append(
                plot_kde_cemc_by_temperature(
                    df, args.output_dir, args.mode, args.alpha, args.kde_bandwidth
                )
            )

    print("Read:")
    for path in paths:
        print(f"  {path}")
    print("Wrote:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
