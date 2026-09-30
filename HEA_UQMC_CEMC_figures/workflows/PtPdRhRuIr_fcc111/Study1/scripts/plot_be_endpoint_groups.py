#!/usr/bin/env python3
"""Compare 300/2000 K BE distributions for endpoint-best trial groups."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ELEMENTS = ["Ir", "Pd", "Pt", "Rh", "Ru"]
COLORS = {300.0: "#2563eb", 2000.0: "#dc2626"}
ELEMENT_COLORS = {
    "Ir": "#4C78A8",
    "Pd": "#F58518",
    "Pt": "#54A24B",
    "Rh": "#B279A2",
    "Ru": "#E45756",
}


def parse_temperatures(text: str) -> list[float]:
    return [float(value.strip()) for value in text.split(",") if value.strip()]


def find_trial_histogram(results_root: Path, trial: int, world_size: int) -> Path | None:
    name = f"trial_{trial:06d}_be_histogram_counts.csv"
    direct = results_root / "be_histograms" / name
    if direct.exists():
        return direct
    for idx in range(world_size):
        candidate = results_root / f"shard_{idx:02d}" / "be_histograms" / name
        if candidate.exists():
            return candidate
    return None


def frequency(frame: pd.DataFrame) -> pd.Series:
    count = frame["count"].astype(float)
    keys = ["group_best_temperature", "snapshot_temperature"]
    denominator = frame.groupby(keys, dropna=False)["count"].transform("sum").astype(float)
    return count / denominator.where(denominator > 0)


def distribution_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    """Return count-weighted mean and population standard deviation per distribution."""
    keys = ["group_best_temperature", "snapshot_temperature", "element"]
    rows: list[dict[str, float | str]] = []
    for key, distribution in frame.groupby(keys, sort=True):
        weights = distribution["count"].to_numpy(dtype=float)
        centers = distribution["bin_center"].to_numpy(dtype=float)
        total = float(weights.sum())
        if total <= 0:
            mu = sigma = np.nan
        else:
            mu = float(np.average(centers, weights=weights))
            sigma = float(np.sqrt(np.average((centers - mu) ** 2, weights=weights)))
        rows.append(
            {
                **dict(zip(keys, key)),
                "mu": mu,
                "sigma": sigma,
                "total_count": total,
            }
        )
    return pd.DataFrame(rows)


def count_trial_histograms(results_root: Path, world_size: int) -> int:
    return sum(
        1
        for idx in range(world_size)
        for _ in (results_root / f"shard_{idx:02d}" / "be_histograms").glob(
            "trial_*_be_histogram_counts.csv"
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--world-size", type=int, default=5)
    parser.add_argument("--selection-csv", default=None)
    parser.add_argument("--group-temperatures", default="300,2000")
    parser.add_argument("--snapshot-temperatures", default="300,2000")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    selection_path = Path(args.selection_csv) if args.selection_csv else (
        results_root / "crps_selection" / "best_temperature_by_trial_crps.csv"
    )
    output_dir = Path(args.output_dir) if args.output_dir else results_root / "be_endpoint_group_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not selection_path.exists():
        raise SystemExit(
            f"Selection file not found: {selection_path}\n"
            "Run scripts/select_best_score.py first; it also works on currently completed trials."
        )

    groups = parse_temperatures(args.group_temperatures)
    snapshots = parse_temperatures(args.snapshot_temperatures)
    selection = pd.read_csv(selection_path)
    required = {"trial", "temperature"}
    if not required.issubset(selection.columns):
        raise SystemExit(f"Selection CSV must contain {sorted(required)}")
    if args.selection_csv is None:
        available_histograms = count_trial_histograms(results_root, args.world_size)
        selected_trials = selection["trial"].nunique()
        if available_histograms and selected_trials < 0.9 * available_histograms:
            raise SystemExit(
                "The default CRPS selection file is stale: "
                f"it contains {selected_trials} trials, while "
                f"{available_histograms} per-trial BE histograms are available.\n"
                "Refresh it first with:\n"
                f"  python scripts/select_best_score.py --results-root "
                f"{results_root} --world-size {args.world_size}"
            )
    selection = selection[selection["temperature"].isin(groups)].copy()
    if selection.empty:
        raise SystemExit(f"No completed trials currently have best temperature in {groups}")

    frames: list[pd.DataFrame] = []
    missing: list[int] = []
    used_trials: dict[float, list[int]] = {temperature: [] for temperature in groups}
    for row in selection.itertuples(index=False):
        trial = int(row.trial)
        group_temperature = float(row.temperature)
        path = find_trial_histogram(results_root, trial, args.world_size)
        if path is None:
            missing.append(trial)
            continue
        frame = pd.read_csv(path)
        frame = frame[(frame["method"] == "cemc") & frame["temperature"].isin(snapshots)].copy()
        if frame.empty:
            missing.append(trial)
            continue
        frame["group_best_temperature"] = group_temperature
        frame["snapshot_temperature"] = frame["temperature"].astype(float)
        frames.append(frame)
        used_trials[group_temperature].append(trial)

    if not frames:
        raise SystemExit("No matching trial BE histogram files were found.")

    raw = pd.concat(frames, ignore_index=True)
    finite = raw.replace([np.inf, -np.inf], np.nan).dropna(subset=["bin_left", "bin_right"])
    aggregated = (
        finite.groupby(
            ["group_best_temperature", "snapshot_temperature", "element", "bin_left", "bin_right"],
            as_index=False,
        )["count"].sum()
    )
    trial_counts = {
        temperature: len(set(trials)) for temperature, trials in used_trials.items()
    }
    aggregated["n_trials"] = aggregated["group_best_temperature"].map(trial_counts)
    aggregated["bin_center"] = (aggregated["bin_left"] + aggregated["bin_right"]) / 2.0
    aggregated["value"] = frequency(aggregated)
    statistics = distribution_statistics(aggregated)
    statistics_path = output_dir / "be_distribution_mu_sigma.csv"
    statistics.to_csv(statistics_path, index=False)

    print("Distribution statistics (count-weighted histogram bin centers):")
    for row in statistics.itertuples(index=False):
        print(
            f"  Best {row.group_best_temperature:g} K | "
            f"snapshot {row.snapshot_temperature:g} K | {row.element}: "
            f"mu={row.mu:.6g} eV, sigma={row.sigma:.6g} eV"
        )

    # Use identical axes for every endpoint panel and both best-temperature
    # groups so visual differences are directly comparable.
    x_min = float(aggregated["bin_left"].min())
    x_max = float(aggregated["bin_right"].max())
    x_pad = max((x_max - x_min) * 0.02, 0.02)
    common_xlim = (x_min - x_pad, x_max + x_pad)
    y_max = float(aggregated["value"].max())
    common_ylim = (0.0, y_max * 1.08 if y_max > 0 else 1.0)

    for group_temperature in groups:
        group = aggregated[aggregated["group_best_temperature"] == group_temperature]
        if group.empty:
            continue

        # Example-style layout: one panel per snapshot temperature, with all
        # five element distributions overlaid and filled by element color.
        fig, axes = plt.subplots(
            1, len(snapshots), figsize=(11, 4.3), sharex=True, sharey=True,
            gridspec_kw={"wspace": 0.05},
        )
        if len(snapshots) == 1:
            axes = [axes]
        for ax, snapshot_temperature in zip(axes, snapshots):
            for element in ELEMENTS:
                subset = group[
                    (group["element"] == element)
                    & (group["snapshot_temperature"] == snapshot_temperature)
                ].sort_values("bin_center")
                if subset.empty:
                    continue
                color = ELEMENT_COLORS[element]
                ax.fill_between(
                    subset["bin_center"], subset["value"],
                    color=color, alpha=0.38, linewidth=0,
                )
                ax.plot(
                    subset["bin_center"], subset["value"],
                    color=color, alpha=0.80, linewidth=0.9, label=element,
                )
            ax.set_title(f"{snapshot_temperature:g} K snapshot")
            ax.set_xlabel("E_OH_used / BE (eV)")
            ax.grid(axis="y", alpha=0.25)
            ax.set_xlim(*common_xlim)
            ax.set_ylim(*common_ylim)
        axes[0].set_ylabel("frequency")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels, loc="upper center", ncol=len(ELEMENTS),
                frameon=False, bbox_to_anchor=(0.5, 0.915),
                borderaxespad=0.0, columnspacing=1.1, handletextpad=0.4,
            )
        n_trials = trial_counts.get(group_temperature, 0)
        fig.suptitle(
            f"Trials with CRPS best temperature = {group_temperature:g} K (n={n_trials})",
            y=0.985,
        )
        fig.subplots_adjust(
            left=0.075, right=0.995, bottom=0.13, top=0.80, wspace=0.05,
        )
        overview_name = (
            f"be_distribution_best_{group_temperature:g}K_"
            "300K_2000K_element_overlay_frequency.png"
        )
        fig.savefig(output_dir / overview_name, dpi=220, bbox_inches="tight")
        plt.close(fig)

    if missing:
        print(f"Skipped {len(set(missing))} trials without complete BE histogram files.")
    for temperature in groups:
        print(f"Best {temperature:g} K group: {trial_counts.get(temperature, 0)} trials")
    print(f"Wrote distribution mu/sigma values to {statistics_path}")
    print(f"Wrote overlay-frequency PNG files to {output_dir}")


if __name__ == "__main__":
    main()
