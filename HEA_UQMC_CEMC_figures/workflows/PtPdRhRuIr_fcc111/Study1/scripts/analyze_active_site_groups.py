#!/usr/bin/env python3
"""Relate trial-level shifted-zone1 active-site regimes to UQMC metrics."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ELEMENTS = ("Ir", "Pd", "Pt", "Rh", "Ru")
PDPT = {"Pd", "Pt"}
SHIFT_COLUMNS = [f"be_shift_{element}" for element in ELEMENTS]
KB_EV_PER_K = 8.617333262145e-5
RELATIVE_ACTIVITY_LEVELS = (("10pct", 0.10), ("1pct", 0.01), ("0p1pct", 0.001))


def parse_activity_model(path: Path) -> tuple[dict[str, float], float, float]:
    elements: list[str] | None = None
    zone1_values: list[float] | None = None
    e_opt: float | None = None
    activity_temperature = 298.0
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            fields = raw.split()
            if not fields:
                continue
            if fields[0] == "elements":
                elements = fields[1:]
            elif fields[0] == "zone1":
                zone1_values = [float(value) for value in fields[1:]]
            elif fields[0] == "e_opt":
                e_opt = float(fields[1])
            elif fields[0] == "activity_temperature":
                activity_temperature = float(fields[1])
    if elements is None or zone1_values is None or e_opt is None:
        raise ValueError(f"Could not read elements, zone1, and e_opt from {path}")
    if len(elements) != len(zone1_values):
        raise ValueError("activity-model elements and zone1 lengths differ")
    zone1 = dict(zip(elements, zone1_values))
    missing = sorted(set(ELEMENTS) - set(zone1))
    if missing:
        raise ValueError(f"activity model is missing elements: {', '.join(missing)}")
    return zone1, e_opt, activity_temperature


def shard_dirs(results_root: Path, world_size: int | None) -> list[Path]:
    if world_size is None:
        return sorted(path for path in results_root.glob("shard_*") if path.is_dir())
    return [results_root / f"shard_{index:02d}" for index in range(world_size)]


def extract_trial_shifts(paths: list[Path], chunksize: int) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        seen: set[int] = set()
        for chunk in pd.read_csv(path, usecols=["trial", *SHIFT_COLUMNS], chunksize=chunksize):
            chunk["trial"] = pd.to_numeric(chunk["trial"], errors="coerce")
            chunk = chunk.dropna(subset=["trial"])
            chunk["trial"] = chunk["trial"].astype(int)
            chunk = chunk[~chunk["trial"].isin(seen)].drop_duplicates("trial", keep="first")
            if chunk.empty:
                continue
            seen.update(chunk["trial"].tolist())
            pieces.append(chunk)
    if not pieces:
        raise FileNotFoundError("No trial_composition_samples.csv rows were found")
    shifts = pd.concat(pieces, ignore_index=True).drop_duplicates("trial", keep="first")
    for column in SHIFT_COLUMNS:
        shifts[column] = pd.to_numeric(shifts[column], errors="coerce")
    if shifts[SHIFT_COLUMNS].isna().any().any():
        raise ValueError("Non-numeric BE shifts were found")
    return shifts.sort_values("trial").reset_index(drop=True)


def add_multilabel_features(out: pd.DataFrame, activity_temperature: float) -> pd.DataFrame:
    kbt = KB_EV_PER_K * activity_temperature
    distances = out[[f"distance_to_eopt_{element}" for element in ELEMENTS]].to_numpy(float)
    unnormalized = np.exp(-distances / kbt)
    total = unnormalized.sum(axis=1)
    for index, element in enumerate(ELEMENTS):
        out[f"zone1_activity_weight_{element}"] = unnormalized[:, index] / total
    out["zone1_activity_weight_pdpt"] = (
        out["zone1_activity_weight_Pd"] + out["zone1_activity_weight_Pt"]
    )
    out["zone1_activity_weight_other"] = 1.0 - out["zone1_activity_weight_pdpt"]

    for label, relative_activity in RELATIVE_ACTIVITY_LEVELS:
        cutoff = -kbt * math.log(relative_activity)
        active_columns = []
        for element in ELEMENTS:
            column = f"active_{element}_{label}"
            out[column] = out[f"distance_to_eopt_{element}"] <= cutoff
            active_columns.append(column)
        pd_both = out[f"active_Pd_{label}"] & out[f"active_Pt_{label}"]
        pd_any = out[f"active_Pd_{label}"] | out[f"active_Pt_{label}"]
        other_all = out[f"active_Ir_{label}"] & out[f"active_Rh_{label}"] & out[f"active_Ru_{label}"]
        other_any = out[f"active_Ir_{label}"] | out[f"active_Rh_{label}"] | out[f"active_Ru_{label}"]
        hierarchy = np.select(
            [pd_both & ~other_any, other_all & ~pd_any, pd_any & ~other_any,
             other_any & ~pd_any, pd_any & other_any],
            ["PdPt_both_only", "Other_all_only", "PdPt_any_only", "Other_any_only", "Mixed"],
            default="None",
        )
        out[f"hierarchical_group_{label}"] = hierarchy
        out[f"strict_group_{label}"] = np.select(
            [hierarchy == "PdPt_both_only", hierarchy == "Other_all_only"],
            ["PdPt_both_only", "Other_all_only"], default="Excluded"
        )
        out[f"family_exclusive_group_{label}"] = np.select(
            [np.char.startswith(hierarchy.astype(str), "PdPt_"),
             np.char.startswith(hierarchy.astype(str), "Other_")],
            ["PdPt_only", "Other_only"], default="Excluded"
        )
        out[f"activity_cutoff_eV_{label}"] = cutoff
    out["activity_temperature_K"] = activity_temperature
    return out


def classify_trials(
    shifts: pd.DataFrame, zone1: dict[str, float], e_opt: float, activity_temperature: float
) -> pd.DataFrame:
    out = shifts.copy()
    distance_columns: list[str] = []
    for element in ELEMENTS:
        corrected = f"shifted_zone1_{element}"
        distance = f"distance_to_eopt_{element}"
        out[corrected] = zone1[element] - out[f"be_shift_{element}"]
        out[distance] = (out[corrected] - e_opt).abs()
        distance_columns.append(distance)

    distances = out[distance_columns].to_numpy(float)
    order = np.argsort(distances, axis=1, kind="stable")
    first = order[:, 0]
    second = order[:, 1]
    labels = np.asarray(ELEMENTS)
    rows = np.arange(len(out))
    out["active_site_element"] = labels[first]
    out["second_closest_element"] = labels[second]
    out["closest_distance"] = distances[rows, first]
    out["second_closest_distance"] = distances[rows, second]
    out["active_site_margin"] = out["second_closest_distance"] - out["closest_distance"]
    out["active_site_group"] = np.where(out["active_site_element"].isin(PDPT), "Pd/Pt", "Other")
    out["distance_pdpt"] = out[["distance_to_eopt_Pd", "distance_to_eopt_Pt"]].min(axis=1)
    out["distance_other"] = out[["distance_to_eopt_Ir", "distance_to_eopt_Rh", "distance_to_eopt_Ru"]].min(axis=1)
    out["delta_distance_other_minus_pdpt"] = out["distance_other"] - out["distance_pdpt"]
    out["e_opt"] = e_opt
    return add_multilabel_features(out, activity_temperature)


def read_shard_csvs(paths: list[Path], filename: str) -> pd.DataFrame:
    frames = [pd.read_csv(path / filename) for path in paths if (path / filename).exists()]
    if not frames:
        raise FileNotFoundError(f"No {filename} files were found")
    return pd.concat(frames, ignore_index=True).drop_duplicates("trial", keep="last")


def assemble_metrics(
    classification: pd.DataFrame,
    shard_paths: list[Path],
    crps_path: Path,
    best_crps_path: Path,
    random_crps_path: Path | None,
) -> pd.DataFrame:
    selected = read_shard_csvs(shard_paths, "selected_temperature_by_trial.csv")
    selected = selected.rename(columns={
        "cemc_selected_temp": "selected_temperature",
        "cemc_tau": "cemc_selected_tau",
        "cemc_mse": "cemc_selected_mse",
        "cemc_score_tau_minus_mse": "cemc_selected_score",
    })
    crps = pd.read_csv(crps_path)
    required = {"trial", "temperature", "mean_crps_scaled"}
    if not required.issubset(crps.columns):
        raise ValueError(f"{crps_path} must contain {sorted(required)}")
    crps_at_selected = selected[["trial", "selected_temperature"]].merge(
        crps[["trial", "temperature", "mean_crps_scaled"]],
        left_on=["trial", "selected_temperature"],
        right_on=["trial", "temperature"],
        how="left",
    )[["trial", "mean_crps_scaled"]].rename(
        columns={"mean_crps_scaled": "crps_at_cemc_selected_temperature"}
    )

    best = pd.read_csv(best_crps_path).rename(columns={
        "temperature": "best_crps_temperature",
        "mean_crps_scaled": "best_mean_crps_scaled",
    })
    best_columns = [column for column in (
        "trial", "best_crps_temperature", "best_mean_crps_scaled",
        "mean_crps_raw", "median_crps_raw", "mean_pred_sd_raw",
    ) if column in best.columns]
    best = best[best_columns].drop_duplicates("trial", keep="last")

    merged = classification.merge(selected, on="trial", how="inner")
    merged = merged.merge(crps_at_selected, on="trial", how="left").merge(best, on="trial", how="left")
    if random_crps_path is not None and random_crps_path.exists():
        random_crps = pd.read_csv(random_crps_path)
        keep = [column for column in ("trial", "random_mean_crps_scaled") if column in random_crps.columns]
        if len(keep) == 2:
            merged = merged.merge(random_crps[keep].drop_duplicates("trial"), on="trial", how="left")
            merged["delta_crps_random_minus_best_cemc"] = (
                merged["random_mean_crps_scaled"] - merged["best_mean_crps_scaled"]
            )
    return merged.sort_values("trial").reset_index(drop=True)


def assemble_temperature_metrics(
    classification: pd.DataFrame,
    shard_paths: list[Path],
    crps_path: Path,
) -> pd.DataFrame:
    frames = []
    for shard in shard_paths:
        path = shard / "metrics_by_trial_temperature.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No metrics_by_trial_temperature.csv files were found")
    metrics = pd.concat(frames, ignore_index=True)
    if "method" in metrics.columns:
        metrics = metrics[metrics["method"].astype(str).str.lower() == "cemc"]
    metrics = metrics.drop_duplicates(["trial", "temperature"], keep="last")
    metrics = metrics.rename(columns={"score_tau_minus_mse": "score"})

    crps = pd.read_csv(crps_path)
    required = {"trial", "temperature", "mean_crps_scaled"}
    if not required.issubset(crps.columns):
        raise ValueError(f"{crps_path} must contain {sorted(required)}")
    crps = crps.drop_duplicates(["trial", "temperature"], keep="last")
    joined = metrics.merge(
        crps[["trial", "temperature", "mean_crps_scaled"]],
        on=["trial", "temperature"],
        how="inner",
    )
    joined = classification.merge(joined, on="trial", how="inner")
    return joined.sort_values(["temperature", "trial"]).reset_index(drop=True)


def finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.dropna()


def group_summary(data: pd.DataFrame, metrics: list[str], subset_name: str) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for metric in metrics:
        if metric not in data.columns:
            continue
        groups = {name: finite(part[metric]) for name, part in data.groupby("active_site_group")}
        for name, values in groups.items():
            rows.append({
                "subset": subset_name,
                "metric": metric,
                "group": name,
                "n": len(values),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "median": values.median(),
                "q025": values.quantile(0.025),
                "q975": values.quantile(0.975),
            })
        if {"Pd/Pt", "Other"}.issubset(groups):
            a, b = groups["Pd/Pt"], groups["Other"]
            pooled_den = len(a) + len(b) - 2
            pooled_sd = math.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / pooled_den) if pooled_den > 0 else math.nan
            rows.append({
                "subset": subset_name,
                "metric": metric,
                "group": "Pd/Pt minus Other",
                "n": min(len(a), len(b)),
                "mean": a.mean() - b.mean(),
                "sd": (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else math.nan,
                "median": a.median() - b.median(),
                "q025": math.nan,
                "q975": math.nan,
            })
    return pd.DataFrame(rows)


def continuous_summary(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    x_name = "delta_distance_other_minus_pdpt"
    rows = []
    for metric in metrics:
        if metric not in data.columns:
            continue
        pair = data[[x_name, metric]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 3:
            continue
        rows.append({
            "predictor": x_name,
            "metric": metric,
            "n": len(pair),
            "pearson_r": pair[x_name].corr(pair[metric], method="pearson"),
            "spearman_rho": pair[x_name].rank().corr(pair[metric].rank(), method="pearson"),
            "linear_slope": np.polyfit(pair[x_name], pair[metric], 1)[0],
        })
    return pd.DataFrame(rows)


def predictor_summary(data: pd.DataFrame, metrics: list[str], predictor: str) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        if metric not in data.columns:
            continue
        pair = data[[predictor, metric]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 3:
            continue
        rows.append({
            "predictor": predictor, "metric": metric, "n": len(pair),
            "pearson_r": pair[predictor].corr(pair[metric]),
            "spearman_rho": pair[predictor].rank().corr(pair[metric].rank()),
            "linear_slope": np.polyfit(pair[predictor], pair[metric], 1)[0],
        })
    return pd.DataFrame(rows)


def two_group_summary(
    data: pd.DataFrame, metrics: list[str], group_column: str,
    group_a: str, group_b: str, analysis: str, temperature: float | None = None,
) -> pd.DataFrame:
    subset = data[data[group_column].isin([group_a, group_b])]
    rows = []
    for metric in metrics:
        if metric not in subset.columns:
            continue
        groups = {name: finite(part[metric]) for name, part in subset.groupby(group_column)}
        for name in (group_a, group_b):
            values = groups.get(name, pd.Series(dtype=float))
            rows.append({"temperature": temperature, "analysis": analysis, "metric": metric,
                         "group": name, "n": len(values), "mean": values.mean(),
                         "sd": values.std(ddof=1), "median": values.median()})
        a, b = groups.get(group_a, pd.Series(dtype=float)), groups.get(group_b, pd.Series(dtype=float))
        pooled_den = len(a) + len(b) - 2
        pooled_sd = math.sqrt(((len(a)-1)*a.var(ddof=1)+(len(b)-1)*b.var(ddof=1))/pooled_den) if pooled_den > 0 else math.nan
        rows.append({"temperature": temperature, "analysis": analysis, "metric": metric,
                     "group": f"{group_a} minus {group_b}", "n": min(len(a), len(b)),
                     "mean": a.mean()-b.mean(),
                     "sd": (a.mean()-b.mean())/pooled_sd if pooled_sd > 0 else math.nan,
                     "median": a.median()-b.median()})
    return pd.DataFrame(rows)


def multilabel_outputs(
    trial_metrics: pd.DataFrame, temperature_metrics: pd.DataFrame,
    trial_metric_names: list[str], temperature_metric_names: list[str], output_dir: Path,
) -> None:
    count_rows = []
    trial_summaries = []
    temperature_summaries = []
    for label, relative_activity in RELATIVE_ACTIVITY_LEVELS:
        hierarchy = f"hierarchical_group_{label}"
        counts = trial_metrics.groupby(hierarchy, as_index=False).agg(n_trials=("trial", "size"))
        counts["relative_activity_cutoff"] = relative_activity
        counts["distance_cutoff_eV"] = trial_metrics[f"activity_cutoff_eV_{label}"].iloc[0]
        counts["fraction_trials"] = counts["n_trials"] / len(trial_metrics)
        counts = counts.rename(columns={hierarchy: "hierarchical_group"})
        count_rows.append(counts)
        trial_summaries.extend([
            two_group_summary(trial_metrics, trial_metric_names, f"strict_group_{label}",
                              "PdPt_both_only", "Other_all_only", f"strict_{label}"),
            two_group_summary(trial_metrics, trial_metric_names, f"family_exclusive_group_{label}",
                              "PdPt_only", "Other_only", f"family_exclusive_{label}"),
        ])
        for temperature, part in temperature_metrics.groupby("temperature"):
            temperature_summaries.extend([
                two_group_summary(part, temperature_metric_names, f"strict_group_{label}",
                                  "PdPt_both_only", "Other_all_only", f"strict_{label}", temperature),
                two_group_summary(part, temperature_metric_names, f"family_exclusive_group_{label}",
                                  "PdPt_only", "Other_only", f"family_exclusive_{label}", temperature),
            ])
    pd.concat(count_rows, ignore_index=True).to_csv(output_dir / "hierarchical_group_counts_by_cutoff.csv", index=False)
    pd.concat(trial_summaries, ignore_index=True).to_csv(output_dir / "threshold_group_metric_summary.csv", index=False)
    pd.concat(temperature_summaries, ignore_index=True).to_csv(
        output_dir / "temperature_threshold_group_metric_summary.csv", index=False
    )
    predictor_summary(trial_metrics, trial_metric_names, "zone1_activity_weight_pdpt").to_csv(
        output_dir / "activity_weight_continuous_summary.csv", index=False
    )
    temp_weight = []
    for temperature, part in temperature_metrics.groupby("temperature"):
        summary = predictor_summary(part, temperature_metric_names, "zone1_activity_weight_pdpt")
        summary.insert(0, "temperature", temperature)
        temp_weight.append(summary)
    pd.concat(temp_weight, ignore_index=True).to_csv(
        output_dir / "temperature_activity_weight_continuous_summary.csv", index=False
    )


def temperature_group_summary(data: pd.DataFrame, metrics: list[str], margin_threshold: float) -> pd.DataFrame:
    pieces = []
    for temperature, part in data.groupby("temperature", sort=True):
        all_summary = group_summary(part, metrics, "all_trials")
        all_summary.insert(0, "temperature", temperature)
        pieces.append(all_summary)
        confident = part[part["active_site_margin"] >= margin_threshold]
        confident_summary = group_summary(confident, metrics, f"margin_ge_{margin_threshold:g}_eV")
        confident_summary.insert(0, "temperature", temperature)
        pieces.append(confident_summary)
    return pd.concat(pieces, ignore_index=True)


def temperature_continuous_summary(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    pieces = []
    for temperature, part in data.groupby("temperature", sort=True):
        summary = continuous_summary(part, metrics)
        summary.insert(0, "temperature", temperature)
        pieces.append(summary)
    return pd.concat(pieces, ignore_index=True)


def plot_group_metrics(data: pd.DataFrame, metrics: list[str], output: Path) -> None:
    available = [metric for metric in metrics if metric in data and finite(data[metric]).size]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(4.7 * len(available), 4.6), squeeze=False)
    colors = ["#4C78A8", "#F58518"]
    for ax, metric in zip(axes[0], available):
        values = [finite(data.loc[data["active_site_group"] == group, metric]) for group in ("Pd/Pt", "Other")]
        box = ax.boxplot(values, tick_labels=["Pd/Pt", "Other"], patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(metric.replace("_", " "))
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Metrics by shifted-zone1 active-site group", weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_continuous(data: pd.DataFrame, metrics: list[str], output: Path) -> None:
    x_name = "delta_distance_other_minus_pdpt"
    available = [metric for metric in metrics if metric in data and finite(data[metric]).size]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(4.9 * len(available), 4.5), squeeze=False)
    for ax, metric in zip(axes[0], available):
        pair = data[[x_name, metric]].replace([np.inf, -np.inf], np.nan).dropna()
        ax.hexbin(pair[x_name], pair[metric], gridsize=35, mincnt=1, cmap="viridis")
        if len(pair) >= 2:
            slope, intercept = np.polyfit(pair[x_name], pair[metric], 1)
            xline = np.linspace(pair[x_name].min(), pair[x_name].max(), 200)
            ax.plot(xline, slope * xline + intercept, color="black", linewidth=1.5)
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.1)
        ax.set_xlabel("distance(Other) - distance(Pd/Pt), eV")
        ax.set_ylabel(metric.replace("_", " "))
    fig.suptitle("Continuous active-site regime analysis", weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_temperature_group_trends(data: pd.DataFrame, metrics: list[str], output: Path) -> None:
    available = [metric for metric in metrics if metric in data and finite(data[metric]).size]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(5.0 * len(available), 4.5), squeeze=False)
    colors = {"Pd/Pt": "#4C78A8", "Other": "#F58518"}
    for ax, metric in zip(axes[0], available):
        clean = data[["temperature", "active_site_group", metric]].replace([np.inf, -np.inf], np.nan).dropna()
        summary = clean.groupby(["temperature", "active_site_group"], as_index=False).agg(
            mean=(metric, "mean"), sd=(metric, "std"), n=(metric, "size")
        )
        summary["sem"] = summary["sd"] / np.sqrt(summary["n"])
        for group in ("Pd/Pt", "Other"):
            part = summary[summary["active_site_group"] == group].sort_values("temperature")
            x = part["temperature"].to_numpy(float)
            y = part["mean"].to_numpy(float)
            ci = 1.96 * part["sem"].to_numpy(float)
            ax.plot(x, y, marker="o", markersize=4, color=colors[group], label=group)
            ax.fill_between(x, y - ci, y + ci, color=colors[group], alpha=0.18)
        ax.set_xlabel("CEMC snapshot temperature (K)")
        ax.set_ylabel(metric.replace("_", " "))
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("Temperature-resolved metrics by shifted-zone1 regime", weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_temperature_effects(summary: pd.DataFrame, metrics: list[str], output: Path) -> None:
    effects = summary[(summary["subset"] == "all_trials") & (summary["group"] == "Pd/Pt minus Other")]
    available = [metric for metric in metrics if metric in set(effects["metric"])]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(4.9 * len(available), 4.3), squeeze=False)
    for ax, metric in zip(axes[0], available):
        part = effects[effects["metric"] == metric].sort_values("temperature")
        ax.plot(part["temperature"], part["mean"], marker="o", color="#6F4E7C")
        ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xlabel("CEMC snapshot temperature (K)")
        ax.set_ylabel("Pd/Pt minus Other")
        ax.set_title(metric.replace("_", " "))
        ax.invert_xaxis()
        ax.grid(alpha=0.25)
    fig.suptitle("Temperature-resolved group mean differences", weight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_family_exclusive_temperature_trends(
    data: pd.DataFrame, metrics: list[str], output: Path
) -> None:
    fig, axes = plt.subplots(
        len(RELATIVE_ACTIVITY_LEVELS), len(metrics),
        figsize=(5.1 * len(metrics), 4.0 * len(RELATIVE_ACTIVITY_LEVELS)),
        squeeze=False, sharex=True,
    )
    colors = {"PdPt_only": "#4C78A8", "Other_only": "#F58518"}
    display = {"PdPt_only": "Pd/Pt only", "Other_only": "Other only"}
    for row, (label, relative_activity) in enumerate(RELATIVE_ACTIVITY_LEVELS):
        group_column = f"family_exclusive_group_{label}"
        subset = data[data[group_column].isin(colors)].copy()
        group_counts = (
            subset[["trial", group_column]].drop_duplicates()
            .groupby(group_column)["trial"].size().to_dict()
        )
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            clean = subset[["temperature", group_column, metric]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            summary = clean.groupby(["temperature", group_column], as_index=False).agg(
                mean=(metric, "mean"), sd=(metric, "std"), n=(metric, "size")
            )
            summary["sem"] = summary["sd"] / np.sqrt(summary["n"])
            for group in ("PdPt_only", "Other_only"):
                part = summary[summary[group_column] == group].sort_values("temperature")
                x = part["temperature"].to_numpy(float)
                y = part["mean"].to_numpy(float)
                ci = 1.96 * part["sem"].to_numpy(float)
                ax.plot(x, y, marker="o", markersize=3.5, linewidth=1.7,
                        color=colors[group],
                        label=f"{display[group]} (n={group_counts.get(group, 0):,})")
                ax.fill_between(x, y - ci, y + ci, color=colors[group], alpha=0.18)
            if row == 0:
                ax.set_title(metric.replace("mean_crps_scaled", "CRPS").upper())
            if col == 0:
                cutoff = -KB_EV_PER_K * float(data["activity_temperature_K"].iloc[0]) * math.log(relative_activity)
                ax.set_ylabel(
                    f"{relative_activity * 100:g}% activity cutoff\n({cutoff:.3f} eV)\nMean {metric}"
                )
            if row == len(RELATIVE_ACTIVITY_LEVELS) - 1:
                ax.set_xlabel("CEMC snapshot temperature (K)")
            ax.invert_xaxis()
            ax.grid(alpha=0.25)
            if col == len(metrics) - 1:
                ax.legend(frameon=False, loc="best")
    fig.suptitle(
        "Temperature-resolved MC metrics: Pd/Pt-only vs Other-only regimes",
        fontsize=16, weight="bold", y=1.002,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--activity-model", type=Path, default=Path("activity_model_oh.txt"))
    parser.add_argument("--crps-input", type=Path, default=Path("results/plots/crps_by_trial_temperature_1400.csv"))
    parser.add_argument("--best-crps-input", type=Path, default=Path("results/plots/best_crps_temperature_by_trial_1400.csv"))
    parser.add_argument("--random-crps-input", type=Path, default=Path("results/plots/random_crps_by_trial_1400.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/active_site_group_analysis"))
    parser.add_argument("--margin-threshold", type=float, default=0.05)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--reuse-classification", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    zone1, e_opt, activity_temperature = parse_activity_model(args.activity_model)
    shards = shard_dirs(args.results_root, args.world_size)
    classification_path = args.output_dir / "trial_active_site_classification.csv"
    if args.reuse_classification and classification_path.exists():
        classification = pd.read_csv(classification_path)
        classification = add_multilabel_features(classification, activity_temperature)
        classification.to_csv(classification_path, index=False)
    else:
        sample_paths = [path / "trial_composition_samples.csv" for path in shards]
        shifts = extract_trial_shifts(sample_paths, args.chunksize)
        classification = classify_trials(shifts, zone1, e_opt, activity_temperature)
        classification.to_csv(classification_path, index=False)

    trial_metrics = assemble_metrics(
        classification, shards, args.crps_input, args.best_crps_input, args.random_crps_input
    )
    trial_metrics["confident_group"] = trial_metrics["active_site_margin"] >= args.margin_threshold
    trial_metrics.to_csv(args.output_dir / "trial_active_site_metrics.csv", index=False)

    metrics = [
        "cemc_selected_tau", "cemc_selected_mse", "crps_at_cemc_selected_temperature",
        "best_mean_crps_scaled", "delta_tau_cemc_minus_random",
        "delta_mse_random_minus_cemc", "delta_crps_random_minus_best_cemc",
    ]
    summaries = [group_summary(trial_metrics, metrics, "all_trials")]
    confident = trial_metrics[trial_metrics["confident_group"]].copy()
    summaries.append(group_summary(confident, metrics, f"margin_ge_{args.margin_threshold:g}_eV"))
    pd.concat(summaries, ignore_index=True).to_csv(args.output_dir / "group_metric_summary.csv", index=False)
    continuous_summary(trial_metrics, metrics).to_csv(args.output_dir / "continuous_association_summary.csv", index=False)

    temperature_metrics = assemble_temperature_metrics(classification, shards, args.crps_input)
    temperature_metrics["confident_group"] = temperature_metrics["active_site_margin"] >= args.margin_threshold
    temperature_columns = [
        "trial", "temperature", "active_site_element", "active_site_group", "active_site_margin",
        "delta_distance_other_minus_pdpt", "tau", "mse", "score", "mean_crps_scaled",
    ]
    temperature_metrics[[column for column in temperature_columns if column in temperature_metrics.columns]].to_csv(
        args.output_dir / "temperature_resolved_trial_metrics.csv", index=False
    )
    temperature_metric_names = ["tau", "mse", "mean_crps_scaled"]
    temp_group_summary = temperature_group_summary(
        temperature_metrics, temperature_metric_names, args.margin_threshold
    )
    temp_group_summary.to_csv(args.output_dir / "temperature_group_metric_summary.csv", index=False)
    temperature_continuous_summary(temperature_metrics, temperature_metric_names).to_csv(
        args.output_dir / "temperature_continuous_association_summary.csv", index=False
    )
    multilabel_outputs(
        trial_metrics, temperature_metrics, metrics, temperature_metric_names, args.output_dir
    )

    counts = (
        trial_metrics.groupby(["active_site_group", "active_site_element"], as_index=False)
        .agg(n_trials=("trial", "size"), median_margin=("active_site_margin", "median"))
    )
    counts["fraction_all_trials"] = counts["n_trials"] / len(trial_metrics)
    counts.to_csv(args.output_dir / "active_site_group_counts.csv", index=False)
    plot_group_metrics(trial_metrics, metrics[:4], args.output_dir / "group_metric_boxplots.png")
    plot_continuous(trial_metrics, metrics[:4], args.output_dir / "continuous_metric_hexbin.png")
    plot_temperature_group_trends(
        temperature_metrics, temperature_metric_names, args.output_dir / "temperature_group_metric_trends.png"
    )
    plot_temperature_effects(
        temp_group_summary, temperature_metric_names, args.output_dir / "temperature_group_mean_differences.png"
    )
    plot_family_exclusive_temperature_trends(
        temperature_metrics, temperature_metric_names,
        args.output_dir / "temperature_family_exclusive_metric_trends.png",
    )

    print(f"Wrote active-site analysis for {len(trial_metrics)} trials to {args.output_dir}")
    print(counts.to_string(index=False))


if __name__ == "__main__":
    main()
