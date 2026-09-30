#!/usr/bin/env python3
"""Plot joint delta MSE vs delta tau hexbin for UQMC trials."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PREFERRED_INPUTS = (
    Path("plots/mse_distribution_cemc_vs_random_trial0-999_data.csv"),
    Path("plots/metric_distribution_rows.csv"),
)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {
        "cemc_selected_tau": "cemc_tau",
        "cemc_selected_mse": "cemc_mse",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    for col in ("trial", "cemc_mse", "random_mse", "cemc_tau", "random_tau"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "delta_mse_random_minus_cemc" not in df.columns:
        if not {"random_mse", "cemc_mse"}.issubset(df.columns):
            raise SystemExit("Missing columns needed for random MSE - CEMC MSE")
        df["delta_mse_random_minus_cemc"] = df["random_mse"] - df["cemc_mse"]

    if "delta_tau_cemc_minus_random" not in df.columns:
        if not {"cemc_tau", "random_tau"}.issubset(df.columns):
            raise SystemExit("Missing columns needed for CEMC tau - random tau")
        df["delta_tau_cemc_minus_random"] = df["cemc_tau"] - df["random_tau"]

    df["delta_mse_random_minus_cemc"] = pd.to_numeric(
        df["delta_mse_random_minus_cemc"], errors="coerce"
    )
    df["delta_tau_cemc_minus_random"] = pd.to_numeric(
        df["delta_tau_cemc_minus_random"], errors="coerce"
    )

    if "trial" in df.columns:
        df = df.dropna(subset=["trial"]).copy()
        df["trial"] = df["trial"].astype(int)
        df = df.drop_duplicates(subset=["trial"], keep="last").sort_values("trial")
    return df


def load_delta_rows(results_root: Path, input_csv: Path | None, world_size: int | None) -> pd.DataFrame:
    if input_csv is not None:
        if not input_csv.exists():
            raise SystemExit(f"Missing input CSV: {input_csv}")
        return normalize_columns(pd.read_csv(input_csv))

    for rel_path in PREFERRED_INPUTS:
        path = results_root / rel_path
        if path.exists() and path.stat().st_size > 0:
            return normalize_columns(pd.read_csv(path))

    if world_size is None:
        shard_dirs = sorted(results_root.glob("shard_*"))
    else:
        shard_dirs = [results_root / f"shard_{idx:02d}" for idx in range(world_size)]

    frames = []
    for shard_dir in shard_dirs:
        path = shard_dir / "paired_comparison_by_trial.csv"
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_csv(path)
        df["source_shard"] = shard_dir.name
        frames.append(df)

    if not frames:
        raise SystemExit(f"No paired comparison CSVs found under {results_root}")
    return normalize_columns(pd.concat(frames, ignore_index=True))


def filter_trials(df: pd.DataFrame, trial_min: int | None, trial_max: int | None) -> pd.DataFrame:
    if "trial" not in df.columns:
        return df
    out = df.copy()
    if trial_min is not None:
        out = out[out["trial"] >= trial_min]
    if trial_max is not None:
        out = out[out["trial"] <= trial_max]
    return out


def quadrant_stats(x: pd.Series, y: pd.Series) -> dict[str, dict[str, float]]:
    n = len(x)
    masks = {
        "upper_right": (x > 0) & (y > 0),
        "upper_left": (x <= 0) & (y > 0),
        "lower_left": (x <= 0) & (y <= 0),
        "lower_right": (x > 0) & (y <= 0),
    }
    labels = {
        "upper_right": r"$P(\Delta\mathrm{MSE}>0,\ \Delta\tau>0)$",
        "upper_left": r"$P(\Delta\mathrm{MSE}\leq0,\ \Delta\tau>0)$",
        "lower_left": r"$P(\Delta\mathrm{MSE}\leq0,\ \Delta\tau\leq0)$",
        "lower_right": r"$P(\Delta\mathrm{MSE}>0,\ \Delta\tau\leq0)$",
    }
    return {
        key: {
            "label": labels[key],
            "count": int(mask.sum()),
            "probability": float(mask.mean()) if n else float("nan"),
        }
        for key, mask in masks.items()
    }


def plot_hexbin(df: pd.DataFrame, output: Path, label: str, gridsize: int, dpi: int) -> pd.DataFrame:
    cols = ["delta_mse_random_minus_cemc", "delta_tau_cemc_minus_random"]
    clean = df.dropna(subset=cols).copy()
    if clean.empty:
        raise SystemExit("No finite delta rows found")

    x = clean["delta_mse_random_minus_cemc"]
    y = clean["delta_tau_cemc_minus_random"]
    both_positive = (x > 0) & (y > 0)
    n = len(clean)
    n_both = int(both_positive.sum())
    p_both = n_both / n
    quadrants = quadrant_stats(x, y)

    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    ax.axvspan(0, max(float(x.max()), 0), ymin=0.5, ymax=1.0, color="#dff0df", alpha=0.45)
    hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=1, cmap="viridis")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)

    title_label = f" {label}" if label else ""
    ax.set_title(
        f"Joint CEMC improvement: $\\Delta\\tau$ vs $\\Delta\\mathrm{{MSE}}${title_label}\n"
        "Trial-wise CEMC selected temperature",
        fontsize=14,
    )
    ax.set_xlabel(r"$\Delta\mathrm{MSE}=\mathrm{MSE}_{random}-\mathrm{MSE}_{CEMC}$", fontsize=12)
    ax.set_ylabel(r"$\Delta\tau=\tau_{CEMC}-\tau_{random}$", fontsize=12)

    annotation_specs = [
        ("upper_left", 0.03, 0.97, "left", "top"),
        ("upper_right", 0.97, 0.97, "right", "top"),
        ("lower_left", 0.03, 0.03, "left", "bottom"),
        ("lower_right", 0.97, 0.03, "right", "bottom"),
    ]
    for key, xpos, ypos, ha, va in annotation_specs:
        q = quadrants[key]
        ax.text(
            xpos,
            ypos,
            f"{q['label']}\n{q['probability']:.3f}",
            transform=ax.transAxes,
            ha=ha,
            va=va,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "0.8", "alpha": 0.90},
            fontsize=9,
        )

    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Hexbin count")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi)
    plt.close(fig)

    return pd.DataFrame(
        [
            {
                "n_trials": n,
                "n_delta_mse_positive": int((x > 0).sum()),
                "p_delta_mse_positive": float((x > 0).mean()),
                "n_delta_tau_positive": int((y > 0).sum()),
                "p_delta_tau_positive": float((y > 0).mean()),
                "n_both_positive": n_both,
                "p_both_positive": p_both,
                "n_mse_nonpositive_tau_positive": quadrants["upper_left"]["count"],
                "p_mse_nonpositive_tau_positive": quadrants["upper_left"]["probability"],
                "n_both_nonpositive": quadrants["lower_left"]["count"],
                "p_both_nonpositive": quadrants["lower_left"]["probability"],
                "n_mse_positive_tau_nonpositive": quadrants["lower_right"]["count"],
                "p_mse_positive_tau_nonpositive": quadrants["lower_right"]["probability"],
                "output": str(output),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results", help="Directory containing UQMC results")
    parser.add_argument("--input-csv", default="", help="Optional combined paired-comparison CSV")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--trial-min", type=int, default=None)
    parser.add_argument("--trial-max", type=int, default=None)
    parser.add_argument("--label", default="1400", help="Text appended to title and output filename")
    parser.add_argument("--gridsize", type=int, default=17)
    parser.add_argument("--output-dir", default="results/plots")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    results_root = Path(args.results_root).resolve()
    input_csv = Path(args.input_csv).resolve() if args.input_csv else None
    output_dir = Path(args.output_dir).resolve()
    label = args.label.strip()
    suffix = label.replace(" ", "_") if label else "UQMC"

    df = load_delta_rows(results_root, input_csv, args.world_size)
    df = filter_trials(df, args.trial_min, args.trial_max)
    if df.empty:
        raise SystemExit("No rows remain after trial filtering")

    output = output_dir / f"delta_tau_vs_delta_mse_hexbin_{suffix}.png"
    summary = plot_hexbin(df, output, label, args.gridsize, args.dpi)
    summary_path = output_dir / f"delta_tau_vs_delta_mse_hexbin_summary_{suffix}.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Loaded {len(df)} trial rows")
    print(f"Wrote {output}")
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
