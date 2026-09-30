import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from scipy.ndimage import gaussian_filter
N_TRIALS=10000
HIST_BINS=20
KDE_SIGMA=1.0

def scale_minus1_0(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    lo = np.min(values, axis=-1, keepdims=True)
    hi = np.max(values, axis=-1, keepdims=True)
    out = np.zeros_like(values)
    np.divide(-(values - lo), hi - lo, out=out, where=(hi - lo) > 0)
    return np.clip(out, -1.0, 0.0)

def mean_abs_within_rows(values: np.ndarray) -> np.ndarray:
    ordered = np.sort(values, axis=1)
    n = ordered.shape[1]
    weights = 2.0 * np.arange(1, n + 1) - n - 1.0
    return 2.0 * (ordered @ weights) / (n * n)

def metrics_batch(pred: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mse = np.mean((pred - observed[None, :]) ** 2, axis=1)
    tau = np.array([kendalltau(observed, row, nan_policy="raise").statistic for row in pred])
    y = np.sort(observed)
    prefix = np.concatenate(([0.0], np.cumsum(y)))
    k = np.searchsorted(y, pred, side="right")
    left = pred * k - prefix[k]
    right = (prefix[-1] - prefix[k]) - pred * (len(y) - k)
    cross = np.mean(left + right, axis=1) / len(y)
    crps = np.maximum(0.0, cross - 0.5 * mean_abs_within_rows(pred)
                      - 0.5 * float(mean_abs_within_rows(observed[None, :])[0]))
    return tau, mse, crps

def select_best_temperature(cemc: pd.DataFrame) -> pd.DataFrame:
    clean = cemc.replace([np.inf, -np.inf], np.nan).dropna(subset=["tau", "mse", "crps"]).copy()
    terms = []
    for column, maximize in (("tau", True), ("mse", False), ("crps", False)):
        values = clean[column].to_numpy(float)
        lo, hi = values.min(), values.max()
        unit = np.zeros_like(values) if np.isclose(lo, hi) else (values - lo) / (hi - lo)
        terms.append(1.0 - unit if maximize else unit)
    clean["temperature_ideal_distance"] = np.sqrt(sum(value * value for value in terms))
    best = (clean.sort_values(
        ["trial", "temperature_ideal_distance", "tau", "mse", "crps", "temperature"],
        ascending=[True, True, False, True, True, True], kind="mergesort")
        .drop_duplicates("trial").sort_values("trial").reset_index(drop=True))
    if len(best) != N_TRIALS or best.trial.nunique() != N_TRIALS:
        raise ValueError("Best-temperature selection did not produce 10,000 trials")
    return best

def kde_assign(
    points: pd.DataFrame, require_positive_tau: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    xyz = points[["tau", "mse", "crps"]].to_numpy(float)
    counts, edges = np.histogramdd(xyz, bins=HIST_BINS)
    smooth = gaussian_filter(counts.astype(float), sigma=KDE_SIGMA, mode="constant", truncate=4.0)
    smooth /= smooth.sum()
    centers = [0.5 * (edge[:-1] + edge[1:]) for edge in edges]
    widths = np.array([np.diff(edge).mean() for edge in edges])
    indices = np.column_stack([
        np.clip(np.digitize(xyz[:, axis], edges[axis]) - 1, 0, HIST_BINS - 1)
        for axis in range(3)])
    assigned = points.copy()
    assigned[["tau_bin", "mse_bin", "crps_bin"]] = indices
    assigned["raw_bin_count"] = counts[indices[:, 0], indices[:, 1], indices[:, 2]].astype(int)
    assigned["kde_probability"] = smooth[indices[:, 0], indices[:, 1], indices[:, 2]]
    eligible = assigned["tau"].to_numpy(float) > 0 if require_positive_tau else np.ones(len(assigned), bool)
    if not np.any(eligible):
        raise ValueError("No eligible positive-tau trial exists for representative selection")
    eligible_bins = np.unique(indices[eligible], axis=0)
    peak_prob = max(float(smooth[tuple(idx)]) for idx in eligible_bins)
    peaks = [idx for idx in eligible_bins if np.isclose(smooth[tuple(idx)], peak_prob, rtol=0, atol=1e-15)]
    peaks.sort(key=lambda idx: (-centers[0][idx[0]], centers[1][idx[1]], centers[2][idx[2]]))
    peak_idx = np.asarray(peaks[0], int)
    members = assigned.loc[np.all(indices == peak_idx, axis=1) & eligible].copy()
    peak_center = np.array([centers[a][peak_idx[a]] for a in range(3)])
    members["distance_to_kde_peak_bin_center"] = np.linalg.norm(
        (members[["tau", "mse", "crps"]].to_numpy(float) - peak_center) / widths, axis=1)
    selected = members.sort_values(
        ["distance_to_kde_peak_bin_center", "tau", "mse", "crps", "trial"],
        ascending=[True, False, True, True, True], kind="mergesort").iloc[0].copy()
    selected["selected_bin_kde_probability"] = peak_prob
    selected["selected_bin_raw_count"] = int(counts[tuple(peak_idx)])
    rows = [{
        "tau_bin": i, "mse_bin": j, "crps_bin": k,
        "tau_center": centers[0][i], "mse_center": centers[1][j], "crps_center": centers[2][k],
        "raw_count": int(counts[i, j, k]), "raw_probability": counts[i, j, k] / N_TRIALS,
        "kde_probability": smooth[i, j, k],
    } for i in range(HIST_BINS) for j in range(HIST_BINS) for k in range(HIST_BINS)]
    return assigned, pd.DataFrame(rows), selected, {
        "kde_probability_sum": float(smooth.sum()), "peak_probability": peak_prob,
        "peak_raw_count": int(counts[tuple(peak_idx)]),
        "representative_tau_constraint": "tau > 0" if require_positive_tau else "none",
    }

