# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 13:53:43 2026

@author: user
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import os

dir_path = '../Data/'

EXP = json.load(open(os.path.join(dir_path, 'exp_scaled_activity_1D.json'), 'r'))
MC = json.load(open(os.path.join(dir_path, 'MC_activity_patch_1D.json'), 'r'))
RS = json.load(open(os.path.join(dir_path, 'RS_activity_patch_1D.json'), 'r'))


EXP = np.array(EXP)
MC = np.array(MC)
RS = np.array(RS)

sim = np.concatenate((MC,RS))
sim_min = sim.min()
sim_max = sim.max()
MC = (MC-sim_min)/(sim_max-sim_min)
RS = (RS-sim_min)/(sim_max-sim_min)

# MC = (MC-MC.min())/(MC.max()-MC.min())
# RS = (RS-RS.min())/(RS.max()-RS.min())

# ========= USER INPUT =========
# EXP, MC, RS should be 1D arrays/vectors of the same length
# Example:
# EXP = np.array([...], dtype=float)
# MC  = np.array([...], dtype=float)   # annealed
# RS  = np.array([...], dtype=float)   # random shuffle / homogeneous
# ==============================

def _as_1d_float(x, name):
    x = np.asarray(x, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError(f"{name} is empty.")
    return x

def _mask_finite(*arrs):
    mask = np.ones(arrs[0].shape[0], dtype=bool)
    for a in arrs:
        mask &= np.isfinite(a)
    return mask

def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return np.nan if ss_tot == 0 else 1.0 - ss_res / ss_tot

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def pearson_r(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    denom = np.sqrt(np.sum(yt**2) * np.sum(yp**2))
    return np.nan if denom == 0 else float(np.sum(yt*yp) / denom)

def _rankdata(a):
    """Average ranks for ties, like scipy.stats.rankdata(method='average')."""
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, a.size + 1, dtype=float)
    # handle ties
    sorted_a = a[order]
    i = 0
    while i < a.size:
        j = i
        while j + 1 < a.size and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        if j > i:
            avg = np.mean(ranks[order][i:j+1])
            ranks[order][i:j+1] = avg
        i = j + 1
    return ranks

def spearman_rho(y_true, y_pred):
    rt = _rankdata(y_true)
    rp = _rankdata(y_pred)
    return pearson_r(rt, rp)

def kendall_tau_b(y_true, y_pred):
    """
    Kendall's tau-b without scipy.
    O(n^2) - OK for n up to a few thousand.
    """
    x = np.asarray(y_true, dtype=float)
    y = np.asarray(y_pred, dtype=float)
    n = x.size
    conc = disc = 0
    tie_x = tie_y = tie_xy = 0
    for i in range(n-1):
        dx = x[i+1:] - x[i]
        dy = y[i+1:] - y[i]
        sx = np.sign(dx)
        sy = np.sign(dy)
        # ties
        tx = (sx == 0)
        ty = (sy == 0)
        tie_x += int(np.sum(tx & ~ty))
        tie_y += int(np.sum(ty & ~tx))
        tie_xy += int(np.sum(tx & ty))
        # concordant/discordant
        prod = sx * sy
        conc += int(np.sum(prod > 0))
        disc += int(np.sum(prod < 0))
    denom = np.sqrt((conc + disc + tie_x) * (conc + disc + tie_y))
    if denom == 0:
        return np.nan
    return float((conc - disc) / denom)

def topk_metrics(y_true, y_pred, k_frac=0.10, higher_is_better=True):
    """
    Compare overlap of top-k fraction between exp and prediction.
    Returns: recall, precision, jaccard, overlap_count, k
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    n = y_true.size
    k = max(1, int(np.ceil(k_frac * n)))

    # ranks: choose top values if higher_is_better else bottom values
    if higher_is_better:
        idx_true = np.argsort(-y_true)[:k]
        idx_pred = np.argsort(-y_pred)[:k]
    else:
        idx_true = np.argsort(y_true)[:k]
        idx_pred = np.argsort(y_pred)[:k]

    set_true = set(idx_true.tolist())
    set_pred = set(idx_pred.tolist())
    inter = set_true & set_pred
    union = set_true | set_pred

    overlap = len(inter)
    recall = overlap / len(set_true)
    precision = overlap / len(set_pred)
    jaccard = overlap / len(union)
    return recall, precision, jaccard, overlap, k

def bin_trend(x, y, n_bins=5):
    """
    Bin x into quantiles and compute mean(y) and 95% CI via normal approx.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, qs)
    # make edges strictly increasing
    edges = np.unique(edges)
    bins = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i+1]
        m = (x >= lo) & (x <= hi if i == len(edges)-2 else x < hi)
        if np.sum(m) < 3:
            continue
        yy = y[m]
        mu = float(np.mean(yy))
        se = float(np.std(yy, ddof=1) / np.sqrt(yy.size))
        ci95 = 1.96 * se
        bins.append((float(lo), float(hi), yy.size, mu, ci95))
    return bins

def evaluate_model(EXP, PRED, name, higher_is_better=True):
    mask = _mask_finite(EXP, PRED)
    y = EXP[mask]
    p = PRED[mask]
    out = {}
    out["n"] = int(y.size)
    out["R2"] = float(r2_score(y, p))
    out["MAE"] = float(mae(y, p))
    out["RMSE"] = float(rmse(y, p))
    out["Pearson_r"] = float(pearson_r(y, p))
    out["Spearman_rho"] = float(spearman_rho(y, p))
    out["Kendall_tau_b"] = float(kendall_tau_b(y, p))
    for kf in [0.05, 0.10, 0.20]:
        rec, prec, jac, ov, k = topk_metrics(y, p, k_frac=kf, higher_is_better=higher_is_better)
        out[f"Top{int(kf*100)}%_recall"] = rec
        out[f"Top{int(kf*100)}%_precision"] = prec
        out[f"Top{int(kf*100)}%_jaccard"] = jac
        out[f"Top{int(kf*100)}%_overlap"] = ov
        out[f"Top{int(kf*100)}%_k"] = k
    return name, out

def print_report(results):
    # results: list of (name, dict)
    keys = ["n", "R2", "MAE", "RMSE", "Pearson_r", "Spearman_rho", "Kendall_tau_b",
            "Top5%_jaccard", "Top10%_jaccard", "Top20%_jaccard",
            "Top10%_overlap", "Top10%_k"]
    # pretty print
    print("\n=== Model Evaluation Summary ===")
    header = "Metric".ljust(18) + "".join([name.rjust(14) for name, _ in results])
    print(header)
    print("-" * len(header))
    for k in keys:
        row = k.ljust(18)
        for _, d in results:
            v = d.get(k, np.nan)
            if isinstance(v, float):
                row += f"{v:14.4f}"
            else:
                row += f"{v:14}"
        print(row)

def compare_error_vs_sro(EXP, RS, MC, SRO=None, higher_is_better=True):
    """
    Optional: if you have SRO per data point/state, check if random-model error increases
    with distance from SRO* (annealed). If SRO is None, this does nothing.
    """
    if SRO is None:
        return
    SRO = _as_1d_float(SRO, "SRO")
    mask = _mask_finite(EXP, RS, MC, SRO)
    y = EXP[mask]; rs = RS[mask]; mc = MC[mask]; sro = SRO[mask]
    # Define SRO* as the max SRO among these points (or you can replace with known annealed value)
    sro_star = float(np.max(sro))
    dist = np.abs(sro - sro_star)

    err_rs = np.abs(rs - y)
    err_mc = np.abs(mc - y)
    delta_err = err_rs - err_mc  # positive means MC better

    print("\n=== Optional SRO Trend Check ===")
    print(f"SRO* (used) = {sro_star:.4f}")
    print(f"Spearman(dist, delta_err) = {spearman_rho(dist, delta_err):.4f}  (positive = farther from SRO* => bigger MC gain)")
    bins = bin_trend(dist, delta_err, n_bins=5)
    if bins:
        print("Binned (dist range, n, mean(delta_err), ±95%CI)")
        for lo, hi, n, mu, ci in bins:
            print(f"  [{lo:.4f}, {hi:.4f}]  n={n:4d}  mean={mu:.4f}  ±{ci:.4f}")
    else:
        print("Not enough data for binning.")

# ======= RUN =======
EXP = _as_1d_float(EXP, "EXP")
MC  = _as_1d_float(MC,  "MC")
RS  = _as_1d_float(RS,  "RS")
if not (EXP.size == MC.size == RS.size):
    raise ValueError(f"Length mismatch: EXP={EXP.size}, MC={MC.size}, RS={RS.size}")

# IMPORTANT: set this correctly for your activity definition
# If lower overpotential = better activity, set higher_is_better=False
higher_is_better = True

results = []
results.append(evaluate_model(EXP, MC, "Annealed(MC)", higher_is_better=higher_is_better))
results.append(evaluate_model(EXP, RS, "Random(RS)",  higher_is_better=higher_is_better))

print_report(results)

# Optional: if you have SRO per entry, pass it here to check Fig4-style trend
# SRO = np.array([...], dtype=float)
# compare_error_vs_sro(EXP, RS, MC, SRO=SRO, higher_is_better=higher_is_better)

def rankdata(a):
    temp = np.argsort(a)
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(a))
    return ranks

# If LOWER activity is better (overpotential), use EXP directly.
# If HIGHER is better, use -EXP
exp_rank = rankdata(EXP)
mc_rank  = rankdata(MC)
rs_rank  = rankdata(RS)

# Figure 1f
fig, ax = plt.subplots(figsize=(8,8))
hb = ax.hexbin(exp_rank, rs_rank, gridsize=15, cmap='Blues', edgecolors='none', extent=[-50,1550,-50,1550])
ax.plot([0,1500],[0,1500], linewidth=2.5, c='k')

ax.set_xlim(-50, 1550)
ax.set_ylim(-50, 1550)

ax.set_xticks(np.linspace(0, 1500, 4))
ax.set_yticks(np.linspace(0, 1500, 4))
ax.tick_params(
    axis='both', 
    direction='in',
    length=7,
    width=2.5,
    labelbottom=False,
    labelleft=False
)

for spine in ax.spines.values():
    spine.set_zorder(10)
    spine.set_linewidth(2.5)

# plt.savefig('ranking_plot_RS.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 1e
fig, ax = plt.subplots(figsize=(8,8))
hb = ax.hexbin(exp_rank, mc_rank, gridsize=15, cmap='Blues', edgecolors='none', extent=[-50,1550,-50,1550])
ax.plot([0,1500],[0,1500], linewidth=2.5, c='k')

ax.set_xlim(-50, 1550)
ax.set_ylim(-50, 1550)

ax.set_xticks(np.linspace(0, 1500, 4))
ax.set_yticks(np.linspace(0, 1500, 4))
ax.tick_params(
    axis='both', 
    direction='in',
    length=7,
    width=2.5,
    labelbottom=False,
    labelleft=False
)

for spine in ax.spines.values():
    spine.set_zorder(10)
    spine.set_linewidth(2.5)

# plt.savefig('ranking_plot_MC.png', dpi=300, bbox_inches='tight')
plt.show()

# Figure 1d
def topk_recall_curve(EXP, PRED, n_points=60, higher_is_better=True, k_max=0.6):
    """
    Continuous top-k recall curve.
    Recall(k) = |Top-k_EXP ∩ Top-k_PRED| / |Top-k_EXP|
    """
    EXP = np.asarray(EXP, float)
    PRED = np.asarray(PRED, float)
    n = len(EXP)

    ks = np.linspace(0.01, k_max, n_points)
    recall = []

    for k in ks:
        m = int(np.ceil(k * n))

        if higher_is_better:
            top_exp = set(np.argsort(-EXP)[:m])
            top_pred = set(np.argsort(-PRED)[:m])
        else:
            top_exp = set(np.argsort(EXP)[:m])
            top_pred = set(np.argsort(PRED)[:m])

        recall.append(len(top_exp & top_pred) / m)

    return ks, np.array(recall)

# ---------------- USER SETTING ----------------
# True if larger EXP = more active
# False if smaller EXP = more active (e.g., overpotential)
higher_is_better = True
# ----------------------------------------------

# Compute curves
k_mc, recall_mc = topk_recall_curve(EXP, MC, higher_is_better=higher_is_better)
k_rs, recall_rs = topk_recall_curve(EXP, RS, higher_is_better=higher_is_better)

# Random baseline: expected recall = k
baseline = k_mc

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(7.8, 6))
ax.plot(k_rs * 100, recall_rs, lw=2.0, label="Homogeneous (RS)")
ax.plot(k_mc * 100, recall_mc, lw=2.0, label="Annealed (MC)")
ax.plot(k_mc * 100, baseline, 'k--', lw=2.0, label="Random baseline (recall = k)")

# ax.set_xlabel("Top fraction tested (%)")
# ax.set_ylabel("Recall of experimental active region")
ax.set_xlim(0, k_mc.max() * 100)
ax.set_ylim(0, 1.0)
ax.fill_between(k_mc*100, baseline, 0,
                 color='gray', alpha=0.08)
ax.legend(frameon=False)

ax.tick_params(
    axis='both', 
    direction='in',
    length=7,
    width=2.0,
    labelbottom=False,
    labelleft=False
)

for spine in ax.spines.values():
    spine.set_zorder(10)
    spine.set_linewidth(2.0)

# plt.tight_layout()
plt.show()
# plt.savefig('recall_fraction.png', dpi=300, bbox_inches='tight')