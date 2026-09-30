from pathlib import Path

import json

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize

import numpy as np

import pandas as pd

from scipy.ndimage import gaussian_filter

NBINS=20

def selected_point(points: pd.DataFrame, trial: int) -> pd.Series:
    rows = points.loc[points["trial"].eq(trial)]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row for trial {trial}, found {len(rows)}")
    return rows.iloc[0]

def add_axis_histograms(ax, frame: pd.DataFrame, limits) -> None:
    """Draw normalized 1-D histograms on the floor and rear wall of a 3-D axis."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = limits
    xspan, yspan, zspan = xmax - xmin, ymax - ymin, zmax - zmin
    values = [
        frame["tau"].to_numpy(float),
        frame["mse"].to_numpy(float),
        frame["crps"].to_numpy(float),
    ]
    ranges = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    histograms = [np.histogram(v, bins=NBINS, range=r) for v, r in zip(values, ranges)]

    # Kendall tau extends outward from y-min.
    counts, edges = histograms[0]
    height = counts / max(counts.max(), 1) * (0.17 * yspan)
    tau_y = ymin - height
    ax.bar3d(
        edges[:-1], tau_y, np.full(NBINS, zmin),
        np.diff(edges) * 0.92, height, np.full(NBINS, 0.010 * zspan),
        color=colors[0], alpha=0.58, shade=False, edgecolor="none",
        zsort="min", clip_on=False,
    )

    # MSE uses the original placement, extending inward from x-max.
    counts, edges = histograms[1]
    height = counts / max(counts.max(), 1) * (0.17 * xspan)
    ax.bar3d(
        xmax - height, edges[:-1], np.full(NBINS, zmin),
        height, np.diff(edges) * 0.92, np.full(NBINS, 0.010 * zspan),
        color=colors[1], alpha=0.58, shade=False, edgecolor="none",
        zsort="min", clip_on=False,
    )

    # CRPS remains attached to the tick-number edge and extends inward in x.
    counts, edges = histograms[2]
    height = counts / max(counts.max(), 1) * (0.17 * xspan)
    ax.bar3d(
        xmax - height, np.full(NBINS, ymax - 0.010 * yspan), edges[:-1],
        height, np.full(NBINS, 0.010 * yspan), np.diff(edges) * 0.92,
        color=colors[2], alpha=0.58, shade=False, edgecolor="none",
        zsort="min", clip_on=False,
    )

    # Shared CEMC-coordinate baselines keep all three marginal histograms
    # visibly attached to their corresponding axes even when an end bin is empty.
    baseline = {"color": "black", "linewidth": 1.05, "alpha": 1.0, "zorder": 1900}
    ax.plot([xmin, xmax], [ymin, ymin], [zmin, zmin], **baseline)
    ax.plot([xmax, xmax], [ymin, ymax], [zmin, zmin], **baseline)
    ax.plot([xmax, xmax], [ymax, ymax], [zmin, zmax], **baseline)

def remove_grid(ax) -> None:
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"]["linewidth"] = 0
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0, 0, 0, 1))
        axis.line.set_color("black")
        axis.line.set_linewidth(1.2)

def draw_black_cube(ax, limits) -> None:
    """Draw all 12 edges of the plotting box as an unambiguous black cube."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = limits
    edge_style = {"color": "black", "linewidth": 1.15, "alpha": 1.0, "zorder": 2000}
    for y in (ymin, ymax):
        for z in (zmin, zmax):
            ax.plot([xmin, xmax], [y, y], [z, z], **edge_style)
    for x in (xmin, xmax):
        for z in (zmin, zmax):
            ax.plot([x, x], [ymin, ymax], [z, z], **edge_style)
    for x in (xmin, xmax):
        for y in (ymin, ymax):
            ax.plot([x, x], [y, y], [zmin, zmax], **edge_style)

def draw_selected_axis_guides(ax, selected: pd.Series, limits) -> None:
    """Connect the selected star to its corresponding value on all three axes."""
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = limits
    x = float(selected["tau"])
    y = float(selected["mse"])
    z = float(selected["crps"])
    guide_style = {
        "color": "black", "linestyle": "--", "linewidth": 1.35,
        "alpha": 0.88, "zorder": 1450,
    }
    # Endpoints follow the exact visible axes used by the marginal histograms:
    # tau=(x, y-min, z-min), MSE=(x-max, y, z-min),
    # CRPS=(x-max, y-max, z).
    ax.plot([x, x], [y, ymin], [z, zmin], **guide_style)
    ax.plot([x, xmax], [y, y], [z, zmin], **guide_style)
    ax.plot([x, xmax], [y, ymax], [z, z], **guide_style)

def draw_inward_tau_ticks(ax, limits) -> None:
    """Draw tau ticks geometrically inward along +MSE on the cube floor."""
    (xmin, xmax), (ymin, ymax), (_, zmax) = limits
    zmin = limits[2][0]
    tick_end = ymin + 0.028 * (ymax - ymin)
    for x in ax.get_xticks():
        if xmin <= x <= xmax:
            ax.plot(
                [x, x], [ymin, tick_end], [zmin, zmin],
                color="black", linewidth=1.15, alpha=1.0, zorder=2050,
            )

def plot_hist(data_dir, output_dir, ml: int) -> None:
    outdir = data_dir
    xlabel_pad, ylabel_pad, zlabel_pad = (22, 22, 30)
    cpoints = pd.read_csv(outdir / "cemc_10000_trial_points_with_kde_probability.csv")
    rpoints = pd.read_csv(outdir / "random_10000_trial_points_with_kde_probability.csv")
    summary = pd.read_csv(outdir / "selected_kde_representative_trials.csv").iloc[0]
    csel = selected_point(cpoints, int(summary["cemc_trial"]))
    rsel = selected_point(rpoints, int(summary["random_trial"]))
    frames = [("CEMC", cpoints, csel), ("Homogeneous", rpoints, rsel)]

    probabilities = np.concatenate([
        frame["kde_probability"].to_numpy(float) for _, frame, _ in frames
    ])
    norm = Normalize(vmin=float(probabilities.min()), vmax=float(probabilities.max()))

    # Match both panels to the original convention: CEMC-derived limits and ticks.
    cxyz = cpoints[["tau", "mse", "crps"]].to_numpy(float)
    limits = []
    for axis_index, values in enumerate(cxyz.T):
        lo, hi = float(values.min()), float(values.max())
        span = hi - lo
        lower_pad = (0.12 if axis_index == 1 else 0.05) * span
        upper_pad = 0.05 * span
        upper_limit = hi + upper_pad
        if axis_index == 0:
            upper_limit = max(0.9, upper_limit)
        limits.append((lo - lower_pad, upper_limit))

    with plt.rc_context({
        "font.size": 18,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    }):
        fig = plt.figure(figsize=(22, 9.5))
        grid = fig.add_gridspec(
            1, 3, width_ratios=(1, 1, 0.035),
            left=0.025, right=0.875, bottom=0.07, top=0.96, wspace=0.10,
        )
        axes = [fig.add_subplot(grid[0, i], projection="3d") for i in range(2)]
        scatter = None
        for ax, (method, frame, selected) in zip(axes, frames):
            axis_title_size = 30
            scatter = ax.scatter(
                frame["tau"], frame["mse"], frame["crps"],
                c=frame["kde_probability"], cmap="viridis", norm=norm,
                s=7, alpha=0.42, linewidths=0, rasterized=True, depthshade=False,
            )
            add_axis_histograms(ax, frame, limits)
            draw_selected_axis_guides(ax, selected, limits)
            ax.plot(
                [selected["tau"]], [selected["mse"]], [selected["crps"]],
                linestyle="none", marker="*", markersize=17,
                markerfacecolor="red", markeredgecolor="black",
                markeredgewidth=1.2, zorder=1000,
            )
            ax.set_xlim3d(limits[0])
            ax.set_ylim3d(limits[1])
            ax.set_zlim3d(limits[2])
            ax.set_box_aspect((1, 1, 1))
            ax.set_xlabel(
                r"Kendall $\tau_b$", labelpad=xlabel_pad, fontsize=axis_title_size
            )
            ax.set_ylabel("MSE", labelpad=ylabel_pad, fontsize=axis_title_size)
            current_zlabel_pad = 38
            ax.set_zlabel(
                "CRPS", labelpad=current_zlabel_pad, fontsize=axis_title_size
            )
            ax.zaxis.label.set_clip_on(False)
            # Hide the ambiguous projected default tau tick lines. Explicit
            # inward 3-D ticks are drawn after both axes share final ticks.
            tau_tick_pad = -22
            ax.tick_params(axis="x", pad=tau_tick_pad, length=0, labelsize=20)
            ax.tick_params(axis="y", labelsize=20)
            ax.xaxis._axinfo["tick"]["inward_factor"] = 0.0
            ax.xaxis._axinfo["tick"]["outward_factor"] = 0.0
            ax.tick_params(axis="z", pad=8, labelsize=20)
            # The panel names are added later as 2-D text just above each cube.
            ax.set_title("")
            remove_grid(ax)
            draw_black_cube(ax, limits)

        # Make the Homogeneous panel geometrically identical to the CEMC panel.
        # This explicitly synchronizes everything that a 3-D Axes may otherwise
        # derive independently from its own artists.
        cemc_ax, homogeneous_ax = axes
        homogeneous_ax.set_xlim3d(cemc_ax.get_xlim3d())
        homogeneous_ax.set_ylim3d(cemc_ax.get_ylim3d())
        homogeneous_ax.set_zlim3d(cemc_ax.get_zlim3d())
        homogeneous_ax.set_xticks(cemc_ax.get_xticks())
        homogeneous_ax.set_yticks(cemc_ax.get_yticks())
        homogeneous_ax.set_zticks(cemc_ax.get_zticks())
        homogeneous_ax.set_box_aspect(cemc_ax.get_box_aspect())
        homogeneous_ax.view_init(
            elev=cemc_ax.elev,
            azim=cemc_ax.azim,
            roll=getattr(cemc_ax, "roll", 0),
            vertical_axis="z",
        )
        cpos = cemc_ax.get_position()
        hpos = homogeneous_ax.get_position()
        homogeneous_shift = 0.018
        homogeneous_ax.set_position([
            hpos.x0 + homogeneous_shift, cpos.y0, cpos.width, cpos.height
        ])
        for ax in axes:
            draw_inward_tau_ticks(ax, limits)
        # Freeze both panels after the CEMC values have been copied. No artist
        # added during rendering is allowed to alter either panel's limits.
        for ax in axes:
            ax.set_autoscale_on(False)
            ax.set_xlim3d(cemc_ax.get_xlim3d(), auto=False)
            ax.set_ylim3d(cemc_ax.get_ylim3d(), auto=False)
            ax.set_zlim3d(cemc_ax.get_zlim3d(), auto=False)

        # Place names outside the 3-D boxes but immediately above their top edges.
        for ax, method in zip(axes, ("CEMC", "Homogeneous")):
            panel_title_size = 36
            ax.text2D(
                0.5, 0.955, method, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=panel_title_size, zorder=4000,
            )

        cax = fig.add_subplot(grid[0, 2])
        pos = cax.get_position()
        colorbar_x_shift = 0.045
        cax.set_position([pos.x0 + colorbar_x_shift, pos.y0, pos.width, pos.height])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_label("Probability", fontsize=32)
        cbar.ax.tick_params(labelsize=24)

        stem = output_dir / "histogram"
        fig.savefig(stem.with_suffix(".png"), dpi=250, facecolor="white")
        fig.savefig(stem.with_suffix(".pdf"), dpi=250, facecolor="white")
        plt.close(fig)

    metadata = {
        "ml": ml,
        "source_points": {
            "cemc": "cemc_10000_trial_points_with_kde_probability.csv",
            "homogeneous": "random_10000_trial_points_with_kde_probability.csv",
        },
        "axis_histogram_bins": NBINS,
        "axis_histogram_scaling": "each marginal normalized independently to its maximum; visual height occupies 17% of adjacent axis span",
        "axis_histogram_placement": "Kendall tau outward; MSE and CRPS inward",
        "axis_tick_label_placement": "Kendall tau numbers at pad=-22; default tick lines hidden and custom ticks drawn inward along +MSE; MSE default outside; CRPS unchanged at pad=8",
        "mse_lower_axis_padding_fraction": 0.12,
        "tau_upper_axis_limit": 0.9,
        "typography": {
            "panel_title": 36,
            "axis_title": 30,
            "axis_tick_label": 20,
            "colorbar_title": 32,
            "colorbar_tick_label": 24,
            "colorbar_text": "Probability",
        },
        "ml4_layout_adjustments": {
            "cemc_crps_label_pad": 38,
            "homogeneous_crps_label_pad": 38,
            "homogeneous_panel_x_shift": 0.018,
            "colorbar_x_shift": 0.045,
            "kendall_tick_label_pad": -22,
        },
        "grid": "disabled",
        "figure_title": "removed",
        "legend": "removed",
        "box_aspect": "1:1:1",
        "box_edges": "12 black lines",
        "panel_geometry": "Homogeneous synchronized to CEMC limits, ticks, camera, box aspect, width, height, and vertical position",
        "panel_title_y": 0.955,
        "panel_title_placement": "2-D text outside and immediately above each 3-D cube",
        "autoscale": "disabled after CEMC limits and ticks are copied to Homogeneous",
        "marginal_histogram_attachment": "three shared CEMC-coordinate black baselines; CRPS attached to the x-max/y-max tick-number edge",
        "selected_point_axis_guides": "black dashed lines from each selected star to its tau, MSE, and CRPS number axes",
        "axis_label_padding": {
            "kendall_tau_b": xlabel_pad,
            "mse": ylabel_pad,
            "crps": 38,
        },
        "axis_limits": "CEMC-derived and shared by both panels",
        "selected_trials": {
            "cemc": int(summary["cemc_trial"]),
            "homogeneous": int(summary["random_trial"]),
        },
    }
    (output_dir / "histogram_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".pdf"))
