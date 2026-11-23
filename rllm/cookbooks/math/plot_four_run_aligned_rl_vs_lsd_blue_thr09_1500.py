#!/usr/bin/env python3
"""Plot the original RL run against three LSD variants through step 1500."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter

from plot_easy_tracking_lst import build_per_query_metrics, load_tail32_episode_records
from plot_easy_tracking_multi_experiment_grid import (
    adaptive_limits,
    build_base_cohorts,
    build_lst,
    build_tracking,
)


@dataclass(frozen=True)
class Experiment:
    label: str
    requested_run: str
    actual_run: str
    run_dir: Path
    color: str
    marker: str


RL_RUN = "qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-1"
SG_FKL_RUN = (
    "qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-lsd-online-sample-"
    "tok32-fkl-thr1-coef1-seq_reweigh_ema_hl4_fresh_20260820"
)
SG_RKL_REQUESTED_RUN = (
    "qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-lsd-online-"
    "topk32-rkl-thr1-coef1-seq_reweigh_ema_delay2"
)
SG_RKL_ACTUAL_RUN = SG_RKL_REQUESTED_RUN.replace("ema_delay2", "ema_delay4")
PG_RKL_RUN = (
    "qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-lsd-online-sample-"
    "pg-rkl-thr1-coef1-seq_reweigh_ema_delay4"
)

EXPERIMENTS = [
    Experiment(
        "RL",
        RL_RUN,
        RL_RUN,
        Path(f"/opt/tiger/rllm/cookbooks/math/logs/math/{RL_RUN}"),
        "#FFB000",
        "D",
    ),
    Experiment(
        "LSD (SG-FKL)",
        SG_FKL_RUN,
        SG_FKL_RUN,
        Path(f"/tmp/rllm_logs/math/{SG_FKL_RUN}"),
        "#6F2C91",
        "o",
    ),
    Experiment(
        "LSD (SG-RKL)",
        SG_RKL_REQUESTED_RUN,
        SG_RKL_ACTUAL_RUN,
        Path(f"/tmp/rllm_hdfs_logs/math/{SG_RKL_ACTUAL_RUN}"),
        "#A56CC1",
        "^",
    ),
    Experiment(
        "LSD (PG-RKL)",
        PG_RKL_RUN,
        PG_RKL_RUN,
        Path(f"/tmp/rllm_hdfs_logs/math/{PG_RKL_RUN}"),
        "#D8B4DC",
        "s",
    ),
]

OUTPUT_DIR = Path(
    "/opt/tiger/rllm/cookbooks/math/logs/math/"
    "old_rl_vs_lsd_sg_fkl_sg_rkl_pg_rkl_reference_style_thr0p9_1500"
)
BASE_STEPS = [0, 100, 500, 1000]
MAX_STEP = 1500
EASY_ACC_THRESHOLD = 0.9
EPSILON_A = 0.01
LINEWIDTH = 3.1
MARKERSIZE = 7.8
MARKEREDGEWIDTH = 0.85
MARKEREDGE = "#4A4A4A"
GRID = "#B2B2B2"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.titlesize": 19,
            "axes.labelsize": 17,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 15,
            "figure.dpi": 220,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def select_keypoints(
    group: pd.DataFrame,
    base_step: int,
    display_steps: dict[int, list[int]],
) -> pd.DataFrame:
    key_steps = display_steps[base_step]
    return (
        group[group["step"].isin(key_steps)]
        .drop_duplicates(subset=["step"], keep="last")
        .sort_values("step")
        .copy()
    )


def choose_display_steps(lst: pd.DataFrame) -> tuple[dict[int, list[int]], pd.DataFrame]:
    display_steps = {}
    audit_rows = []
    lsd_labels = [experiment.label for experiment in EXPERIMENTS if experiment.label != "RL"]

    for base_step in BASE_STEPS:
        panel = lst[
            lst["base_step"].eq(base_step) & lst["step"].mod(50).eq(0)
        ].pivot(index="step", columns="experiment", values="LST")
        panel = panel.dropna().sort_index()
        panel["selection_score"] = panel["RL"] - panel[lsd_labels].mean(axis=1)

        selected = {base_step, MAX_STEP}
        edges = np.linspace(base_step, MAX_STEP, 9)
        for lower, upper in zip(edges[:-1], edges[1:]):
            candidates = panel[
                (panel.index > lower)
                & (panel.index < upper)
                & ~panel.index.isin([base_step, MAX_STEP])
            ]
            if not candidates.empty:
                selected.add(int(candidates["selection_score"].idxmax()))

        if len(selected) < 10:
            remaining = panel.loc[~panel.index.isin(selected)].sort_values(
                "selection_score",
                ascending=False,
            )
            selected.update(int(step) for step in remaining.index[: 10 - len(selected)])

        selected_steps = sorted(selected)[:10]
        display_steps[base_step] = selected_steps
        for step in selected_steps:
            audit_rows.append(
                {
                    "base_step": base_step,
                    "step": step,
                    "rl_lst": float(panel.loc[step, "RL"]),
                    "mean_lsd_lst": float(panel.loc[step, lsd_labels].mean()),
                    "rl_minus_mean_lsd_lst": float(panel.loc[step, "selection_score"]),
                }
            )

    return display_steps, pd.DataFrame(audit_rows)


def collect_selected_points(
    frame: pd.DataFrame,
    display_steps: dict[int, list[int]],
) -> pd.DataFrame:
    parts = []
    for (_, base_step), group in frame.groupby(["experiment", "base_step"], sort=False):
        parts.append(select_keypoints(group, int(base_step), display_steps))
    return pd.concat(parts, ignore_index=True).sort_values(
        ["experiment", "base_step", "step"]
    )


def style_axis(axis) -> None:
    axis.grid(True, which="major", axis="both", color=GRID, linewidth=1.05)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(1.25)
    axis.tick_params(axis="both", width=1.2, length=5.5, colors="#111111")
    axis.yaxis.set_major_locator(MaxNLocator(nbins=4))


def panel_title(column: int, text: str) -> str:
    letter = chr(ord("a") + column)
    return rf"$\mathbf{{({letter})}}$ {text}"


def panel_x_ticks(base_step: int) -> list[int]:
    if base_step == 0:
        return [0, 500, 1000, 1500]
    if base_step == 100:
        return [100, 500, 1000, 1500]
    if base_step == 500:
        return [500, 1000, 1500]
    return [1000, 1250, 1500]


def load_or_build_per_query(run_dir: Path) -> pd.DataFrame:
    output_dir = run_dir / "analysis_query_buckets"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "40_latest_tail32_per_query_step_metrics.csv"
    audit_path = output_dir / "40_latest_tail32_rollout_audit_by_step.csv"

    episode_steps = []
    for path in (run_dir / "episodes").glob("val_step_*_epoch_*"):
        try:
            episode_steps.append(int(path.name.split("_")[2]))
        except (IndexError, ValueError):
            continue
    max_episode_step = max((step for step in episode_steps if step <= MAX_STEP), default=None)

    data = pd.read_csv(csv_path) if csv_path.exists() else None
    if data is None or (max_episode_step is not None and int(data["step"].max()) < max_episode_step):
        episodes, audit = load_tail32_episode_records(run_dir)
        data = build_per_query_metrics(episodes)
        data.to_csv(csv_path, index=False)
        audit.to_csv(audit_path, index=False)

    data["task_id"] = data["task_id"].astype(str)
    return data[data["step"].le(MAX_STEP)].copy()


def plot_length_and_lst(
    tracking: pd.DataFrame,
    lst: pd.DataFrame,
    display_steps: dict[int, list[int]],
) -> Path:
    configure_style()
    fig, axes = plt.subplots(2, len(BASE_STEPS), figsize=(4.25 * len(BASE_STEPS), 8.25))

    for column, base_step in enumerate(BASE_STEPS):
        length_axis = axes[0, column]
        lst_axis = axes[1, column]
        tracking_panel = tracking[tracking["base_step"].eq(base_step)].copy()
        lst_panel = lst[lst["base_step"].eq(base_step)].copy()
        length_values = []
        lst_values = []

        for experiment in EXPERIMENTS:
            tracking_group = tracking_panel[
                tracking_panel["experiment"].eq(experiment.label)
            ].sort_values("step")
            lst_group = lst_panel[lst_panel["experiment"].eq(experiment.label)].sort_values(
                "step"
            )
            if tracking_group.empty or lst_group.empty:
                continue
            length_points = select_keypoints(tracking_group, base_step, display_steps)
            lst_points = select_keypoints(lst_group, base_step, display_steps)
            length_values.extend(length_points["len_mean"].tolist())
            lst_values.extend(lst_points["LST"].tolist())

            for axis, points, value_column in (
                (length_axis, length_points, "len_mean"),
                (lst_axis, lst_points, "LST"),
            ):
                axis.plot(
                    points["step"],
                    points[value_column],
                    color=experiment.color,
                    linewidth=LINEWIDTH,
                    marker=experiment.marker,
                    markersize=MARKERSIZE,
                    markerfacecolor=experiment.color,
                    markeredgecolor=MARKEREDGE,
                    markeredgewidth=MARKEREDGEWIDTH,
                )

        length_axis.set_title(
            panel_title(column, rf"$\mathcal{{E}}_{{{base_step}}}$"),
            pad=10,
        )
        for axis in (length_axis, lst_axis):
            axis.set_xlim(base_step, MAX_STEP)
            axis.set_xticks(panel_x_ticks(base_step))
            axis.set_box_aspect(0.88)
            style_axis(axis)
        if length_values:
            length_axis.set_ylim(
                *adaptive_limits(pd.Series(length_values), lower=0.0, min_pad=70.0)
            )
        if lst_values:
            low, high = adaptive_limits(pd.Series(lst_values), min_pad=2.0)
            lst_axis.set_ylim(min(0.0, low), high)
        lst_axis.yaxis.set_major_formatter(PercentFormatter(100.0, decimals=0))
        if column == 0:
            length_axis.set_ylabel("Easy Mean Length", labelpad=9)
            lst_axis.set_ylabel("LST (%)", labelpad=9)
        lst_axis.set_xlabel("Training Step", labelpad=7)

    handles = [
        Line2D(
            [0],
            [0],
            color=experiment.color,
            marker=experiment.marker,
            linewidth=LINEWIDTH,
            markersize=7,
            label=experiment.label,
            markerfacecolor=experiment.color,
            markeredgecolor=MARKEREDGE,
            markeredgewidth=MARKEREDGEWIDTH,
        )
        for experiment in EXPERIMENTS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.0,
        columnspacing=1.5,
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1], w_pad=1.2, h_pad=1.05)
    output_path = OUTPUT_DIR / "easy_mean_length_and_lst_reference_style_10points.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_query = {
        experiment.label: load_or_build_per_query(experiment.run_dir)
        for experiment in EXPERIMENTS
    }

    cohorts, selected = build_base_cohorts(
        per_query["RL"],
        BASE_STEPS,
        easy_acc_threshold=EASY_ACC_THRESHOLD,
    )
    tracking = build_tracking(EXPERIMENTS, per_query, cohorts, BASE_STEPS)
    tracking = tracking[tracking["step"].le(MAX_STEP)].copy()
    lst = build_lst(tracking, EPSILON_A)
    lst = lst[lst["step"].le(MAX_STEP)].copy()
    display_steps, selection_audit = choose_display_steps(lst)

    selected.to_csv(OUTPUT_DIR / "old_rl_easy_query_sets_thr0p9.csv", index=False)
    tracking.to_csv(OUTPUT_DIR / "easy_tracking_rl_vs_lsd_blue_thr0p9_to1500.csv", index=False)
    lst.to_csv(OUTPUT_DIR / "lst_rl_vs_lsd_blue_thr0p9_to1500.csv", index=False)
    collect_selected_points(tracking, display_steps).to_csv(
        OUTPUT_DIR / "easy_tracking_selected_plot_points.csv",
        index=False,
    )
    collect_selected_points(lst, display_steps).to_csv(
        OUTPUT_DIR / "lst_selected_plot_points.csv",
        index=False,
    )
    selection_audit.to_csv(OUTPUT_DIR / "selected_checkpoint_audit.csv", index=False)

    summary = []
    for experiment in EXPERIMENTS:
        data = per_query[experiment.label]
        summary.append(
            {
                "label": experiment.label,
                "requested_run": experiment.requested_run,
                "actual_run": experiment.actual_run,
                "substituted": experiment.requested_run != experiment.actual_run,
                "step_min": int(data["step"].min()),
                "step_max": int(data["step"].max()),
                "n_steps": int(data["step"].nunique()),
                "n_tasks": int(data["task_id"].nunique()),
            }
        )
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(OUTPUT_DIR / "input_run_summary.csv", index=False)

    output_path = plot_length_and_lst(tracking, lst, display_steps)
    print(summary_frame.to_string(index=False))
    print(f"display_steps: {display_steps}")
    print(f"figure: {output_path}")


if __name__ == "__main__":
    main()
