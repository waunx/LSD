#!/usr/bin/env python3
"""Plot within-run LST for the current DeepResearch LSD validation episodes."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter


RUN_NAME = (
    "qwen3_8b_base_cutthebill_lsd_online_ema_direct_fkl_coef1_thr1_loss_reweigh_"
    "ema_delay2_g340-cd51-4900-2d65-9d6b-b6e4-6020"
)
RUN_DIR = Path("/tmp/deepresearch_lsd_logs/deepresearch") / RUN_NAME
OUTPUT_DIR = (
    Path("/opt/tiger/rllm/cookbooks/deepresearch/plots")
    / "current_lsd_within_run_lst"
)

BASE_STEPS = [0, 25, 50, 75, 100]
EASY_ACC_THRESHOLD = 0.9
EPSILON_A = 0.01
COLOR = "#2563EB"
GRID = "#B2B2B2"


def total_response_tokens(record: dict) -> float:
    return float(
        sum(
            float(step.get("response_length") or 0.0)
            for trajectory in record.get("trajectories") or []
            for step in trajectory.get("steps") or []
        )
    )


def load_per_query(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pattern = re.compile(r"val_step_(\d+)_epoch_(\d+)$")
    records: list[dict] = []
    audit_rows: list[dict] = []

    for step_dir in sorted(
        (run_dir / "episodes").glob("val_step_*_epoch_*"),
        key=lambda path: int(pattern.match(path.name).group(1))
        if pattern.match(path.name)
        else -1,
    ):
        match = pattern.match(step_dir.name)
        episode_file = step_dir / "episodes.jsonl"
        if not match or not episode_file.exists():
            continue

        step = int(match.group(1))
        epoch = int(match.group(2))
        step_records = []
        with episode_file.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("mode") != "val":
                    continue
                task = record.get("task") or {}
                task_id = task.get("query_id") or record.get("task_hash")
                if task_id is None:
                    continue
                metrics = record.get("metrics") or {}
                step_records.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "line_no": line_no,
                        "task_id": str(task_id),
                        "data_source": task.get("data_source"),
                        "acc": float(bool(record.get("is_correct"))),
                        "f1": float(metrics.get("f1_score") or 0.0),
                        "len_mean": total_response_tokens(record),
                    }
                )

        if not step_records:
            continue
        frame = pd.DataFrame(step_records)
        duplicate_count = int(frame.duplicated("task_id", keep=False).sum())
        frame = frame.drop_duplicates("task_id", keep="last")
        audit_rows.append(
            {
                "step": step,
                "epoch": epoch,
                "raw_rows": len(step_records),
                "unique_tasks": frame["task_id"].nunique(),
                "duplicate_rows": duplicate_count,
            }
        )
        records.extend(frame.to_dict("records"))

    if not records:
        raise RuntimeError(f"No validation episode records found under {run_dir}")

    per_query = pd.DataFrame(records).sort_values(["step", "task_id"])
    audit = pd.DataFrame(audit_rows).sort_values("step")
    expected_tasks = int(audit["unique_tasks"].max())
    complete_steps = audit.loc[audit["unique_tasks"].eq(expected_tasks), "step"]
    per_query = per_query[per_query["step"].isin(complete_steps)].copy()
    return per_query, audit


def build_lst(per_query: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tracking_rows = []
    selected_rows = []

    for base_step in BASE_STEPS:
        base = per_query[
            per_query["step"].eq(base_step)
            & per_query["acc"].ge(EASY_ACC_THRESHOLD)
        ].copy()
        task_ids = sorted(base["task_id"].unique())
        if not task_ids:
            continue

        for _, row in base.iterrows():
            selected_rows.append(
                {
                    "base_step": base_step,
                    "task_id": row["task_id"],
                    "data_source": row["data_source"],
                    "base_acc": row["acc"],
                    "base_f1": row["f1"],
                    "base_total_response_tokens": row["len_mean"],
                }
            )

        cohort = per_query[
            per_query["task_id"].isin(task_ids)
            & per_query["step"].ge(base_step)
        ]
        cohort = (
            cohort.groupby("step", as_index=False)
            .agg(
                n_queries=("task_id", "count"),
                A_E=("acc", "mean"),
                F1_E=("f1", "mean"),
                L_E=("len_mean", "mean"),
            )
            .sort_values("step")
            .reset_index(drop=True)
        )

        l_stars = []
        l_star_steps = []
        for _, row in cohort.iterrows():
            candidates = cohort[cohort["A_E"].ge(row["A_E"] - EPSILON_A)]
            best_index = candidates["L_E"].idxmin()
            l_stars.append(float(cohort.loc[best_index, "L_E"]))
            l_star_steps.append(int(cohort.loc[best_index, "step"]))

        cohort["base_step"] = base_step
        cohort["epsilon_A"] = EPSILON_A
        cohort["L_star"] = l_stars
        cohort["L_star_step"] = l_star_steps
        cohort["LST"] = 100.0 * (cohort["L_E"] / cohort["L_star"] - 1.0)
        cohort["extra_tokens_vs_L_star"] = cohort["L_E"] - cohort["L_star"]
        tracking_rows.extend(cohort.to_dict("records"))

    if not tracking_rows:
        raise RuntimeError("No easy cohorts were found")
    return (
        pd.DataFrame(tracking_rows).sort_values(["base_step", "step"]),
        pd.DataFrame(selected_rows).sort_values(["base_step", "task_id"]),
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
            "figure.dpi": 220,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(axis) -> None:
    axis.grid(True, color=GRID, linewidth=0.9)
    axis.set_axisbelow(True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#111111")
        spine.set_linewidth(1.15)
    axis.tick_params(width=1.1, length=5, colors="#111111")


def plot_lst(tracking: pd.DataFrame) -> Path:
    configure_style()
    fig, axes = plt.subplots(2, len(BASE_STEPS), figsize=(20.5, 8.4))
    max_step = int(tracking["step"].max())

    for column, base_step in enumerate(BASE_STEPS):
        panel = tracking[tracking["base_step"].eq(base_step)].sort_values("step")
        length_axis = axes[0, column]
        lst_axis = axes[1, column]
        if panel.empty:
            length_axis.axis("off")
            lst_axis.axis("off")
            continue

        for axis, metric in ((length_axis, "L_E"), (lst_axis, "LST")):
            axis.plot(
                panel["step"],
                panel[metric],
                color=COLOR,
                marker="o",
                linewidth=2.8,
                markersize=6.4,
                markeredgecolor="#4A4A4A",
                markeredgewidth=0.75,
            )
            axis.axvline(
                base_step,
                color="#64748B",
                linestyle="--",
                linewidth=1.1,
                alpha=0.65,
            )
            axis.set_xlim(base_step, max_step)
            axis.set_xlabel("Validation step")
            style_axis(axis)

        length_axis.set_title(
            rf"$\mathbf{{({chr(ord('a') + column)})}}$ "
            rf"$\mathcal{{E}}_{{{base_step}}}$ · n={int(panel['n_queries'].iloc[0])}",
            pad=9,
        )
        length_axis.set_ylim(bottom=0)
        lst_axis.set_ylim(bottom=0)
        lst_axis.yaxis.set_major_formatter(PercentFormatter(100.0, decimals=0))
        if column == 0:
            length_axis.set_ylabel("Easy mean completion tokens")
            lst_axis.set_ylabel("LST (%)")

    handle = Line2D(
        [0],
        [0],
        color=COLOR,
        marker="o",
        linewidth=2.8,
        markersize=6.4,
        markeredgecolor="#4A4A4A",
        markeredgewidth=0.75,
        label="LSD (direct FKL, EMA HL=2)",
    )
    fig.legend(
        handles=[handle],
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
    )
    fig.suptitle(
        "Within-run LST on BrowseComp-Plus easy cohorts",
        fontsize=19,
        y=0.995,
    )
    fig.text(
        0.5,
        0.958,
        "One validation rollout per query; easy means correct at the cohort base step",
        ha="center",
        fontsize=11,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=[0, 0.065, 1, 0.94], w_pad=1.25, h_pad=1.35)
    output_path = OUTPUT_DIR / "current_lsd_easy_mean_length_and_lst.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_query, audit = load_per_query(RUN_DIR)
    tracking, selected = build_lst(per_query)

    audit.to_csv(OUTPUT_DIR / "episode_step_audit.csv", index=False)
    per_query.to_csv(OUTPUT_DIR / "per_query_validation_metrics.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "easy_query_sets.csv", index=False)
    tracking.to_csv(OUTPUT_DIR / "current_lsd_lst_timeseries.csv", index=False)
    output_path = plot_lst(tracking)

    latest = (
        tracking.sort_values("step")
        .groupby("base_step", as_index=False)
        .tail(1)[["base_step", "step", "n_queries", "A_E", "L_E", "L_star", "LST"]]
    )
    print(latest.to_string(index=False))
    print(f"figure: {output_path}")


if __name__ == "__main__":
    main()
