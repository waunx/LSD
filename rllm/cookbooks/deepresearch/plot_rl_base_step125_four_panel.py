#!/usr/bin/env python3
"""Plot four validation diagnostics for the RL Base checkpoint through step 125."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator, PercentFormatter


RUN_NAME = "8b_stale05_rs_h20_remote_rag_4x4_seq_mean_24k_hdfs_val"
VALIDATION_DIR = (
    Path("/mnt/hdfs/__MERLIN_USER_DIR__/rllm/ckpts/deepresearch")
    / RUN_NAME
    / "validation"
)
OUTPUT_DIR = (
    Path("/opt/tiger/rllm/cookbooks/deepresearch/plots")
    / "rl_base_step125_four_panel"
)
FIRST_STEP = 25
LST_ACCURACY_EPSILON = 0.01
COLOR = "#F59F00"
GRID = "#CBD5E1"


def load_validation_rows() -> pd.DataFrame:
    records: list[dict] = []
    pattern = re.compile(r"val_version_(\d+)\.jsonl$")

    for path in sorted(VALIDATION_DIR.glob("val_version_*.jsonl")):
        match = pattern.match(path.name)
        if not match:
            continue
        step = int(match.group(1))
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                metrics = row.get("metrics") or {}
                records.append(
                    {
                        "step": step,
                        "query_id": int(row["idx"]),
                        "raw_reward": float(
                            metrics.get("raw_reward", row.get("reward", 0.0)) or 0.0
                        ),
                        "is_correct": float(metrics.get("is_correct", False) or 0.0),
                        "policy_tokens": float(
                            metrics.get("total_completion_tokens", 0.0) or 0.0
                        ),
                        "tool_calls": float(
                            metrics.get("total_tool_calls", 0.0) or 0.0
                        ),
                    }
                )

    if not records:
        raise RuntimeError(f"No validation records found under {VALIDATION_DIR}")

    data = pd.DataFrame(records).sort_values(["step", "query_id"])
    counts = data.groupby("step")["query_id"].nunique()
    expected = int(counts.max())
    if not counts.eq(expected).all():
        raise RuntimeError(f"Incomplete validation steps: {counts.to_dict()}")
    return data


def build_series(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall = (
        data.groupby("step", as_index=False)
        .agg(
            n_queries=("query_id", "nunique"),
            success_rate=("is_correct", "mean"),
            raw_reward=("raw_reward", "mean"),
            mean_policy_tokens=("policy_tokens", "mean"),
            mean_tool_calls=("tool_calls", "mean"),
        )
        .sort_values("step")
    )

    base = data[data["step"].eq(FIRST_STEP)]
    easy_ids = set(base.loc[base["is_correct"].ge(0.5), "query_id"])
    if not easy_ids:
        raise RuntimeError(f"No correct queries found at step {FIRST_STEP}")

    cohort = (
        data[data["query_id"].isin(easy_ids)]
        .groupby("step", as_index=False)
        .agg(
            n_queries=("query_id", "nunique"),
            easy_success_rate=("is_correct", "mean"),
            easy_mean_policy_tokens=("policy_tokens", "mean"),
        )
        .sort_values("step")
    )

    l_stars = []
    l_star_steps = []
    for row in cohort.itertuples(index=False):
        candidates = cohort[
            cohort["easy_success_rate"].ge(
                row.easy_success_rate - LST_ACCURACY_EPSILON
            )
        ]
        best_index = candidates["easy_mean_policy_tokens"].idxmin()
        l_stars.append(float(cohort.loc[best_index, "easy_mean_policy_tokens"]))
        l_star_steps.append(int(cohort.loc[best_index, "step"]))

    cohort["base_step"] = FIRST_STEP
    cohort["accuracy_epsilon"] = LST_ACCURACY_EPSILON
    cohort["l_star"] = l_stars
    cohort["l_star_step"] = l_star_steps
    cohort["lst_percent"] = 100.0 * (
        cohort["easy_mean_policy_tokens"] / cohort["l_star"] - 1.0
    )
    return overall, cohort


def style_axis(axis) -> None:
    axis.grid(True, color=GRID, linewidth=0.8, alpha=0.75)
    axis.set_axisbelow(True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.set_xlim(left=FIRST_STEP)
    axis.set_ylim(bottom=0)
    for spine in axis.spines.values():
        spine.set_color("#334155")
        spine.set_linewidth(1.0)
    axis.tick_params(width=1.0, length=4, colors="#334155")


def plot(overall: pd.DataFrame, cohort: pd.DataFrame) -> Path:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 220,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.86, 6.7))
    panels = [
        (axes[0, 0], overall, "success_rate", "(a) Task success", "Success rate"),
        (
            axes[0, 1],
            overall,
            "mean_policy_tokens",
            "(b) Policy-generated length",
            "Mean policy tokens",
        ),
        (
            axes[1, 0],
            cohort,
            "lst_percent",
            "(c) Fixed-easy-query LST",
            "LST (%)",
        ),
        (
            axes[1, 1],
            overall,
            "mean_tool_calls",
            "(d) Interaction cost",
            "Mean tool calls",
        ),
    ]

    for axis, frame, metric, title, ylabel in panels:
        axis.plot(
            frame["step"],
            frame[metric],
            color=COLOR,
            marker="o",
            linewidth=2.6,
            markersize=6.5,
            markeredgecolor="#7C2D12",
            markeredgewidth=0.7,
            label="RL Base",
        )
        axis.set_title(title, pad=8, fontweight="semibold")
        axis.set_xlabel("Training step")
        axis.set_ylabel(ylabel)
        style_axis(axis)

    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[1, 0].yaxis.set_major_formatter(PercentFormatter(100.0, decimals=0))
    axes[1, 0].text(
        0.02,
        0.96,
        f"step-{FIRST_STEP} easy cohort, n={int(cohort['n_queries'].iloc[0])}",
        transform=axes[1, 0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="#475569",
    )

    fig.suptitle(
        "DeepResearch RL Base validation through step 125",
        fontsize=18,
        fontweight="semibold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.955,
        "BrowseComp-Plus | 830 queries/checkpoint | fixed-easy LST uses an accuracy-matched length frontier",
        ha="center",
        fontsize=10,
        color="#475569",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=[0.02, 0.07, 0.99, 0.93], w_pad=2.0, h_pad=2.0)

    output = OUTPUT_DIR / "rl_base_step125_four_panel.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_validation_rows()
    overall, cohort = build_series(data)
    overall.to_csv(OUTPUT_DIR / "rl_base_validation_metrics.csv", index=False)
    cohort.to_csv(OUTPUT_DIR / "rl_base_fixed_easy_lst.csv", index=False)
    output = plot(overall, cohort)
    print(overall.to_string(index=False))
    print()
    print(cohort.to_string(index=False))
    print(f"\nfigure: {output}")


if __name__ == "__main__":
    main()
