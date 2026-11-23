#!/usr/bin/env python3
"""Plot easy-query tracking and LST figures from rLLM episode JSONL logs.

This script reproduces:
- analysis_query_buckets/37_selected_base_0_25_50_75_100_easy_tracking_mean_only_tail32.png
- analysis_query_buckets/39_lst_initial_easy_acc_ge_7_8_simple.png
- analysis_query_buckets/41_lst_base_0_25_50_75_100_easy_tracking.png

Important data rule:
For every eval step and task_id, keep only the last 32 rollouts in JSONL order.
This handles resumed runs where the same step was appended twice, yielding
64 rollouts per task.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter


DEFAULT_RUN_DIR = Path(
    "/opt/tiger/rllm/cookbooks/math/logs/math/"
    "qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-1"
)

TREND_37_CSV = "40_latest_selected_base_0_25_50_75_100_tail32_trend.csv"
LST_41_CSV = "41_lst_base_0_25_50_75_100_easy_tracking_timeseries.csv"
BASE_STEPS = [0, 25, 50, 75, 100]

POLISHED_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
    "figure.dpi": 220,
}


def _apply_polished_spines(ax, *, spine_width: float = 2.15) -> None:
    """Use thick black panel borders for paper-ready plots."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(spine_width)


def _set_tick_style(
    ax,
    *,
    axis: str = "both",
    color: str = "black",
    labelsize: int = 15,
    width: float = 1.75,
    length: float = 5.6,
) -> None:
    ax.tick_params(
        axis=axis,
        which="major",
        labelsize=labelsize,
        width=width,
        length=length,
        colors=color,
        labelcolor=color,
    )
    ax.tick_params(axis=axis, which="minor", width=1.25, length=3.4, colors=color)


def _apply_polished_rc() -> None:
    plt.rcParams.update(POLISHED_RC)


def load_tail32_episode_records(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    episode_dir = run_dir / "episodes"
    step_re = re.compile(r"val_step_(\d+)_epoch_(\d+)")
    records: list[pd.DataFrame] = []
    audit_rows: list[dict] = []

    step_dirs = sorted(
        episode_dir.glob("val_step_*_epoch_*"),
        key=lambda path: int(step_re.match(path.name).group(1)) if step_re.match(path.name) else -1,
    )
    for step_dir in step_dirs:
        match = step_re.match(step_dir.name)
        if not match:
            continue
        step = int(match.group(1))
        epoch = int(match.group(2))
        jsonl_path = step_dir / "episodes.jsonl"
        if not jsonl_path.exists():
            continue

        rows = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("mode") != "val":
                    continue
                task_id = obj.get("task_id") or obj.get("source_id")
                if task_id is None:
                    continue
                episode_id = obj.get("episode_id") or f"{task_id}:{obj.get('rollout_idx')}"
                rows.append(
                    {
                        "step": step,
                        "epoch": epoch,
                        "line_no": line_no,
                        "episode_id": str(episode_id),
                        "task_id": str(task_id),
                        "rollout_idx": str(obj.get("rollout_idx")),
                        "data_source": obj.get("data_source"),
                        "is_correct": bool(obj.get("is_correct")),
                        "response_length": float(obj.get("response_length") or 0.0),
                    }
                )

        if not rows:
            continue

        raw = pd.DataFrame(rows).sort_values(["task_id", "line_no"])
        raw_counts = raw.groupby("task_id").size()
        kept = raw.groupby("task_id", group_keys=False).tail(32).copy()
        kept_counts = kept.groupby("task_id").size()
        audit_rows.append(
            {
                "step": step,
                "epoch": epoch,
                "raw_rows": len(raw),
                "raw_min_rollouts_per_task": int(raw_counts.min()),
                "raw_max_rollouts_per_task": int(raw_counts.max()),
                "kept_rows": len(kept),
                "kept_min_rollouts_per_task": int(kept_counts.min()),
                "kept_max_rollouts_per_task": int(kept_counts.max()),
            }
        )
        records.append(kept)

    if not records:
        raise RuntimeError(f"No validation episode records found under {episode_dir}")

    audit = pd.DataFrame(audit_rows).sort_values("step")
    episodes = pd.concat(records, ignore_index=True)
    complete_steps = sorted(audit.loc[audit["kept_min_rollouts_per_task"].ge(32), "step"].tolist())
    episodes = episodes[episodes["step"].isin(complete_steps)].copy()
    return episodes, audit


def build_per_query_metrics(episodes: pd.DataFrame) -> pd.DataFrame:
    per_query = episodes.groupby(["step", "task_id"], as_index=False).agg(
        data_source=("data_source", "first"),
        n_rollouts=("episode_id", "count"),
        acc=("is_correct", "mean"),
        correct=("is_correct", "sum"),
        len_mean=("response_length", "mean"),
        len_median=("response_length", "median"),
        len_p10=("response_length", lambda series: float(np.percentile(series, 10))),
        len_p90=("response_length", lambda series: float(np.percentile(series, 90))),
    )
    per_query["correct"] = per_query["correct"].astype(int)
    return per_query


def plot_easy_tracking(per_query: pd.DataFrame, out_dir: Path) -> Path:
    base_steps = BASE_STEPS
    complete_steps = sorted(per_query["step"].unique())
    selected_rows = []
    trend_rows = []

    for base_step in base_steps:
        base_df = per_query[(per_query["step"] == base_step) & (per_query["acc"] >= 7 / 8)].copy()
        task_ids = sorted(base_df["task_id"].astype(str).unique())
        for _, row in base_df.sort_values(["data_source", "task_id"]).iterrows():
            selected_rows.append(
                {
                    "base_step": base_step,
                    "task_id": row.task_id,
                    "data_source": row.data_source,
                    "base_acc": row.acc,
                    "base_correct": row.correct,
                    "base_len_mean": row.len_mean,
                }
            )
        for step in complete_steps:
            if step < base_step:
                continue
            cur = per_query[
                (per_query["step"] == step) & (per_query["task_id"].astype(str).isin(task_ids))
            ]
            if cur.empty:
                continue
            trend_rows.append(
                {
                    "base_step": base_step,
                    "step": step,
                    "n_queries": len(task_ids),
                    "n_rollouts": int(cur.n_rollouts.sum()),
                    "acc_mean": float(cur.acc.mean()),
                    "len_mean": float(cur.len_mean.mean()),
                }
            )

    selected = pd.DataFrame(selected_rows)
    trend = pd.DataFrame(trend_rows).sort_values(["base_step", "step"])
    selected.to_csv(out_dir / "40_latest_selected_base_0_25_50_75_100_tail32_selected_queries.csv", index=False)
    trend.to_csv(out_dir / TREND_37_CSV, index=False)
    return plot_easy_tracking_from_trend(trend, out_dir)


def plot_easy_tracking_from_trend(trend: pd.DataFrame, out_dir: Path) -> Path:
    base_steps = BASE_STEPS
    _apply_polished_rc()
    blue = "#2563eb"
    red = "#dc2626"
    gray = "#64748b"

    fig, axes = plt.subplots(1, len(base_steps), figsize=(23, 4.6), sharey=False)
    for ax, base_step in zip(axes, base_steps):
        g = trend[trend.base_step == base_step].sort_values("step")
        ax2 = ax.twinx()
        if g.empty:
            ax.set_title(f"{base_step} step Easy Data Tracking")
            ax.axis("off")
            ax2.axis("off")
            continue

        ax.plot(g.step, g.acc_mean, color=blue, marker="o", linewidth=2.8, markersize=4.8)
        ax2.plot(g.step, g.len_mean, color=red, marker="s", linewidth=2.55, markersize=4.5)
        ax.axvline(base_step, color=gray, linestyle="--", linewidth=1.25, alpha=0.5)
        ax.set_ylim(0.8, 1.0)
        ax2.set_ylim(max(0, g.len_mean.min() - 55), g.len_mean.max() + 65)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.9)
        ax.grid(True, axis="x", alpha=0.10, linewidth=0.8)
        ax.set_xlabel("eval step", fontsize=17)
        if ax is axes[0]:
            ax.set_ylabel("mean acc", color=blue, fontsize=17)
            _set_tick_style(ax, axis="y", color=blue)
        else:
            _set_tick_style(ax, axis="y", color=blue)
            ax.tick_params(axis="y", labelleft=False)
        _set_tick_style(ax, axis="x", color="black")
        if ax is axes[-1]:
            ax2.set_ylabel("mean length", color=red, fontsize=17)
            _set_tick_style(ax2, axis="y", color=red)
        else:
            ax2.set_ylabel("")
            _set_tick_style(ax2, axis="y", color=red)
            ax2.tick_params(axis="y", labelright=False, right=False)
        ax.set_title(f"{base_step} step Easy Data Tracking", fontsize=18, pad=12)
        _apply_polished_spines(ax)
        _apply_polished_spines(ax2)
        n_handle = Line2D([], [], linestyle="none", label=f"n={int(g.n_queries.iloc[0])}")
        ax.legend(
            handles=[n_handle],
            loc="lower left",
            frameon=True,
            handlelength=0,
            handletextpad=0,
            borderpad=0.4,
            fontsize=14,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.98], w_pad=2.4)
    out_path = out_dir / "37_selected_base_0_25_50_75_100_easy_tracking_mean_only_tail32.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_lst(per_query: pd.DataFrame, out_dir: Path, epsilon_a: float = 0.01) -> Path:
    step0 = per_query[per_query.step == 0]
    easy_tasks = sorted(step0.loc[step0.acc >= 7 / 8, "task_id"].astype(str).unique())
    easy = per_query[per_query.task_id.astype(str).isin(easy_tasks)].copy()
    cohort = easy.groupby("step", as_index=False).agg(
        n_queries=("task_id", "count"),
        A_E=("acc", "mean"),
        L_E=("len_mean", "mean"),
    )
    cohort = cohort.sort_values("step").reset_index(drop=True)

    lstars = []
    lstar_steps = []
    for _, row in cohort.iterrows():
        candidates = cohort[cohort.A_E >= row.A_E - epsilon_a]
        idx = candidates.L_E.idxmin()
        lstars.append(float(cohort.loc[idx, "L_E"]))
        lstar_steps.append(int(cohort.loc[idx, "step"]))

    cohort["epsilon_A"] = epsilon_a
    cohort["L_star"] = lstars
    cohort["L_star_step"] = lstar_steps
    cohort["LST"] = 100 * (cohort.L_E / cohort.L_star - 1)
    cohort["extra_tokens_vs_L_star"] = cohort.L_E - cohort.L_star
    cohort.to_csv(out_dir / "40_latest_lst_initial_easy_acc_ge_7_8_timeseries.csv", index=False)
    step0[step0.task_id.astype(str).isin(easy_tasks)].to_csv(
        out_dir / "40_latest_lst_initial_easy_acc_ge_7_8_query_set.csv", index=False
    )

    red = "#dc2626"
    gray = "#64748b"
    max_row = cohort.loc[cohort.LST.idxmax()]
    last_row = cohort.iloc[-1]

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(cohort.step, cohort.LST, color=red, marker="o", linewidth=2.8, markersize=4.0)
    ax.fill_between(cohort.step, 0, cohort.LST, color="#fecaca", alpha=0.32)
    ax.axhline(0, color=gray, linestyle="--", linewidth=1.0)
    ax.grid(True, alpha=0.24)
    ax.set_xlabel("eval step")
    ax.set_ylabel("LST_E (%)")
    ax.set_title(f"LST over training on easy queries (step0 acc >= 7/8, n={len(easy_tasks)})")
    ax.text(
        0.02,
        0.92,
        f"latest step {int(last_row.step)}: {last_row.LST:.1f}%  |  "
        f"max: {max_row.LST:.1f}% at step {int(max_row.step)}",
        transform=ax.transAxes,
        fontsize=11,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cbd5e1", alpha=0.92),
    )
    fig.tight_layout()
    out_path = out_dir / "39_lst_initial_easy_acc_ge_7_8_simple.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _lst_cohort_timeseries(
    per_query: pd.DataFrame,
    task_ids: list[str],
    *,
    base_step: int,
    epsilon_a: float,
) -> pd.DataFrame:
    easy = per_query[
        (per_query.task_id.astype(str).isin(task_ids)) & (per_query.step >= base_step)
    ].copy()
    cohort = easy.groupby("step", as_index=False).agg(
        n_queries=("task_id", "count"),
        A_E=("acc", "mean"),
        L_E=("len_mean", "mean"),
    )
    cohort = cohort.sort_values("step").reset_index(drop=True)

    lstars = []
    lstar_steps = []
    for _, row in cohort.iterrows():
        candidates = cohort[cohort.A_E >= row.A_E - epsilon_a]
        idx = candidates.L_E.idxmin()
        lstars.append(float(cohort.loc[idx, "L_E"]))
        lstar_steps.append(int(cohort.loc[idx, "step"]))

    cohort["base_step"] = base_step
    cohort["epsilon_A"] = epsilon_a
    cohort["L_star"] = lstars
    cohort["L_star_step"] = lstar_steps
    cohort["LST"] = 100 * (cohort.L_E / cohort.L_star - 1)
    cohort["extra_tokens_vs_L_star"] = cohort.L_E - cohort.L_star
    return cohort


def plot_base_step_lst_tracking(
    per_query: pd.DataFrame,
    out_dir: Path,
    epsilon_a: float = 0.01,
) -> Path:
    """Plot LST for easy-query cohorts selected at base steps 0/25/50/75/100."""
    base_steps = BASE_STEPS
    all_rows = []
    selected_rows = []

    for base_step in base_steps:
        base_df = per_query[(per_query.step == base_step) & (per_query.acc >= 7 / 8)].copy()
        task_ids = sorted(base_df.task_id.astype(str).unique())
        for _, row in base_df.sort_values(["data_source", "task_id"]).iterrows():
            selected_rows.append(
                {
                    "base_step": base_step,
                    "task_id": row.task_id,
                    "data_source": row.data_source,
                    "base_acc": row.acc,
                    "base_correct": row.correct,
                    "base_len_mean": row.len_mean,
                }
            )
        if task_ids:
            all_rows.append(
                _lst_cohort_timeseries(
                    per_query,
                    task_ids,
                    base_step=base_step,
                    epsilon_a=epsilon_a,
                )
            )

    if not all_rows:
        raise RuntimeError("No base-step easy cohorts found for LST tracking")

    lst = pd.concat(all_rows, ignore_index=True).sort_values(["base_step", "step"])
    selected = pd.DataFrame(selected_rows).sort_values(["base_step", "data_source", "task_id"])
    lst.to_csv(out_dir / LST_41_CSV, index=False)
    selected.to_csv(out_dir / "41_lst_base_0_25_50_75_100_easy_tracking_query_sets.csv", index=False)
    return plot_base_step_lst_tracking_from_timeseries(lst, out_dir)


def plot_base_step_lst_tracking_from_timeseries(lst: pd.DataFrame, out_dir: Path) -> Path:
    base_steps = BASE_STEPS
    _apply_polished_rc()
    red = "#dc2626"
    gray = "#64748b"

    fig, axes = plt.subplots(1, len(base_steps), figsize=(23, 4.6), sharey=False)
    for ax, base_step in zip(axes, base_steps):
        g = lst[lst.base_step == base_step].sort_values("step")
        if g.empty:
            ax.set_title(f"{base_step} step Easy LST Tracking")
            ax.axis("off")
            continue

        ax.plot(g.step, g.LST, color=red, marker="o", linewidth=2.8, markersize=4.8)
        ax.axhline(0, color=gray, linestyle="--", linewidth=1.25, alpha=0.65)
        ax.axvline(base_step, color=gray, linestyle="--", linewidth=1.25, alpha=0.45)
        y_min = min(0.0, float(g.LST.min()))
        y_max = max(0.0, float(g.LST.max()))
        y_pad = max(2.0, (y_max - y_min) * 0.12)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.9)
        ax.grid(True, axis="x", alpha=0.10, linewidth=0.8)
        ax.set_xlabel("eval step", fontsize=17)
        ax.set_ylabel("LST_E (%)", color=red, fontsize=17)
        _set_tick_style(ax, axis="y", color=red)
        _set_tick_style(ax, axis="x", color="black")
        ax.set_title(f"{base_step} step Easy LST Tracking", fontsize=18, pad=12)
        _apply_polished_spines(ax)

        n_handle = Line2D([], [], linestyle="none", label=f"n={int(g.n_queries.iloc[0])}")
        ax.legend(
            handles=[n_handle],
            loc="lower left",
            frameon=True,
            handlelength=0,
            handletextpad=0,
            borderpad=0.4,
            fontsize=14,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.98], w_pad=2.4)
    out_path = out_dir / "41_lst_base_0_25_50_75_100_easy_tracking.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _label_from_run_dir(run_dir: Path) -> str:
    name = run_dir.name
    if "lsd-online-topk32" in name:
        return "lsd_topk32"
    if name.endswith("pass32-1"):
        return "base_pass32"
    return name[:32]


def ensure_analysis_tables(run_dir: Path, epsilon_a: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return 37/41 source CSVs, generating them from episodes if needed."""
    out_dir = run_dir / "analysis_query_buckets"
    trend_path = out_dir / TREND_37_CSV
    lst_path = out_dir / LST_41_CSV
    if trend_path.exists() and lst_path.exists():
        return pd.read_csv(trend_path), pd.read_csv(lst_path)

    if not (run_dir / "episodes").exists():
        raise RuntimeError(
            f"Missing {trend_path.name}/{lst_path.name} and no episodes directory under {run_dir}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    episodes, audit = load_tail32_episode_records(run_dir)
    per_query = build_per_query_metrics(episodes)
    audit.to_csv(out_dir / "40_latest_tail32_rollout_audit_by_step.csv", index=False)
    per_query.to_csv(out_dir / "40_latest_tail32_per_query_step_metrics.csv", index=False)
    plot_easy_tracking(per_query, out_dir)
    plot_lst(per_query, out_dir, epsilon_a=epsilon_a)
    plot_base_step_lst_tracking(per_query, out_dir, epsilon_a=epsilon_a)
    return pd.read_csv(trend_path), pd.read_csv(lst_path)


def _apply_common_step_cap(
    tables: list[pd.DataFrame],
    max_step: int | None,
) -> tuple[list[pd.DataFrame], int]:
    if max_step is None:
        max_step = min(int(table["step"].max()) for table in tables if not table.empty)
    return [table[table["step"] <= max_step].copy() for table in tables], int(max_step)


def _comparison_palette(primary_label: str, compare_label: str) -> dict[str, str]:
    """Run-level colors used by comparison plots."""
    return {
        primary_label: "#0f9f8f",
        compare_label: "#7c3aed",
    }


def plot_easy_tracking_comparison(
    primary_trend: pd.DataFrame,
    compare_trend: pd.DataFrame,
    *,
    primary_label: str,
    compare_label: str,
    out_dir: Path,
    max_step: int | None = None,
) -> Path:
    """Overlay two 37-style easy tracking plots on the same axes."""
    capped_tables, _ = _apply_common_step_cap(
        [primary_trend, compare_trend],
        max_step,
    )
    primary_trend, compare_trend = capped_tables
    combined = pd.concat(
        [
            primary_trend.assign(run_label=primary_label),
            compare_trend.assign(run_label=compare_label),
        ],
        ignore_index=True,
    ).sort_values(["base_step", "run_label", "step"])
    combined.to_csv(out_dir / "37_compare_easy_tracking_tail32_trend.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 180,
        }
    )
    gray = "#64748b"
    colors = _comparison_palette(primary_label, compare_label)
    styles = {
        primary_label: {"acc_marker": "o", "len_marker": "s", "alpha": 0.95},
        compare_label: {"acc_marker": "^", "len_marker": "D", "alpha": 0.9},
    }

    fig, axes = plt.subplots(1, len(BASE_STEPS), figsize=(22, 3.95), sharey=False)
    for ax, base_step in zip(axes, BASE_STEPS):
        ax2 = ax.twinx()
        panel = combined[combined.base_step == base_step]
        if panel.empty:
            ax.set_title(f"{base_step} step Easy Data Tracking")
            ax.axis("off")
            ax2.axis("off")
            continue

        n_lines = []
        for label in [primary_label, compare_label]:
            g = panel[panel.run_label == label].sort_values("step")
            if g.empty:
                continue
            style = styles[label]
            color = colors[label]
            ax.plot(
                g.step,
                g.acc_mean,
                color=color,
                marker=style["acc_marker"],
                linestyle="-",
                linewidth=2.5,
                markersize=3.8,
                alpha=style["alpha"],
            )
            ax2.plot(
                g.step,
                g.len_mean,
                color=color,
                marker=style["len_marker"],
                linestyle="--",
                linewidth=2.15,
                markersize=3.5,
                alpha=style["alpha"] * 0.78,
            )
            n_lines.append(f"{label}: n={int(g.n_queries.iloc[0])}")

        ax.axvline(base_step, color=gray, linestyle="--", linewidth=1.0, alpha=0.5)
        ax.set_ylim(0.8, 1.0)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax2.set_ylim(max(0, panel.len_mean.min() - 55), panel.len_mean.max() + 65)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.10, linewidth=0.7)
        ax.set_xlabel("eval step")
        if ax is axes[0]:
            ax.set_ylabel("mean acc")
            ax.tick_params(axis="y")
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax2.set_ylabel("mean length")
        ax2.tick_params(axis="y")
        ax.set_title(f"{base_step} step Easy Data Tracking")
        ax.text(
            0.03,
            0.05,
            "\n".join(n_lines),
            transform=ax.transAxes,
            fontsize=8,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.9),
        )

    legend_handles = [
        Line2D([], [], color=colors[primary_label], marker="o", linestyle="-", linewidth=2.5, label=f"{primary_label} acc"),
        Line2D([], [], color=colors[primary_label], marker="s", linestyle="--", linewidth=2.15, alpha=0.74, label=f"{primary_label} length"),
        Line2D([], [], color=colors[compare_label], marker="^", linestyle="-", linewidth=2.5, label=f"{compare_label} acc"),
        Line2D([], [], color=colors[compare_label], marker="D", linestyle="--", linewidth=2.15, alpha=0.70, label=f"{compare_label} length"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=2.1)
    out_path = out_dir / "37_compare_easy_tracking_mean_only_tail32.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_lst_tracking_comparison(
    primary_lst: pd.DataFrame,
    compare_lst: pd.DataFrame,
    *,
    primary_label: str,
    compare_label: str,
    out_dir: Path,
    max_step: int | None = None,
) -> Path:
    """Overlay two 41-style base-step LST tracking plots on the same axes."""
    capped_tables, _ = _apply_common_step_cap(
        [primary_lst, compare_lst],
        max_step,
    )
    primary_lst, compare_lst = capped_tables
    combined = pd.concat(
        [
            primary_lst.assign(run_label=primary_label),
            compare_lst.assign(run_label=compare_label),
        ],
        ignore_index=True,
    ).sort_values(["base_step", "run_label", "step"])
    combined.to_csv(out_dir / "41_compare_lst_easy_tracking_timeseries.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 180,
        }
    )
    gray = "#64748b"
    colors = _comparison_palette(primary_label, compare_label)
    styles = {
        primary_label: {"marker": "o", "alpha": 0.95},
        compare_label: {"marker": "^", "alpha": 0.9},
    }

    fig, axes = plt.subplots(1, len(BASE_STEPS), figsize=(22, 3.95), sharey=False)
    for ax, base_step in zip(axes, BASE_STEPS):
        panel = combined[combined.base_step == base_step]
        if panel.empty:
            ax.set_title(f"{base_step} step Easy LST Tracking")
            ax.axis("off")
            continue

        n_lines = []
        for label in [primary_label, compare_label]:
            g = panel[panel.run_label == label].sort_values("step")
            if g.empty:
                continue
            style = styles[label]
            color = colors[label]
            ax.plot(
                g.step,
                g.LST,
                color=color,
                marker=style["marker"],
                linestyle="-",
                linewidth=2.35,
                markersize=4.0,
                alpha=style["alpha"],
            )
            n_lines.append(f"{label}: n={int(g.n_queries.iloc[0])}")

        ax.axhline(0, color=gray, linestyle="--", linewidth=1.0, alpha=0.65)
        ax.axvline(base_step, color=gray, linestyle="--", linewidth=1.0, alpha=0.45)
        y_min = min(0.0, float(panel.LST.min()))
        y_max = max(0.0, float(panel.LST.max()))
        y_pad = max(2.0, (y_max - y_min) * 0.12)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
        ax.grid(True, axis="x", alpha=0.10, linewidth=0.7)
        ax.set_xlabel("eval step")
        ax.set_ylabel("LST_E (%)")
        ax.tick_params(axis="y")
        ax.set_title(f"{base_step} step Easy LST Tracking")
        ax.text(
            0.03,
            0.05,
            "\n".join(n_lines),
            transform=ax.transAxes,
            fontsize=8,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cbd5e1", alpha=0.9),
        )

    legend_handles = [
        Line2D([], [], color=colors[primary_label], marker="o", linestyle="-", linewidth=2.35, label=primary_label),
        Line2D([], [], color=colors[compare_label], marker="^", linestyle="-", linewidth=2.35, label=compare_label),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=2.1)
    out_path = out_dir / "41_compare_lst_easy_tracking.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epsilon-a", type=float, default=0.01)
    parser.add_argument("--compare-run-dir", type=Path, default=None)
    parser.add_argument("--run-label", type=str, default=None)
    parser.add_argument("--compare-label", type=str, default=None)
    parser.add_argument(
        "--comparison-max-step",
        type=int,
        default=None,
        help="Maximum eval step for comparison plots. Defaults to the shared latest step.",
    )
    args = parser.parse_args()

    out_dir = args.run_dir / "analysis_query_buckets"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_run_dir is None:
        episodes, audit = load_tail32_episode_records(args.run_dir)
        per_query = build_per_query_metrics(episodes)
        audit.to_csv(out_dir / "40_latest_tail32_rollout_audit_by_step.csv", index=False)
        per_query.to_csv(out_dir / "40_latest_tail32_per_query_step_metrics.csv", index=False)

        out37 = plot_easy_tracking(per_query, out_dir)
        out39 = plot_lst(per_query, out_dir, epsilon_a=args.epsilon_a)
        out41 = plot_base_step_lst_tracking(per_query, out_dir, epsilon_a=args.epsilon_a)

        latest_step = int(per_query.step.max())
        print(f"latest_step={latest_step}")
        print(f"wrote {out37}")
        print(f"wrote {out39}")
        print(f"wrote {out41}")
        print(f"wrote {out_dir / '40_latest_tail32_rollout_audit_by_step.csv'}")
        return

    primary_label = args.run_label or _label_from_run_dir(args.run_dir)
    compare_label = args.compare_label or _label_from_run_dir(args.compare_run_dir)
    primary_trend, primary_lst = ensure_analysis_tables(args.run_dir, args.epsilon_a)
    compare_trend, compare_lst = ensure_analysis_tables(args.compare_run_dir, args.epsilon_a)

    comparison_max_step = args.comparison_max_step
    if comparison_max_step is None:
        comparison_max_step = min(
            int(primary_trend.step.max()),
            int(primary_lst.step.max()),
            int(compare_trend.step.max()),
            int(compare_lst.step.max()),
        )

    out37 = plot_easy_tracking_comparison(
        primary_trend,
        compare_trend,
        primary_label=primary_label,
        compare_label=compare_label,
        out_dir=out_dir,
        max_step=comparison_max_step,
    )
    out41 = plot_lst_tracking_comparison(
        primary_lst,
        compare_lst,
        primary_label=primary_label,
        compare_label=compare_label,
        out_dir=out_dir,
        max_step=comparison_max_step,
    )

    print(f"comparison_max_step={comparison_max_step}")
    print(f"wrote {out37}")
    print(f"wrote {out41}")
    print(f"wrote {out_dir / '37_compare_easy_tracking_tail32_trend.csv'}")
    print(f"wrote {out_dir / '41_compare_lst_easy_tracking_timeseries.csv'}")


if __name__ == "__main__":
    main()
