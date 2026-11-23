#!/usr/bin/env python3
"""Plot aligned validation metrics for the historical RL Base and current LSD run."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, PercentFormatter


BASE_RUN = (
    "qwen3_8b_base_cutthebill_stale05_rs_h20_remote_rag_4x4_token_mean_refine_"
    "g340-cd51-4900-43f3-5d5c-d5f7-d3d1_20260823_225231"
)
LSD_RUN = (
    "qwen3_8b_base_cutthebill_lsd_online_ema_direct_fkl_coef1_thr1_loss_reweigh_"
    "ema_delay2_g340-cd51-4900-2d65-9d6b-b6e4-6020"
)
LSD_RUN_DIR = Path("/tmp/deepresearch_lsd_logs/deepresearch") / LSD_RUN
OUTPUT_DIR = (
    Path("/opt/tiger/rllm/cookbooks/deepresearch/plots")
    / "current_lsd_vs_historical_base"
)

# Immutable W&B validation history from the five resumed Base runs. Duplicate
# checkpoints are resolved in favor of the latest re-evaluation.
BASE_VALIDATION_ROWS = [
    (0, 0.0148542545, 0.0180722892, 0.0012048193, 666.9879518, 1.0144578313, 0.0216867470, "n5hog4b5", 0),
    (25, 0.0314941473, 0.0397590361, 0.0036540804, 352.4614458, 1.0240963855, 1.1265060241, "n5hog4b5", 0),
    (25, 0.0603724640, 0.0746987952, 0.0304136253, 478.7819277, 1.3108433735, 1.1710843373, "47ugbrdp", 1),
    (50, 0.0900667623, 0.1144578313, 0.0542168675, 293.6373494, 1.3024096386, 1.0012048193, "47ugbrdp", 1),
    (50, 0.1695991084, 0.1891566265, 0.1254523522, 431.1987952, 2.0024096386, 1.0060240964, "a0bw1tf6", 2),
    (75, 0.1690069869, 0.1963855422, 0.1180722892, 424.0746988, 1.9963855422, 0.9987951807, "a0bw1tf6", 2),
    (100, 0.1760335295, 0.2000000000, 0.1192771084, 700.9963855, 1.9987951807, 1.0036144578, "a0bw1tf6", 2),
    (125, 0.1752050337, 0.2012048193, 0.1180722892, 580.8168675, 2.0000000000, 1.0084337349, "a0bw1tf6", 2),
    (150, 0.1807931751, 0.2000000000, 0.1319612591, 658.5192771, 2.0060240964, 1.0891566265, "a0bw1tf6", 2),
    (175, 0.2046335750, 0.2120481928, 0.1491002571, 956.1698795, 2.6638554217, 1.9951807229, "a0bw1tf6", 2),
    (200, 0.2367357763, 0.2626506024, 0.1730531520, 1348.1867470, 2.8951807229, 2.9879518072, "a0bw1tf6", 2),
    (200, 0.2317406899, 0.2506024096, 0.1730531520, 1247.8445783, 2.9024096386, 3.0228915663, "oektf3xf", 3),
    (225, 0.2461211402, 0.2566265060, 0.1854219949, 1646.1277108, 3.2734939759, 4.4506024096, "oektf3xf", 3),
    (225, 0.2409542435, 0.2566265060, 0.1825095057, 1588.2457831, 3.2493975904, 4.2975903614, "sy9n34q5", 4),
    (250, 0.2595843945, 0.2746987952, 0.1979823455, 2123.2626506, 3.9313253012, 6.0807228916, "sy9n34q5", 4),
    (275, 0.3038808904, 0.3192771084, 0.2357320099, 3199.6867470, 5.4144578313, 8.6987951807, "sy9n34q5", 4),
]

METRICS = [
    ("f1_score", "F1 reward", True),
    ("is_correct", "Correct rate", True),
    ("exact_match", "Exact match", True),
    ("completion_tokens", "Completion tokens", False),
    ("num_turns", "Agent turns", False),
    ("tool_calls", "Search calls", False),
]

COLORS = {"RL Base": "#FFB000", "LSD (direct FKL, HL=2)": "#2563EB"}
MARKERS = {"RL Base": "D", "LSD (direct FKL, HL=2)": "o"}


def load_base_history() -> pd.DataFrame:
    columns = [
        "step",
        "f1_score",
        "is_correct",
        "exact_match",
        "completion_tokens",
        "num_turns",
        "tool_calls",
        "source_run_id",
        "source_order",
    ]
    data = pd.DataFrame(BASE_VALIDATION_ROWS, columns=columns)
    data = data.sort_values(["step", "source_order"]).drop_duplicates(
        subset=["step"], keep="last"
    )
    data["experiment"] = "RL Base"
    data["n_episodes"] = 830
    return data.reset_index(drop=True)


def _total_response_tokens(record: dict) -> float:
    return float(
        sum(
            float(step.get("response_length") or 0.0)
            for trajectory in record.get("trajectories") or []
            for step in trajectory.get("steps") or []
        )
    )


def load_lsd_history(run_dir: Path) -> pd.DataFrame:
    step_pattern = re.compile(r"val_step_(\d+)_epoch_(\d+)$")
    rows: list[dict] = []

    for step_dir in sorted(
        (run_dir / "episodes").glob("val_step_*_epoch_*"),
        key=lambda path: int(step_pattern.match(path.name).group(1))
        if step_pattern.match(path.name)
        else -1,
    ):
        match = step_pattern.match(step_dir.name)
        episode_file = step_dir / "episodes.jsonl"
        if not match or not episode_file.exists():
            continue

        episodes = []
        with episode_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("mode") != "val":
                    continue
                metrics = record.get("metrics") or {}
                episodes.append(
                    {
                        "f1_score": float(metrics.get("f1_score") or 0.0),
                        "is_correct": float(bool(record.get("is_correct"))),
                        "exact_match": float(bool(metrics.get("exact_match"))),
                        "completion_tokens": _total_response_tokens(record),
                        "num_turns": float(metrics.get("num_turns") or 0.0),
                        "tool_calls": float(metrics.get("total_tool_calls") or 0.0),
                        "duplicate_search": float(
                            metrics.get("duplicate_search_detected") or 0.0
                        ),
                        "tool_error": float(
                            metrics.get("total_tool_return_error") or 0.0
                        ),
                        "parse_error": float(
                            metrics.get("total_parse_tool_args_error") or 0.0
                        ),
                    }
                )

        if not episodes:
            continue
        frame = pd.DataFrame(episodes)
        row = {
            "step": int(match.group(1)),
            "epoch": int(match.group(2)),
            "experiment": "LSD (direct FKL, HL=2)",
            "source_run_id": LSD_RUN,
            "n_episodes": len(frame),
        }
        row.update(frame.mean(numeric_only=True).to_dict())
        rows.append(row)

    if not rows:
        raise RuntimeError(f"No validation episodes found under {run_dir / 'episodes'}")
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
            "figure.dpi": 220,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def style_axis(axis, percent: bool) -> None:
    axis.grid(True, color="#B2B2B2", linewidth=0.9)
    axis.set_axisbelow(True)
    axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    if percent:
        axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    for spine in axis.spines.values():
        spine.set_color("#111111")
        spine.set_linewidth(1.1)
    axis.tick_params(width=1.1, length=5, colors="#111111")


def plot_metrics(data: pd.DataFrame) -> Path:
    configure_style()
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2))

    for index, (metric, title, percent) in enumerate(METRICS):
        axis = axes.flat[index]
        for experiment in ("RL Base", "LSD (direct FKL, HL=2)"):
            group = data[data["experiment"].eq(experiment)].sort_values("step")
            axis.plot(
                group["step"],
                group[metric],
                color=COLORS[experiment],
                marker=MARKERS[experiment],
                linewidth=2.8,
                markersize=6.7,
                markeredgecolor="#4A4A4A",
                markeredgewidth=0.75,
                label=experiment,
            )
        letter = chr(ord("a") + index)
        axis.set_title(rf"$\mathbf{{({letter})}}$ {title}", pad=9)
        axis.set_xlabel("Training step")
        axis.set_xlim(left=0)
        if percent:
            axis.set_ylim(bottom=0)
        else:
            axis.set_ylim(bottom=0)
        style_axis(axis, percent)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[name],
            marker=MARKERS[name],
            linewidth=2.8,
            markersize=6.7,
            markeredgecolor="#4A4A4A",
            markeredgewidth=0.75,
            label=name,
        )
        for name in ("RL Base", "LSD (direct FKL, HL=2)")
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        handlelength=2.3,
        columnspacing=2.0,
    )
    fig.suptitle(
        "BrowseComp-Plus validation: historical RL Base vs current LSD",
        fontsize=19,
        y=0.995,
    )
    fig.text(
        0.5,
        0.958,
        "830 episodes/checkpoint; Base uses W&B aggregates, LSD is recomputed from episode JSONL",
        ha="center",
        fontsize=11,
        color="#4A4A4A",
    )
    fig.tight_layout(rect=[0, 0.065, 1, 0.945], w_pad=1.5, h_pad=1.6)
    output_path = OUTPUT_DIR / "deepresearch_base_vs_current_lsd_validation.png"
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = load_base_history()
    lsd = load_lsd_history(LSD_RUN_DIR)
    aligned = pd.concat([base, lsd], ignore_index=True, sort=False).sort_values(
        ["experiment", "step"]
    )

    base.to_csv(OUTPUT_DIR / "historical_base_validation.csv", index=False)
    lsd.to_csv(OUTPUT_DIR / "current_lsd_validation_from_episodes.csv", index=False)
    aligned.to_csv(OUTPUT_DIR / "aligned_validation_metrics.csv", index=False)
    output_path = plot_metrics(aligned)

    print(
        aligned.groupby("experiment", sort=False)
        .agg(
            step_min=("step", "min"),
            step_max=("step", "max"),
            n_steps=("step", "nunique"),
            episodes_per_step=("n_episodes", "min"),
        )
        .to_string()
    )
    print(f"figure: {output_path}")


if __name__ == "__main__":
    main()
