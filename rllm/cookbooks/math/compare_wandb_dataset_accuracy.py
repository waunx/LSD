#!/usr/bin/env python3
"""Compare W&B validation accuracy for AMC/AIME25/AIME26 between two runs.

Usage:
  1. Paste your key into WANDB_API_KEY below, or export WANDB_API_KEY.
  2. If auto entity resolution fails, set ENTITY below or pass --entity.
  3. Run:
       python3 compare_wandb_dataset_accuracy.py

The script writes:
  - wandb_dataset_accuracy_timeseries.csv
  - wandb_dataset_accuracy_summary.csv
  - wandb_dataset_accuracy_summary.md
  - wandb_dataset_accuracy_<metric>.png
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# Paste the key here if you do not want to use the WANDB_API_KEY env var.
WANDB_API_KEY = ""

# Leave empty to infer from the W&B API key. If that fails, pass --entity.
ENTITY = ""
PROJECT = "math"

BASE_RUN_NAMES = [
    "xuwan-zhejiang-university/math/paab29vk",
    "xuwan-zhejiang-university/math/9xmy646v",
]
LSD_RUN_NAMES = [
    "qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-lsd-online-topk32-fkl-thr0.875-coef0.1-ema_delay4",
]

DEFAULT_DATASETS = ["AMC", "aime25", "aime26"]
DEFAULT_METRICS = ["accuracy"]
DEFAULT_LAST_K = 5
DEFAULT_OUTPUT_DIR = Path(
    "/tmp/rllm_logs/math/"
    "qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-lsd-online-topk32-fkl-thr0.875-coef0.1-ema_delay4/"
    "analysis_query_buckets"
)

RUN_LABELS = {
    "base": "base_pass32",
    "lsd": "lsd_topk32",
}
RUN_COLORS = {
    "base_pass32": "#7c3aed",
    "lsd_topk32": "#0f9f8f",
}

# Prefer the real training/eval step if it is logged. Fall back to W&B _step.
STEP_KEYS = [
    "trainer/global_step",
    "global_step",
    "training_step",
    "_step",
]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def flatten_run_names(values: Iterable[str] | None, defaults: list[str]) -> list[str]:
    if not values:
        return defaults
    names: list[str] = []
    for value in values:
        names.extend(split_csv(value))
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None, help="W&B API key. Overrides WANDB_API_KEY in this file/env.")
    parser.add_argument("--entity", default=ENTITY, help="W&B entity/team/user. Empty means infer from API key.")
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument(
        "--base-run",
        nargs="*",
        default=None,
        help=(
            "Base run display name/id/path(s), comma or space separated. "
            "If multiple runs overlap on the same step, later runs override earlier runs."
        ),
    )
    parser.add_argument("--lsd-run", nargs="*", default=None, help="LSD run display name(s), comma or space separated.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS), help="Examples: accuracy or accuracy,pass@32")
    parser.add_argument("--last-k", type=int, default=DEFAULT_LAST_K)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def make_wandb_api(api_key: str, timeout: int):
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is not installed in this Python environment") from exc

    try:
        return wandb.Api(api_key=api_key or None, timeout=timeout)
    except TypeError:
        # Older wandb versions do not accept api_key in the constructor.
        return wandb.Api(timeout=timeout)


def resolve_entity(api, entity: str) -> str:
    if entity:
        return entity
    viewer = api.viewer
    inferred = getattr(viewer, "entity", None) or getattr(viewer, "username", None)
    if not inferred:
        raise RuntimeError("Could not infer W&B entity. Pass --entity or set ENTITY in the script.")
    return inferred


def hashable_run_key(run) -> tuple | str:
    """Return a stable hashable key for de-duplicating W&B Run objects."""
    path = getattr(run, "path", None)
    if isinstance(path, list):
        return tuple(path)
    if path:
        return str(path)
    return str(getattr(run, "id", None) or getattr(run, "name", None))


def run_names_for_match(run) -> set[str]:
    names = {
        name
        for name in (getattr(run, "name", None), getattr(run, "display_name", None), getattr(run, "id", None))
        if name
    }
    path = getattr(run, "path", None)
    if isinstance(path, list):
        names.add("/".join(path))
    elif path:
        names.add(str(path))
    return names


def find_run(api, entity: str, project: str, run_names: list[str], label: str):
    path = f"{entity}/{project}"
    candidates = []

    for run_name in run_names:
        if "/" in run_name:
            try:
                candidates.append(api.run(run_name))
                continue
            except Exception:
                pass
        for filters in ({"display_name": run_name}, {"name": run_name}):
            try:
                candidates.extend(list(api.runs(path, filters=filters)))
            except Exception:
                pass

    exact = []
    seen = set()
    for run in candidates:
        key = hashable_run_key(run)
        if key in seen:
            continue
        seen.add(key)
        names = run_names_for_match(run)
        if names.intersection(run_names):
            exact.append(run)

    if not exact:
        wanted = "\n  - ".join(run_names)
        raise RuntimeError(f"Could not find {label} run in {path}. Tried:\n  - {wanted}")

    exact.sort(
        key=lambda run: str(getattr(run, "updated_at", None) or getattr(run, "created_at", None) or ""),
        reverse=True,
    )
    if len(exact) > 1:
        print(f"[warn] found {len(exact)} candidate {label} runs; using newest: {exact[0].name}", file=sys.stderr)
    return exact[0]


def find_runs(api, entity: str, project: str, run_specs: list[str], label: str) -> list:
    """Resolve run specs in order. Later runs can override earlier runs when merged."""
    runs = []
    for index, run_spec in enumerate(run_specs, start=1):
        run = find_run(api, entity, project, [run_spec], f"{label}[{index}]")
        runs.append(run)
    return runs


def metric_key(dataset: str, metric: str) -> str:
    return f"val/{dataset}/{metric}"


def finite_float(value) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def pick_step(row: dict) -> tuple[float | None, str | None]:
    for key in STEP_KEYS:
        if key in row:
            step = finite_float(row.get(key))
            if step is not None:
                return step, key
    return None, None


def scan_metric_history(run, run_label: str, datasets: list[str], metrics: list[str]) -> pd.DataFrame:
    wanted_keys = {(dataset, metric): metric_key(dataset, metric) for dataset in datasets for metric in metrics}
    rows = []

    for row in run.scan_history(page_size=1000):
        step, step_key = pick_step(row)
        if step is None:
            continue
        for (dataset, metric), key in wanted_keys.items():
            value = finite_float(row.get(key))
            if value is None:
                continue
            rows.append(
                {
                    "run_label": run_label,
                    "run_name": run.name,
                    "run_id": run.id,
                    "step": step,
                    "step_key": step_key,
                    "dataset": dataset,
                    "metric": metric,
                    "metric_key": key,
                    "value": value,
                }
            )

    if not rows:
        keys = ", ".join(wanted_keys.values())
        raise RuntimeError(f"No requested validation metrics found in run {run.name}. Looked for: {keys}")

    df = pd.DataFrame(rows)
    return df.drop_duplicates(["run_label", "dataset", "metric", "step"], keep="last").sort_values(
        ["metric", "dataset", "run_label", "step"]
    )


def merge_metric_histories(frames: list[pd.DataFrame], run_label: str) -> pd.DataFrame:
    """Merge segmented runs; later frames win on overlapping dataset/metric/step."""
    if not frames:
        raise RuntimeError(f"No history frames to merge for {run_label}")
    annotated = []
    for priority, frame in enumerate(frames):
        cur = frame.copy()
        cur["merge_priority"] = priority
        cur["run_label"] = run_label
        annotated.append(cur)
    merged = pd.concat(annotated, ignore_index=True).sort_values(
        ["dataset", "metric", "step", "merge_priority"]
    )
    merged = merged.drop_duplicates(["run_label", "dataset", "metric", "step"], keep="last")
    return merged.drop(columns=["merge_priority"]).sort_values(["metric", "dataset", "run_label", "step"])


def std_or_zero(values: pd.Series) -> float:
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1))


def row_at_or_before(series: pd.DataFrame, step: float) -> pd.Series:
    eligible = series[series.step <= step].sort_values("step")
    if eligible.empty:
        raise RuntimeError("No row at or before comparison step")
    return eligible.iloc[-1]


def summarize(timeseries: pd.DataFrame, datasets: list[str], metrics: list[str], last_k: int) -> pd.DataFrame:
    rows = []
    base_label = RUN_LABELS["base"]
    lsd_label = RUN_LABELS["lsd"]

    for metric in metrics:
        for dataset in datasets:
            base = timeseries[
                (timeseries.run_label == base_label)
                & (timeseries.dataset == dataset)
                & (timeseries.metric == metric)
            ].sort_values("step")
            lsd = timeseries[
                (timeseries.run_label == lsd_label)
                & (timeseries.dataset == dataset)
                & (timeseries.metric == metric)
            ].sort_values("step")
            if base.empty or lsd.empty:
                continue

            common_max_step = min(float(base.step.max()), float(lsd.step.max()))
            base_latest = row_at_or_before(base, common_max_step)
            lsd_latest = row_at_or_before(lsd, common_max_step)

            common_steps = sorted(set(base.step.tolist()).intersection(set(lsd.step.tolist())))
            common_steps = [step for step in common_steps if step <= common_max_step]
            last_steps = common_steps[-last_k:]
            if last_steps:
                base_last = base[base.step.isin(last_steps)].sort_values("step")
                lsd_last = lsd[lsd.step.isin(last_steps)].sort_values("step")
                last_k_mode = "common_steps"
            else:
                base_last = base[base.step <= common_max_step].tail(last_k)
                lsd_last = lsd[lsd.step <= common_max_step].tail(last_k)
                last_steps = sorted(set(base_last.step.tolist()).union(set(lsd_last.step.tolist())))
                last_k_mode = "independent_tail"

            base_best = base[base.step <= common_max_step].loc[base[base.step <= common_max_step].value.idxmax()]
            lsd_best = lsd[lsd.step <= common_max_step].loc[lsd[lsd.step <= common_max_step].value.idxmax()]

            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "common_max_step": common_max_step,
                    "last_k_mode": last_k_mode,
                    "last_k_steps": ",".join(str(int(s)) if float(s).is_integer() else str(s) for s in last_steps),
                    "base_latest_step": float(base_latest.step),
                    "lsd_latest_step": float(lsd_latest.step),
                    "base_latest": float(base_latest.value),
                    "lsd_latest": float(lsd_latest.value),
                    "delta_latest_lsd_minus_base": float(lsd_latest.value - base_latest.value),
                    "base_last_k_mean": float(base_last.value.mean()),
                    "base_last_k_std": std_or_zero(base_last.value),
                    "lsd_last_k_mean": float(lsd_last.value.mean()),
                    "lsd_last_k_std": std_or_zero(lsd_last.value),
                    "delta_last_k_mean_lsd_minus_base": float(lsd_last.value.mean() - base_last.value.mean()),
                    "base_best": float(base_best.value),
                    "base_best_step": float(base_best.step),
                    "lsd_best": float(lsd_best.value),
                    "lsd_best_step": float(lsd_best.step),
                    "delta_best_lsd_minus_base": float(lsd_best.value - base_best.value),
                }
            )

    if not rows:
        raise RuntimeError("No overlapping base/LSD metric rows found")
    return pd.DataFrame(rows)


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def signed_pct(value: float) -> str:
    return f"{100.0 * value:+.2f}%"


def fmt_step(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def write_markdown(summary: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# W&B Dataset Accuracy Comparison",
        "",
        "| dataset | metric | step | base latest | lsd latest | delta | base last-k | lsd last-k | delta mean | base best | lsd best |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["metric", "dataset"]).itertuples(index=False):
        lines.append(
            "| "
            f"{row.dataset} | {row.metric} | {fmt_step(row.common_max_step)} | "
            f"{pct(row.base_latest)} | {pct(row.lsd_latest)} | {signed_pct(row.delta_latest_lsd_minus_base)} | "
            f"{pct(row.base_last_k_mean)}+-{pct(row.base_last_k_std)} | "
            f"{pct(row.lsd_last_k_mean)}+-{pct(row.lsd_last_k_std)} | "
            f"{signed_pct(row.delta_last_k_mean_lsd_minus_base)} | "
            f"{pct(row.base_best)}@{fmt_step(row.base_best_step)} | "
            f"{pct(row.lsd_best)}@{fmt_step(row.lsd_best_step)} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(summary: pd.DataFrame) -> None:
    print("\nLatest / last-k summary")
    for row in summary.sort_values(["metric", "dataset"]).itertuples(index=False):
        print(
            f"{row.dataset:6s} {row.metric:8s} step={fmt_step(row.common_max_step):>5s} "
            f"base={pct(row.base_latest):>7s} lsd={pct(row.lsd_latest):>7s} "
            f"delta={signed_pct(row.delta_latest_lsd_minus_base):>8s} | "
            f"last-k delta={signed_pct(row.delta_last_k_mean_lsd_minus_base):>8s} "
            f"steps=[{row.last_k_steps}]"
        )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def plot_timeseries(timeseries: pd.DataFrame, datasets: list[str], metrics: list[str], output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    paths: list[Path] = []
    base_label = RUN_LABELS["base"]
    lsd_label = RUN_LABELS["lsd"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 180,
        }
    )

    for metric in metrics:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.0), sharey=False)
        if len(datasets) == 1:
            axes = [axes]
        for ax, dataset in zip(axes, datasets):
            panel = timeseries[(timeseries.dataset == dataset) & (timeseries.metric == metric)].sort_values("step")
            for label, marker in [(base_label, "o"), (lsd_label, "s")]:
                g = panel[panel.run_label == label].sort_values("step")
                if g.empty:
                    continue
                ax.plot(
                    g.step,
                    g.value,
                    color=RUN_COLORS[label],
                    marker=marker,
                    linewidth=2.35,
                    markersize=3.4,
                    alpha=0.92,
                    label=label,
                )
            ax.set_title(f"{dataset} {metric}")
            ax.set_xlabel("eval step")
            ax.set_ylabel(metric)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.grid(True, axis="y", alpha=0.22, linewidth=0.8)
            ax.grid(True, axis="x", alpha=0.10, linewidth=0.7)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        out_path = output_dir / f"wandb_dataset_accuracy_{safe_name(metric)}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)
    return paths


def main() -> None:
    args = parse_args()
    api_key = args.api_key or WANDB_API_KEY or os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        raise RuntimeError("No W&B API key provided. Set WANDB_API_KEY in this script/env or pass --api-key.")

    datasets = split_csv(args.datasets)
    metrics = split_csv(args.metrics)
    base_names = flatten_run_names(args.base_run, BASE_RUN_NAMES)
    lsd_names = flatten_run_names(args.lsd_run, LSD_RUN_NAMES)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = make_wandb_api(api_key, timeout=args.timeout)
    entity = resolve_entity(api, args.entity)

    print(f"project={entity}/{args.project}")
    base_runs = find_runs(api, entity, args.project, base_names, "base")
    lsd_run = find_run(api, entity, args.project, lsd_names, "lsd")
    print("base_runs=" + ", ".join(f"{run.name} id={run.id}" for run in base_runs))
    if len(base_runs) > 1:
        print(f"base overlap policy: later runs override earlier runs ({base_runs[-1].id} has highest priority)")
    print(f"lsd_run={lsd_run.name} id={lsd_run.id}")

    base_frames = [
        scan_metric_history(run, RUN_LABELS["base"], datasets, metrics)
        for run in base_runs
    ]
    base_ts = merge_metric_histories(base_frames, RUN_LABELS["base"])
    lsd_ts = scan_metric_history(lsd_run, RUN_LABELS["lsd"], datasets, metrics)
    timeseries = pd.concat([base_ts, lsd_ts], ignore_index=True).sort_values(
        ["metric", "dataset", "run_label", "step"]
    )
    summary = summarize(timeseries, datasets, metrics, args.last_k)

    timeseries_path = args.output_dir / "wandb_dataset_accuracy_timeseries.csv"
    summary_path = args.output_dir / "wandb_dataset_accuracy_summary.csv"
    markdown_path = args.output_dir / "wandb_dataset_accuracy_summary.md"
    timeseries.to_csv(timeseries_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_markdown(summary, markdown_path)

    plot_paths: list[Path] = []
    if not args.no_plot:
        plot_paths = plot_timeseries(timeseries, datasets, metrics, args.output_dir)

    print_summary(summary)
    print("\nwrote:")
    print(f"  {timeseries_path}")
    print(f"  {summary_path}")
    print(f"  {markdown_path}")
    for path in plot_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
