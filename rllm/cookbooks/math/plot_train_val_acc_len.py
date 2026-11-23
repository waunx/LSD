#!/usr/bin/env python3
"""Plot train/validation accuracy and response length in a polished 4-panel figure.

Default panels:
- train: batch/accuracy + batch/mean_response_len
- AMC: val/AMC/accuracy + val/AMC/mean_response_len
- AIME25: val/aime25/accuracy + val/aime25/mean_response_len
- AIME26: val/aime26/accuracy + val/aime26/mean_response_len

Default W&B run composition is the 4k base run:
- xuwan-zhejiang-university/math/paab29vk
- xuwan-zhejiang-university/math/9xmy646v

If the same step appears in multiple run segments, later segments override earlier
segments. This means the default overlap region uses 9xmy646v.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# Prefer environment variable. You may paste a key here if needed.
WANDB_API_KEY = ""
ENTITY = ""
PROJECT = "math"

DEFAULT_RUN_SPECS = [
    "xuwan-zhejiang-university/math/paab29vk",
    "xuwan-zhejiang-university/math/9xmy646v",
]
DEFAULT_OUTPUT_DIR = Path(
    "/opt/tiger/rllm/cookbooks/math/logs/math/"
    "qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-1/analysis_query_buckets"
)

PANEL_SPECS = [
    {
        "panel": "train",
        "title": "Train",
        "acc_key": "batch/accuracy",
        "len_key": "batch/mean_response_len",
    },
    {
        "panel": "AMC",
        "title": "AMC",
        "acc_key": "val/AMC/accuracy",
        "len_key": "val/AMC/mean_response_len",
    },
    {
        "panel": "aime25",
        "title": "AIME25",
        "acc_key": "val/aime25/accuracy",
        "len_key": "val/aime25/mean_response_len",
    },
    {
        "panel": "aime26",
        "title": "AIME26",
        "acc_key": "val/aime26/accuracy",
        "len_key": "val/aime26/mean_response_len",
    },
]

STEP_KEYS = [
    "training/global_step",
    "trainer/global_step",
    "global_step",
    "_step",
]

BLUE = "#2563eb"
RED = "#dc2626"
GRAY = "#64748b"

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


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def flatten_values(values: Iterable[str] | None, defaults: list[str]) -> list[str]:
    if not values:
        return defaults
    out: list[str] = []
    for value in values:
        out.extend(split_csv(value))
    return out


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
            value = finite_float(row.get(key))
            if value is not None:
                return value, key
    return None, None


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
        return wandb.Api(timeout=timeout)


def resolve_entity(api, entity: str) -> str:
    if entity:
        return entity
    viewer = api.viewer
    inferred = getattr(viewer, "entity", None) or getattr(viewer, "username", None)
    if not inferred:
        raise RuntimeError("Could not infer W&B entity. Pass --entity or set ENTITY in the script.")
    return inferred


def find_run(api, entity: str, project: str, run_spec: str):
    if "/" in run_spec:
        return api.run(run_spec)
    path = f"{entity}/{project}"
    for filters in ({"display_name": run_spec}, {"name": run_spec}, {"id": run_spec}):
        matches = list(api.runs(path, filters=filters))
        if matches:
            return matches[0]
    raise RuntimeError(f"Could not find W&B run: {run_spec}")


def scan_wandb_run(run, *, source_priority: int) -> pd.DataFrame:
    wanted_keys = set()
    for spec in PANEL_SPECS:
        wanted_keys.add(spec["acc_key"])
        wanted_keys.add(spec["len_key"])

    rows = []
    for row in run.scan_history(page_size=1000):
        step, step_key = pick_step(row)
        if step is None:
            continue
        for spec in PANEL_SPECS:
            acc = finite_float(row.get(spec["acc_key"]))
            length = finite_float(row.get(spec["len_key"]))
            if acc is None and length is None:
                continue
            rows.append(
                {
                    "panel": spec["panel"],
                    "title": spec["title"],
                    "step": step,
                    "step_key": step_key,
                    "acc": acc,
                    "length": length,
                    "acc_key": spec["acc_key"],
                    "len_key": spec["len_key"],
                    "source": getattr(run, "id", None) or getattr(run, "name", None),
                    "source_priority": source_priority,
                }
            )
    return pd.DataFrame(rows)


def parse_step_line(line: str) -> dict | None:
    if "step:" not in line or " - " not in line:
        return None
    match = re.search(r"(?:^|\s)step:(-?\d+(?:\.\d+)?)\s+-\s+(.*)", line)
    if not match:
        return None

    row: dict[str, float] = {"_step": float(match.group(1))}
    payload = match.group(2)
    for item in payload.split(" - "):
        if ":" not in item:
            continue
        key, raw_value = item.split(":", 1)
        key = key.strip()
        value_parts = raw_value.strip().split()
        if not value_parts:
            continue
        value_text = value_parts[0]
        value = finite_float(value_text)
        if value is not None:
            row[key] = value
    return row


def scan_output_log(log_path: Path, *, source_priority: int) -> pd.DataFrame:
    rows = []
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parsed = parse_step_line(line)
            if not parsed:
                continue
            step, step_key = pick_step(parsed)
            if step is None:
                continue
            for spec in PANEL_SPECS:
                acc = finite_float(parsed.get(spec["acc_key"]))
                length = finite_float(parsed.get(spec["len_key"]))
                if acc is None and length is None:
                    continue
                rows.append(
                    {
                        "panel": spec["panel"],
                        "title": spec["title"],
                        "step": step,
                        "step_key": step_key,
                        "acc": acc,
                        "length": length,
                        "acc_key": spec["acc_key"],
                        "len_key": spec["len_key"],
                        "source": str(log_path),
                        "source_priority": source_priority,
                    }
                )
    return pd.DataFrame(rows)


def merge_sources(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise RuntimeError("No input frames were loaded")
    merged = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if merged.empty:
        raise RuntimeError("No train/val accuracy-length rows were found")
    merged = merged.sort_values(["panel", "step", "source_priority"])
    merged = merged.drop_duplicates(["panel", "step"], keep="last")
    return merged.sort_values(["panel", "step"]).reset_index(drop=True)


def apply_spines(ax, *, spine_width: float = 2.15) -> None:
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(spine_width)


def set_tick_style(
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


def plot_four_panel(data: pd.DataFrame, output_dir: Path, *, filename: str) -> Path:
    plt.rcParams.update(POLISHED_RC)
    fig, axes = plt.subplots(1, len(PANEL_SPECS), figsize=(19.2, 4.8), sharey=False)

    for ax, spec in zip(axes, PANEL_SPECS):
        panel = data[data.panel == spec["panel"]].sort_values("step")
        ax2 = ax.twinx()
        if panel.empty:
            ax.set_title(spec["title"], fontsize=18, pad=12)
            ax.axis("off")
            ax2.axis("off")
            continue

        acc = panel.dropna(subset=["acc"])
        length = panel.dropna(subset=["length"])
        if not acc.empty:
            ax.plot(acc.step, acc.acc, color=BLUE, marker="o", linewidth=2.8, markersize=4.6)
        if not length.empty:
            ax2.plot(length.step, length.length, color=RED, marker="s", linewidth=2.55, markersize=4.3)

        ax.grid(True, axis="y", alpha=0.22, linewidth=0.9)
        ax.grid(True, axis="x", alpha=0.10, linewidth=0.8)
        ax.set_xlabel("step", fontsize=17)
        ax.set_title(spec["title"], fontsize=18, pad=12)
        ax.set_ylim(max(0.0, panel.acc.min() - 0.04), min(1.0, panel.acc.max() + 0.04))

        if ax is axes[0]:
            ax.set_ylabel("accuracy", color=BLUE, fontsize=17)
            set_tick_style(ax, axis="y", color=BLUE)
        else:
            set_tick_style(ax, axis="y", color=BLUE)
            ax.tick_params(axis="y", labelleft=False)
        set_tick_style(ax, axis="x", color="black")

        if ax is axes[-1]:
            ax2.set_ylabel("mean length", color=RED, fontsize=17)
            set_tick_style(ax2, axis="y", color=RED)
        else:
            ax2.set_ylabel("")
            set_tick_style(ax2, axis="y", color=RED)
            ax2.tick_params(axis="y", labelright=False, right=False)

        if not length.empty:
            pad = max(50.0, (length.length.max() - length.length.min()) * 0.12)
            ax2.set_ylim(max(0.0, length.length.min() - pad), length.length.max() + pad)
        apply_spines(ax)
        apply_spines(ax2)

    fig.tight_layout(rect=[0, 0, 1, 0.98], w_pad=2.4)
    out_path = output_dir / filename
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--run", nargs="*", default=None, help="W&B run path/id/name. Multiple runs are merged in order.")
    parser.add_argument("--output-log", nargs="*", default=None, help="Local W&B output.log path(s). Multiple logs are merged in order.")
    parser.add_argument("--input-csv", type=Path, default=None, help="Plot from a previously written source CSV.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--filename", default="train_val_acc_len_4panel.png")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "train_val_acc_len_4panel_timeseries.csv"

    if args.input_csv:
        data = pd.read_csv(args.input_csv)
    else:
        frames: list[pd.DataFrame] = []
        run_specs = flatten_values(args.run, DEFAULT_RUN_SPECS)
        if run_specs:
            api_key = args.api_key or WANDB_API_KEY or os.environ.get("WANDB_API_KEY", "")
            if api_key:
                api = make_wandb_api(api_key, args.timeout)
                entity = resolve_entity(api, args.entity)
                for priority, run_spec in enumerate(run_specs):
                    run = find_run(api, entity, args.project, run_spec)
                    print(f"loaded W&B run {run.path}")
                    frames.append(scan_wandb_run(run, source_priority=priority))
            elif not args.output_log:
                raise RuntimeError("No W&B API key found. Set WANDB_API_KEY, pass --api-key, or use --output-log.")

        for priority, log_path in enumerate(flatten_values(args.output_log, []), start=len(frames)):
            path = Path(log_path)
            print(f"loaded output log {path}")
            frames.append(scan_output_log(path, source_priority=priority))

        data = merge_sources(frames)
        data.to_csv(csv_path, index=False)

    out_png = plot_four_panel(data, args.output_dir, filename=args.filename)
    print(f"wrote {out_png}")
    print(f"wrote {csv_path}")
    for spec in PANEL_SPECS:
        panel = data[data.panel == spec["panel"]]
        if panel.empty:
            print(f"{spec['panel']}: missing")
        else:
            print(f"{spec['panel']}: step {int(panel.step.min())}->{int(panel.step.max())}, rows={len(panel)}")


if __name__ == "__main__":
    main()
