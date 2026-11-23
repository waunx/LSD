#!/usr/bin/env python3
"""Compare test pass metrics and easy-query length/LST between two runs.

This script intentionally avoids `wandb.Api` because in this environment the
W&B SDK service can hang while validating the API key. It talks to W&B GraphQL
directly with urllib.

Default comparison:
- base: xuwan-zhejiang-university/math/9xmy646v
- lsd/current: local output.log from q8nbfssy + vth0crds

Pass --api-key, export WANDB_API_KEY, or fill WANDB_API_KEY below before running.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd


# Paste your key here if you do not want to use the environment variable.
WANDB_API_KEY = ""

DEFAULT_BASE_RUN = "xuwan-zhejiang-university/math/9xmy646v"
DEFAULT_LSD_LOGS = [
    "/tmp/rllm_wandb/wandb/run-20260722_175420-q8nbfssy/files/output.log",
    "/tmp/rllm_wandb/wandb/run-20260724_103147-vth0crds/files/output.log",
]

DEFAULT_BASE_ANALYSIS_DIR = Path(
    "/opt/tiger/rllm/cookbooks/math/logs/math/"
    "qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-1/analysis_query_buckets"
)
DEFAULT_LSD_ANALYSIS_DIR = Path(
    "/tmp/rllm_logs/math/"
    "qwen3-4b-base-dapo17k-aime25-26-amc23-pass32-lsd-online-topk32-fkl-thr0.875-coef0.1-ema_delay4/"
    "analysis_query_buckets"
)
DEFAULT_OUTPUT_DIR = DEFAULT_LSD_ANALYSIS_DIR

TREND_37_CSV = "40_latest_selected_base_0_25_50_75_100_tail32_trend.csv"
LST_41_CSV = "41_lst_base_0_25_50_75_100_easy_tracking_timeseries.csv"

DATASETS = [
    ("AMC23", "AMC"),
    ("AIME25", "aime25"),
    ("AIME26", "aime26"),
]
WAND_BASE = "https://api.wandb.ai/graphql"


def split_values(values: Iterable[str] | None, defaults: list[str]) -> list[str]:
    if not values:
        return defaults
    out: list[str] = []
    for value in values:
        out.extend([item.strip() for item in value.split(",") if item.strip()])
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


def parse_run_path(path: str) -> tuple[str, str, str]:
    parts = path.strip("/").split("/")
    if len(parts) != 3:
        raise ValueError(f"Expected W&B run path entity/project/run_id, got: {path}")
    return parts[0], parts[1], parts[2]


def graphql_request(api_key: str, query: str, variables: dict, timeout: int) -> dict:
    token = base64.b64encode((f"api:{api_key.strip()}").encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        WAND_BASE,
        data=json.dumps({"query": query, "variables": variables}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Basic {token}",
            "User-Agent": "rllm-math-metrics-compare/1.0",
        },
    )
    retries = 2
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"W&B HTTP {exc.code}: {body[:1000]}") from exc
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            raise RuntimeError(f"W&B request failed after {retries + 1} attempts: {exc.reason}") from exc
    if payload.get("errors"):
        raise RuntimeError(f"W&B GraphQL errors: {payload['errors']}")
    return payload["data"]


def fetch_sampled_history(
    *,
    api_key: str,
    run_path: str,
    keys: list[str],
    samples: int,
    timeout: int,
) -> pd.DataFrame:
    entity, project, run_id = parse_run_path(run_path)
    spec = {"keys": ["_step", "training/global_step"] + keys, "samples": samples}
    query = """
    query RunSampledHistory($project: String!, $entity: String!, $name: String!, $specs: [JSONString!]!) {
      project(name: $project, entityName: $entity) {
        run(name: $name) {
          id
          name
          displayName
          sampledHistory(specs: $specs)
        }
      }
    }
    """
    data = graphql_request(
        api_key,
        query,
        {
            "entity": entity,
            "project": project,
            "name": run_id,
            "specs": [json.dumps(spec)],
        },
        timeout,
    )
    run = data["project"]["run"]
    if run is None:
        raise RuntimeError(f"W&B run not found: {run_path}")
    rows = run["sampledHistory"][0]
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No sampled history returned for {run_path}")
    df["run_path"] = run_path
    return df


def parse_step_line(line: str) -> tuple[int, dict[str, float]] | None:
    if "step:" not in line or " - " not in line:
        return None
    match = re.search(r"(?:^|\s)step:(\d+)\s+-\s+(.*)", line)
    if not match:
        return None
    step = int(match.group(1))
    values: dict[str, float] = {}
    for item in match.group(2).split(" - "):
        if ":" not in item:
            continue
        key, raw = item.split(":", 1)
        parts = raw.strip().split()
        if not parts:
            continue
        value = finite_float(parts[0])
        if value is not None:
            values[key.strip()] = value
    return step, values


def parse_local_output_logs(paths: list[str]) -> pd.DataFrame:
    rows = []
    for priority, path_text in enumerate(paths):
        path = Path(path_text)
        if not path.exists():
            print(f"[warn] missing local output log: {path}", file=sys.stderr)
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parsed = parse_step_line(line)
                if not parsed:
                    continue
                step, values = parsed
                row = {"_step": step, "source": str(path), "priority": priority}
                for _, ds in DATASETS:
                    for metric in ["pass@1", "pass@32", "mean_response_len"]:
                        key = f"val/{ds}/{metric}"
                        if key in values:
                            row[key] = values[key]
                rows.append(row)
    if not rows:
        raise RuntimeError("No validation metric rows found in local output logs")
    df = pd.DataFrame(rows).sort_values(["_step", "priority"])
    return df.drop_duplicates(["_step"], keep="last").drop(columns=["priority"])


def summarize_test_metrics(df: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    step_col = "training/global_step" if "training/global_step" in df.columns else "_step"
    for display, ds in DATASETS:
        for metric_name, key in [
            ("pass@1", f"val/{ds}/pass@1"),
            ("pass@32", f"val/{ds}/pass@32"),
            ("avg_len", f"val/{ds}/mean_response_len"),
        ]:
            if key not in df.columns:
                continue
            series = df[[step_col, key]].dropna().copy()
            if series.empty:
                continue
            max_idx = series[key].idxmax() if metric_name != "avg_len" else series[step_col].idxmax()
            latest = series.loc[series[step_col].idxmax()]
            best = series.loc[max_idx]
            rows.append(
                {
                    "run": label,
                    "dataset": display,
                    "metric": metric_name,
                    "latest_step": int(latest[step_col]),
                    "latest": float(latest[key]),
                    "best_step": int(best[step_col]),
                    "best_or_latest": float(best[key]),
                    "n_points": int(len(series)),
                }
            )
    return pd.DataFrame(rows)


def compare_test_metrics(base: pd.DataFrame, lsd: pd.DataFrame) -> pd.DataFrame:
    merged = base.merge(lsd, on=["dataset", "metric"], suffixes=("_base", "_lsd"))
    merged["delta_latest_lsd_minus_base"] = merged["latest_lsd"] - merged["latest_base"]
    merged["delta_best_lsd_minus_base"] = merged["best_or_latest_lsd"] - merged["best_or_latest_base"]
    cols = [
        "dataset",
        "metric",
        "latest_step_base",
        "latest_base",
        "latest_step_lsd",
        "latest_lsd",
        "delta_latest_lsd_minus_base",
        "best_step_base",
        "best_or_latest_base",
        "best_step_lsd",
        "best_or_latest_lsd",
        "delta_best_lsd_minus_base",
    ]
    return merged[cols].sort_values(["dataset", "metric"])


def load_easy_tables(base_dir: Path, lsd_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_trend = pd.read_csv(base_dir / TREND_37_CSV)
    lsd_trend = pd.read_csv(lsd_dir / TREND_37_CSV)
    base_lst = pd.read_csv(base_dir / LST_41_CSV)
    lsd_lst = pd.read_csv(lsd_dir / LST_41_CSV)

    rows = []
    for base_step in sorted(set(base_trend.base_step).intersection(set(lsd_trend.base_step))):
        b = base_trend[base_trend.base_step == base_step]
        l = lsd_trend[lsd_trend.base_step == base_step]
        common_step = int(min(b.step.max(), l.step.max()))
        b_latest = b[b.step <= common_step].sort_values("step").iloc[-1]
        l_latest = l[l.step <= common_step].sort_values("step").iloc[-1]
        rows.append(
            {
                "base_step": int(base_step),
                "common_step": common_step,
                "base_easy_len": float(b_latest.len_mean),
                "lsd_easy_len": float(l_latest.len_mean),
                "delta_easy_len_lsd_minus_base": float(l_latest.len_mean - b_latest.len_mean),
                "base_easy_acc": float(b_latest.acc_mean),
                "lsd_easy_acc": float(l_latest.acc_mean),
                "delta_easy_acc_lsd_minus_base": float(l_latest.acc_mean - b_latest.acc_mean),
            }
        )
    easy_len = pd.DataFrame(rows)

    rows = []
    for base_step in sorted(set(base_lst.base_step).intersection(set(lsd_lst.base_step))):
        b = base_lst[base_lst.base_step == base_step]
        l = lsd_lst[lsd_lst.base_step == base_step]
        common_step = int(min(b.step.max(), l.step.max()))
        b_latest = b[b.step <= common_step].sort_values("step").iloc[-1]
        l_latest = l[l.step <= common_step].sort_values("step").iloc[-1]
        rows.append(
            {
                "base_step": int(base_step),
                "common_step": common_step,
                "base_LST": float(b_latest.LST),
                "lsd_LST": float(l_latest.LST),
                "delta_LST_lsd_minus_base": float(l_latest.LST - b_latest.LST),
                "base_L_E": float(b_latest.L_E),
                "lsd_L_E": float(l_latest.L_E),
                "base_A_E": float(b_latest.A_E),
                "lsd_A_E": float(l_latest.A_E),
            }
        )
    easy_lst = pd.DataFrame(rows)
    return easy_len, easy_lst


def fmt(value: float, metric: str) -> str:
    if metric in {"pass@1", "pass@32"}:
        return f"{value:.4f}"
    if metric == "avg_len":
        return f"{value:.1f}"
    return f"{value:.3f}"


def print_test_table(comp: pd.DataFrame) -> None:
    print("\nTest metrics comparison")
    print("| dataset | metric | base latest | lsd latest | delta | base best/latest | lsd best/latest | delta best |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in comp.itertuples(index=False):
        print(
            f"| {row.dataset} | {row.metric} | "
            f"{fmt(row.latest_base, row.metric)}@{row.latest_step_base} | "
            f"{fmt(row.latest_lsd, row.metric)}@{row.latest_step_lsd} | "
            f"{fmt(row.delta_latest_lsd_minus_base, row.metric)} | "
            f"{fmt(row.best_or_latest_base, row.metric)}@{row.best_step_base} | "
            f"{fmt(row.best_or_latest_lsd, row.metric)}@{row.best_step_lsd} | "
            f"{fmt(row.delta_best_lsd_minus_base, row.metric)} |"
        )


def print_easy_tables(easy_len: pd.DataFrame, easy_lst: pd.DataFrame) -> None:
    print("\nEasy Len comparison")
    print("| base_step | common_step | base len | lsd len | delta len | base acc | lsd acc | delta acc |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in easy_len.itertuples(index=False):
        print(
            f"| {row.base_step} | {row.common_step} | {row.base_easy_len:.1f} | "
            f"{row.lsd_easy_len:.1f} | {row.delta_easy_len_lsd_minus_base:+.1f} | "
            f"{row.base_easy_acc:.4f} | {row.lsd_easy_acc:.4f} | {row.delta_easy_acc_lsd_minus_base:+.4f} |"
        )

    print("\nLST comparison")
    print("| base_step | common_step | base LST | lsd LST | delta LST | base L_E | lsd L_E | base A_E | lsd A_E |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in easy_lst.itertuples(index=False):
        print(
            f"| {row.base_step} | {row.common_step} | {row.base_LST:.2f} | "
            f"{row.lsd_LST:.2f} | {row.delta_LST_lsd_minus_base:+.2f} | "
            f"{row.base_L_E:.1f} | {row.lsd_L_E:.1f} | {row.base_A_E:.4f} | {row.lsd_A_E:.4f} |"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", default=None, help="W&B API key. Overrides script/env key.")
    parser.add_argument("--base-run", default=DEFAULT_BASE_RUN)
    parser.add_argument("--lsd-log", nargs="*", default=None, help="Local LSD output.log paths.")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--base-analysis-dir", type=Path, default=DEFAULT_BASE_ANALYSIS_DIR)
    parser.add_argument("--lsd-analysis-dir", type=Path, default=DEFAULT_LSD_ANALYSIS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = (args.api_key or os.environ.get("WANDB_API_KEY") or WANDB_API_KEY).strip()
    if not api_key:
        raise RuntimeError("Set WANDB_API_KEY in this script/env or pass --api-key.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    keys = []
    for _, ds in DATASETS:
        keys.extend(
            [
                f"val/{ds}/pass@1",
                f"val/{ds}/pass@32",
                f"val/{ds}/mean_response_len",
            ]
        )

    print(f"Fetching base W&B history: {args.base_run}")
    base_history = fetch_sampled_history(
        api_key=api_key,
        run_path=args.base_run,
        keys=keys,
        samples=args.samples,
        timeout=args.timeout,
    )
    lsd_logs = split_values(args.lsd_log, DEFAULT_LSD_LOGS)
    print("Parsing LSD local logs:")
    for path in lsd_logs:
        print(f"  {path}")
    lsd_history = parse_local_output_logs(lsd_logs)

    base_summary = summarize_test_metrics(base_history, "base")
    lsd_summary = summarize_test_metrics(lsd_history, "lsd")
    test_comp = compare_test_metrics(base_summary, lsd_summary)

    easy_len, easy_lst = load_easy_tables(args.base_analysis_dir, args.lsd_analysis_dir)

    base_history.to_csv(args.output_dir / "compare_9xmy_base_wandb_history_sampled.csv", index=False)
    lsd_history.to_csv(args.output_dir / "compare_lsd_local_val_history.csv", index=False)
    test_comp.to_csv(args.output_dir / "compare_9xmy_vs_lsd_test_metrics.csv", index=False)
    easy_len.to_csv(args.output_dir / "compare_9xmy_vs_lsd_easy_len.csv", index=False)
    easy_lst.to_csv(args.output_dir / "compare_9xmy_vs_lsd_lst.csv", index=False)

    print_test_table(test_comp)
    print_easy_tables(easy_len, easy_lst)
    print("\nwrote:")
    for name in [
        "compare_9xmy_base_wandb_history_sampled.csv",
        "compare_lsd_local_val_history.csv",
        "compare_9xmy_vs_lsd_test_metrics.csv",
        "compare_9xmy_vs_lsd_easy_len.csv",
        "compare_9xmy_vs_lsd_lst.csv",
    ]:
        print(f"  {args.output_dir / name}")


if __name__ == "__main__":
    main()
