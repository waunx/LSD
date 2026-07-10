#!/usr/bin/env python3
"""Visualize eval dynamics from verl validation jsonl dumps.

The script is intentionally self-contained so it can be reused across runs:

    /data/conda/envs/verl_lsd/bin/python scripts/visualize_eval_training_dynamics.py \
        --validation-dir outputs/0609-qwen3_4b_base-math-4k-3/validation_data

It prefers validation_data/length_scaling_eval/*.jsonl because those files keep
all rollouts for a prompt together and include precomputed length/pattern stats.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_PATTERNS = {
    "reflection": [
        r"\bwait\b",
        r"\bactually\b",
        r"\brecheck\b",
        r"\breconsider\b",
        r"\bmistake\b",
        r"\berror\b",
        r"\bhowever\b",
        r"\balternatively\b",
        r"\bon the other hand\b",
        r"\blet'?s check\b",
        r"\blet'?s verify\b",
        r"\bverify\b",
        r"\bcheck\b",
    ],
    "self_talk": [
        r"\blet'?s\b",
        r"\bwe need\b",
        r"\bwe can\b",
        r"\bI need\b",
        r"\bI will\b",
    ],
    "conclusion": [
        r"\btherefore\b",
        r"\bthus\b",
        r"\bso\b",
        r"\bhence\b",
        r"\bfinal answer\b",
        r"\\boxed",
    ],
    "uncertainty": [
        r"\bmaybe\b",
        r"\bperhaps\b",
        r"\bseems\b",
        r"\blikely\b",
        r"\bassume\b",
        r"\bsuppose\b",
    ],
    "code_trace": [
        r"```",
        r"\bpython\b",
        r"\bprint\b",
        r"\boutput\b",
        r"\bimplement\b",
    ],
}


TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|\\boxed|\\frac|\\sqrt|[=+\-*/^<>]+")


def numeric_step(path: Path) -> int | None:
    try:
        return int(path.stem)
    except ValueError:
        return None


def iter_step_files(input_dir: Path) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in input_dir.glob("*.jsonl"):
        step = numeric_step(path)
        if step is not None:
            files.append((step, path))
    return sorted(files)


def choose_input_dir(validation_dir: Path, prefer_length_scaling: bool) -> Path:
    grouped_dir = validation_dir / "length_scaling_eval"
    if prefer_length_scaling and grouped_dir.exists() and iter_step_files(grouped_dir):
        return grouped_dir
    return validation_dir


def tokenize(text: str) -> list[str]:
    return [tok.lower() for tok in TOKEN_RE.findall(text)]


def ngrams(tokens: list[str], n: int) -> Iterable[tuple[str, ...]]:
    if len(tokens) < n:
        return ()
    return zip(*(tokens[i:] for i in range(n)))


def safe_quantile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    return float(np.quantile(np.asarray(values, dtype=float), q))


def extract_outputs(record: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(record.get("outputs"), list):
        return [x for x in record["outputs"] if isinstance(x, dict)]
    return [record]


def sample_accept(index: int, max_items: int | None, stride: int) -> bool:
    if max_items is None:
        return True
    return index % stride == 0


def analyze_step(
    step: int,
    path: Path,
    compiled_patterns: dict[str, list[re.Pattern[str]]],
    ngram_ns: tuple[int, ...],
    max_text_responses: int | None,
) -> tuple[dict[str, Any], dict[str, Counter], dict[str, dict[str, float]]]:
    lengths: list[float] = []
    scores: list[float] = []
    accs: list[float] = []
    rep20: list[float] = []
    max_rep20: list[float] = []
    pass_at_8: list[float] = []
    avg_reward_at_8: list[float] = []
    line_pattern_counts: Counter[str] = Counter()
    pattern_counts: Counter[str] = Counter()
    word_count_for_patterns = 0
    output_count = 0
    sampled_output_count = 0
    ngram_counters = {n: Counter() for n in ngram_ns}
    total_ngrams = {n: 0 for n in ngram_ns}

    # We sample text-heavy metrics deterministically to keep this fast on full
    # 44-step x 12k-response evals. Length/acc stats still use every output.
    total_lines = sum(1 for _ in path.open("r", encoding="utf-8"))
    estimated_outputs = total_lines * 8
    if max_text_responses is None or estimated_outputs <= max_text_responses:
        stride = 1
    else:
        stride = max(1, math.ceil(estimated_outputs / max_text_responses))

    flat_index = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if "pass@8" in record:
                pass_at_8.append(float(record["pass@8"]))
            if "avg_reward@8" in record:
                avg_reward_at_8.append(float(record["avg_reward@8"]))
            if isinstance(record.get("pattern_counts_sum@8"), dict):
                line_pattern_counts.update(record["pattern_counts_sum@8"])

            for output in extract_outputs(record):
                output_count += 1
                length = output.get("response_length", output.get("length"))
                if isinstance(length, (int, float)):
                    lengths.append(float(length))
                score = output.get("score", output.get("reward"))
                if isinstance(score, (int, float)):
                    scores.append(float(score))
                acc = output.get("acc")
                if isinstance(acc, bool):
                    accs.append(float(acc))
                elif isinstance(acc, (int, float)):
                    accs.append(float(acc))
                if isinstance(output.get("repetition_ratio_20gram"), (int, float)):
                    rep20.append(float(output["repetition_ratio_20gram"]))
                if isinstance(output.get("most_repeated_20gram"), (int, float)):
                    max_rep20.append(float(output["most_repeated_20gram"]))

                if not sample_accept(flat_index, max_text_responses, stride):
                    flat_index += 1
                    continue
                flat_index += 1
                text = str(output.get("output", ""))
                if not text:
                    continue
                sampled_output_count += 1
                for name, patterns in compiled_patterns.items():
                    count = sum(len(p.findall(text)) for p in patterns)
                    pattern_counts[name] += count
                tokens = tokenize(text)
                word_count_for_patterns += len(tokens)
                for n in ngram_ns:
                    grams = list(ngrams(tokens, n))
                    total_ngrams[n] += len(grams)
                    ngram_counters[n].update(grams)

    row: dict[str, Any] = {
        "step": step,
        "file": str(path),
        "num_outputs": output_count,
        "num_text_outputs_sampled": sampled_output_count,
        "mean_length": float(np.mean(lengths)) if lengths else math.nan,
        "median_length": float(np.median(lengths)) if lengths else math.nan,
        "p10_length": safe_quantile(lengths, 0.10),
        "p90_length": safe_quantile(lengths, 0.90),
        "p95_length": safe_quantile(lengths, 0.95),
        "max_length": max(lengths) if lengths else math.nan,
        "mean_score": float(np.mean(scores)) if scores else math.nan,
        "mean_acc": float(np.mean(accs)) if accs else math.nan,
        "mean_pass_at_8": float(np.mean(pass_at_8)) if pass_at_8 else math.nan,
        "mean_reward_at_8": (
            float(np.mean(avg_reward_at_8)) if avg_reward_at_8 else math.nan
        ),
        "mean_repetition_ratio_20gram": (
            float(np.mean(rep20)) if rep20 else math.nan
        ),
        "mean_most_repeated_20gram": (
            float(np.mean(max_rep20)) if max_rep20 else math.nan
        ),
        "sample_token_count": word_count_for_patterns,
    }

    denom = max(1, word_count_for_patterns)
    for name, count in pattern_counts.items():
        row[f"{name}_per_1k_tokens"] = 1000.0 * count / denom
        row[f"{name}_count_sampled"] = count

    for name, count in line_pattern_counts.items():
        row[f"precomputed_{name}_per_output"] = count / max(1, output_count)

    ngram_stats: dict[str, dict[str, float]] = {}
    for n, counter in ngram_counters.items():
        total = total_ngrams[n]
        unique = len(counter)
        repeated = sum(c - 1 for c in counter.values() if c > 1)
        row[f"distinct_{n}gram"] = unique / total if total else math.nan
        row[f"repeat_{n}gram_ratio"] = repeated / total if total else math.nan
        row[f"unique_{n}grams"] = unique
        row[f"total_{n}grams_sampled"] = total
        ngram_stats[f"{n}gram"] = {
            "distinct": row[f"distinct_{n}gram"],
            "repeat_ratio": row[f"repeat_{n}gram_ratio"],
        }

    top_ngrams = {
        f"{n}gram": Counter({" ".join(k): v for k, v in counter.most_common(40)})
        for n, counter in ngram_counters.items()
    }
    return row, top_ngrams, ngram_stats


def build_length_histograms(input_dir: Path, bins: np.ndarray) -> pd.DataFrame:
    rows = []
    for step, path in iter_step_files(input_dir):
        lengths: list[float] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                for output in extract_outputs(record):
                    length = output.get("response_length", output.get("length"))
                    if isinstance(length, (int, float)):
                        lengths.append(float(length))
        hist, _ = np.histogram(lengths, bins=bins)
        total = max(1, int(hist.sum()))
        for idx, count in enumerate(hist):
            rows.append(
                {
                    "step": step,
                    "length_bin": f"{int(bins[idx])}-{int(bins[idx + 1])}",
                    "density": count / total,
                }
            )
    return pd.DataFrame(rows)


def save_overview(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(summary["step"], summary["mean_acc"], label="mean acc", lw=2.2)
    if summary["mean_pass_at_8"].notna().any():
        ax.plot(summary["step"], summary["mean_pass_at_8"], label="pass@8", lw=2.2)
    ax.set_title("Accuracy over training")
    ax.set_xlabel("step")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(summary["step"], summary["mean_length"], label="mean", lw=2.2)
    ax.plot(summary["step"], summary["median_length"], label="median", lw=2.0)
    ax.fill_between(
        summary["step"],
        summary["p10_length"],
        summary["p90_length"],
        alpha=0.2,
        label="p10-p90",
    )
    ax.set_title("Response length")
    ax.set_xlabel("step")
    ax.set_ylabel("tokens")
    ax.legend()

    ax = axes[1, 0]
    cols = [
        "reflection_per_1k_tokens",
        "self_talk_per_1k_tokens",
        "conclusion_per_1k_tokens",
        "uncertainty_per_1k_tokens",
        "code_trace_per_1k_tokens",
    ]
    for col in cols:
        if col in summary:
            ax.plot(summary["step"], summary[col], label=col.replace("_per_1k_tokens", ""))
    ax.set_title("Special word families")
    ax.set_xlabel("step")
    ax.set_ylabel("matches / 1k sampled tokens")
    ax.legend(ncol=2, fontsize=9)

    ax = axes[1, 1]
    for col in ["distinct_1gram", "distinct_2gram", "distinct_3gram"]:
        if col in summary:
            ax.plot(summary["step"], summary[col], label=col, lw=2.0)
    if "mean_repetition_ratio_20gram" in summary:
        ax.plot(
            summary["step"],
            summary["mean_repetition_ratio_20gram"],
            label="precomputed repetition_ratio_20gram",
            lw=2.0,
            ls="--",
        )
    ax.set_title("N-gram diversity / repetition")
    ax.set_xlabel("step")
    ax.set_ylabel("ratio")
    ax.legend(fontsize=9)

    fig.suptitle("Eval dynamics overview", fontsize=18)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_length_heatmap(hist_df: pd.DataFrame, out: Path) -> None:
    pivot = hist_df.pivot(index="length_bin", columns="step", values="density").fillna(0)
    # Keep bins in numeric order after pivot.
    pivot = pivot.reindex(
        sorted(pivot.index, key=lambda x: int(str(x).split("-")[0]))
    )
    fig, ax = plt.subplots(figsize=(18, 10), constrained_layout=True)
    sns.heatmap(pivot, ax=ax, cmap="mako", cbar_kws={"label": "density"})
    ax.set_title("Response length distribution over training")
    ax.set_xlabel("step")
    ax.set_ylabel("response length bin")
    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_pattern_plot(summary: pd.DataFrame, out: Path) -> None:
    pattern_cols = [c for c in summary.columns if c.startswith("precomputed_")]
    sampled_cols = [c for c in summary.columns if c.endswith("_per_1k_tokens")]
    fig, axes = plt.subplots(2, 1, figsize=(16, 11), constrained_layout=True)

    ax = axes[0]
    for col in sampled_cols:
        ax.plot(summary["step"], summary[col], lw=2, label=col.replace("_per_1k_tokens", ""))
    ax.set_title("Regex special/reflection word families")
    ax.set_xlabel("step")
    ax.set_ylabel("matches / 1k sampled tokens")
    ax.legend(ncol=3, fontsize=9)

    ax = axes[1]
    for col in pattern_cols:
        label = col.replace("precomputed_", "").replace("_per_output", "")
        ax.plot(summary["step"], summary[col], lw=2, label=label)
    ax.set_title("Precomputed pattern_counts from eval files")
    ax.set_xlabel("step")
    ax.set_ylabel("matches / output")
    ax.legend(ncol=4, fontsize=9)

    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_ngram_plot(summary: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)

    ax = axes[0]
    for col in ["distinct_1gram", "distinct_2gram", "distinct_3gram"]:
        if col in summary:
            ax.plot(summary["step"], summary[col], marker="o", ms=3, lw=2, label=col)
    ax.set_title("Distinct n-gram ratio")
    ax.set_xlabel("step")
    ax.set_ylabel("unique ngrams / total ngrams")
    ax.legend()

    ax = axes[1]
    for col in ["repeat_1gram_ratio", "repeat_2gram_ratio", "repeat_3gram_ratio"]:
        if col in summary:
            ax.plot(summary["step"], summary[col], marker="o", ms=3, lw=2, label=col)
    if "mean_repetition_ratio_20gram" in summary:
        ax.plot(
            summary["step"],
            summary["mean_repetition_ratio_20gram"],
            lw=2,
            ls="--",
            label="precomputed repetition_ratio_20gram",
        )
    ax.set_title("N-gram repetition")
    ax.set_xlabel("step")
    ax.set_ylabel("ratio")
    ax.legend()

    fig.savefig(out, dpi=180)
    plt.close(fig)


def save_top_ngram_shift(
    all_top_ngrams: dict[int, dict[str, Counter]],
    out_dir: Path,
    selected_steps: list[int],
    ngram_kind: str,
    top_k: int,
) -> None:
    vocab = Counter()
    for step in selected_steps:
        vocab.update(all_top_ngrams.get(step, {}).get(ngram_kind, Counter()))
    top_terms = [term for term, _ in vocab.most_common(top_k)]
    if not top_terms:
        return
    matrix = []
    for term in top_terms:
        row = []
        for step in selected_steps:
            counter = all_top_ngrams.get(step, {}).get(ngram_kind, Counter())
            total = sum(counter.values()) or 1
            row.append(1000.0 * counter.get(term, 0) / total)
        matrix.append(row)
    df = pd.DataFrame(matrix, index=top_terms, columns=selected_steps)
    fig, ax = plt.subplots(figsize=(14, max(8, top_k * 0.35)), constrained_layout=True)
    sns.heatmap(df, ax=ax, cmap="rocket_r", cbar_kws={"label": "freq / 1k top-ngram counts"})
    ax.set_title(f"Top {ngram_kind} shift across selected steps")
    ax.set_xlabel("step")
    ax.set_ylabel(ngram_kind)
    fig.savefig(out_dir / f"top_{ngram_kind}_shift.png", dpi=180)
    plt.close(fig)


def save_markdown_report(out_dir: Path, input_dir: Path, summary: pd.DataFrame) -> None:
    first = summary.iloc[0]
    last = summary.iloc[-1]
    lines = [
        "# Eval Training Dynamics",
        "",
        f"Input dir: `{input_dir}`",
        f"Steps: {int(first['step'])} to {int(last['step'])} ({len(summary)} eval points)",
        "",
        "## Quick Delta",
        "",
        f"- mean length: {first['mean_length']:.1f} -> {last['mean_length']:.1f}",
        f"- median length: {first['median_length']:.1f} -> {last['median_length']:.1f}",
        f"- mean acc: {first['mean_acc']:.4f} -> {last['mean_acc']:.4f}",
        f"- pass@8: {first['mean_pass_at_8']:.4f} -> {last['mean_pass_at_8']:.4f}",
        f"- reflection / 1k sampled tokens: "
        f"{first.get('reflection_per_1k_tokens', math.nan):.3f} -> "
        f"{last.get('reflection_per_1k_tokens', math.nan):.3f}",
        "",
        "## Figures",
        "",
        "- `eval_training_dynamics_overview.png`",
        "- `response_length_distribution_heatmap.png`",
        "- `special_words_over_training.png`",
        "- `ngram_diversity_over_training.png`",
        "- `top_2gram_shift.png`",
        "- `top_3gram_shift.png`",
        "",
        "## Tables",
        "",
        "- `eval_training_dynamics_summary.csv`",
        "- `response_length_distribution.csv`",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path(
            "/data/wx_data/verl-lsd/outputs/0609-qwen3_4b_base-math-4k-3/validation_data"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-length-scaling", action="store_true")
    parser.add_argument(
        "--max-text-responses-per-step",
        type=int,
        default=2000,
        help="Cap text-heavy regex/ngram metrics per step; length/acc use all outputs.",
    )
    parser.add_argument("--top-k-ngrams", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_dir = args.validation_dir.resolve()
    input_dir = choose_input_dir(validation_dir, not args.no_length_scaling)
    output_dir = args.output_dir or validation_dir / "analysis_training_dynamics"
    output_dir.mkdir(parents=True, exist_ok=True)

    step_files = iter_step_files(input_dir)
    if not step_files:
        raise SystemExit(f"No numeric *.jsonl eval files found in {input_dir}")

    sns.set_theme(style="whitegrid", context="talk")
    compiled_patterns = {
        name: [re.compile(p, re.IGNORECASE) for p in patterns]
        for name, patterns in DEFAULT_PATTERNS.items()
    }

    rows = []
    all_top_ngrams: dict[int, dict[str, Counter]] = {}
    for step, path in step_files:
        row, top_ngrams, _ = analyze_step(
            step=step,
            path=path,
            compiled_patterns=compiled_patterns,
            ngram_ns=(1, 2, 3),
            max_text_responses=args.max_text_responses_per_step,
        )
        rows.append(row)
        all_top_ngrams[step] = top_ngrams
        print(f"analyzed step {step}: {path.name}", flush=True)

    summary = pd.DataFrame(rows).sort_values("step")
    summary.to_csv(output_dir / "eval_training_dynamics_summary.csv", index=False)

    max_len = summary["p95_length"].max()
    if not np.isfinite(max_len):
        max_len = summary["max_length"].max()
    upper = int(max(512, math.ceil(float(max_len) / 256) * 256))
    bins = np.arange(0, upper + 257, 256)
    hist_df = build_length_histograms(input_dir, bins)
    hist_df.to_csv(output_dir / "response_length_distribution.csv", index=False)

    save_overview(summary, output_dir / "eval_training_dynamics_overview.png")
    save_length_heatmap(hist_df, output_dir / "response_length_distribution_heatmap.png")
    save_pattern_plot(summary, output_dir / "special_words_over_training.png")
    save_ngram_plot(summary, output_dir / "ngram_diversity_over_training.png")

    steps = list(summary["step"].astype(int))
    selected_steps = sorted(set([steps[0], steps[len(steps) // 4], steps[len(steps) // 2], steps[-1]]))
    save_top_ngram_shift(
        all_top_ngrams,
        output_dir,
        selected_steps,
        "2gram",
        args.top_k_ngrams,
    )
    save_top_ngram_shift(
        all_top_ngrams,
        output_dir,
        selected_steps,
        "3gram",
        args.top_k_ngrams,
    )
    save_markdown_report(output_dir, input_dir, summary)

    print(f"\nWrote analysis to {output_dir}")


if __name__ == "__main__":
    main()
