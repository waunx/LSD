#!/usr/bin/env python3
"""Prepare agentica-org/DeepScaleR-Preview-Dataset for LSD/verl training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


INSTRUCTION_FOLLOWING = "Let's think step by step and output the final answer within \\boxed{}."
REWARD_DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
SOURCE_DATASET = "agentica-org/DeepScaleR-Preview-Dataset"


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_row(item: dict[str, Any], split: str, index: int) -> dict[str, Any]:
    problem = str(item["problem"]).strip()
    answer = normalize_answer(item.get("answer"))

    return {
        "data_source": REWARD_DATA_SOURCE,
        "prompt": [
            {
                "role": "user",
                "content": f"{problem} {INSTRUCTION_FOLLOWING}",
            }
        ],
        "ability": "math",
        "reward_model": {
            "style": "rule",
            "ground_truth": answer,
        },
        "extra_info": {
            "split": split,
            "index": index,
            "source_dataset": SOURCE_DATASET,
        },
    }


def load_items(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected a JSON list in {input_path}, got {type(data).__name__}")
    return data


def write_json_example(path: Path, row: dict[str, Any]) -> None:
    with path.open("w") as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/opt/tiger/datasets/DeepScaleR-Preview-Dataset/deepscaler.json",
        help="Path to the raw DeepScaleR JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        default="/opt/tiger/datasets/DeepScaleR-Preview-Dataset",
        help="Directory for prepared parquet files.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=512,
        help="Number of examples reserved for validation from the end of the file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_items(input_path)
    if args.val_size <= 0 or args.val_size >= len(items):
        raise ValueError(f"--val-size must be in [1, {len(items) - 1}], got {args.val_size}")

    split_at = len(items) - args.val_size
    train_items = items[:split_at]
    val_items = items[split_at:]

    train_rows = [build_row(item, "train", idx) for idx, item in enumerate(train_items)]
    val_rows = [build_row(item, "validation", split_at + idx) for idx, item in enumerate(val_items)]

    train_path = output_dir / "deepscaler_train_math.parquet"
    val_path = output_dir / "deepscaler_val_math.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    write_json_example(output_dir / "deepscaler_train_math_example.json", train_rows[0])
    write_json_example(output_dir / "deepscaler_val_math_example.json", val_rows[0])

    print(f"wrote {len(train_rows)} train rows -> {train_path}")
    print(f"wrote {len(val_rows)} validation rows -> {val_path}")


if __name__ == "__main__":
    main()
