#!/usr/bin/env python3
"""Normalize DeepScaleR eval parquet files for verl dataset concatenation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUTS = [
    "/data/wx_data/datasets/DeepScaleR/aime.parquet",
    "/data/wx_data/datasets/DeepScaleR/amc.parquet",
    "/data/wx_data/datasets/DeepScaleR/math.parquet",
    "/data/wx_data/datasets/DeepScaleR/minerva.parquet",
    "/data/wx_data/datasets/DeepScaleR/olympiad_bench.parquet",
]

DATA_SOURCE_BY_STEM = {
    "aime": "AIME",
    "amc": "AMC",
    "math": "MATH",
    "minerva": "MINERVA",
    "olympiad_bench": "OLYMPIAD_BENCH",
}


def normalize_ground_truth(value: Any) -> str:
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""

    if value is None:
        return ""

    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return str(int(value))
        return str(value)

    return str(value).strip()


def normalize_reward_model(value: Any) -> dict[str, str]:
    if isinstance(value, dict):
        ground_truth = normalize_ground_truth(value.get("ground_truth", ""))
        style = str(value.get("style", "rule"))
    else:
        ground_truth = normalize_ground_truth(value)
        style = "rule"

    return {"ground_truth": ground_truth, "style": style}


def convert_file(input_path: Path, output_dir: Path) -> Path:
    df = pd.read_parquet(input_path)

    if "reward_model" not in df.columns:
        raise ValueError(f"{input_path} does not contain reward_model column")

    df = df.copy()
    df["reward_model"] = df["reward_model"].map(normalize_reward_model)
    data_source = DATA_SOURCE_BY_STEM.get(input_path.stem, input_path.stem.upper())
    df["data_source"] = data_source

    output_path = output_dir / input_path.name
    df.to_parquet(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default="/data/wx_data/datasets/LSD",
        help="Directory for normalized parquet files.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        default=DEFAULT_INPUTS,
        help="Input parquet files. Defaults to DeepScaleR eval files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for raw_input in args.inputs:
        input_path = Path(raw_input)
        output_path = convert_file(input_path, output_dir)
        print(f"{input_path} -> {output_path}")


if __name__ == "__main__":
    main()
