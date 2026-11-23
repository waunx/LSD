"""Train the math agent using the Python API.

Usage (from rllm repo root)::

    python cookbooks/math/train.py rllm/backend=tinker

    # Hydra overrides:
    python cookbooks/math/train.py model.name=Qwen/Qwen3-1.7B training.group_size=4
"""

import os

import hydra
from math_eval import math_evaluator
from math_flow import math_flow
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset_name = os.environ.get("RLLM_MATH_TRAIN_DATASET", "dapo_math_17k")
    val_dataset_name = os.environ.get("RLLM_MATH_VAL_DATASET", "aime25_26_amc23")

    train_dataset = DatasetRegistry.load_dataset(train_dataset_name, "train")
    val_dataset = DatasetRegistry.load_dataset(val_dataset_name, "test")

    if train_dataset is None:
        raise RuntimeError(
            f"{train_dataset_name} train split not found. Register the HDFS DAPO parquet under RLLM_HOME first."
        )
    if val_dataset is None:
        raise RuntimeError(
            f"{val_dataset_name} test split not found. Register the HDFS validation parquets under RLLM_HOME first."
        )

    # LSD routing can use this direct row index for the full DAPO 17K split.
    # Do not synthesize source_index for filtered subsets: their row order no
    # longer matches the 17917-row base-eval map.
    if train_dataset_name == "dapo_math_17k":
        for source_index, row in enumerate(train_dataset.get_data()):
            row.setdefault("source_index", source_index)

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=math_flow,
        evaluator=math_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
