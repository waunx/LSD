"""Train the math tool agent using the Python API.

Usage (from rllm repo root):
    python cookbooks/math_tool_agent/train.py

Or with Hydra overrides:
    python cookbooks/math_tool_agent/train.py model.name=Qwen/Qwen3-1.7B training.group_size=4
"""

import os

import hydra
from math_tool_agent import math_tool_agent
from math_tool_eval import math_tool_evaluator
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_dataset_name = os.environ.get("RLLM_MATH_TOOL_TRAIN_DATASET", "deepscaler_math")
    val_dataset_name = os.environ.get("RLLM_MATH_TOOL_VAL_DATASET", "math500")

    train_dataset = DatasetRegistry.load_dataset(train_dataset_name, "train")
    test_dataset = DatasetRegistry.load_dataset(val_dataset_name, "test")

    if train_dataset is None:
        raise RuntimeError(
            f"{train_dataset_name} train split not found. "
            "Set RLLM_MATH_TOOL_TRAIN_DATASET or run: rllm dataset pull deepscaler_math"
        )
    if test_dataset is None:
        raise RuntimeError(
            f"{val_dataset_name} test split not found. "
            "Set RLLM_MATH_TOOL_VAL_DATASET or run: rllm dataset pull math500"
        )

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "tinker"),
        agent_flow=math_tool_agent,
        evaluator=math_tool_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=test_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
