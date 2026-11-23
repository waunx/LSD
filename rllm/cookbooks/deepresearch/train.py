"""Train DeepResearch with the regular rLLM AgentTrainer path."""

from __future__ import annotations

import os

import hydra
from deepresearch_eval import deepresearch_evaluator
from deepresearch_flow import deepresearch_flow
from omegaconf import DictConfig

from rllm.data.dataset import DatasetRegistry
from rllm.trainer import AgentTrainer


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="unified", version_base=None)
def main(config: DictConfig):
    train_name = os.environ.get("RLLM_DEEPRESEARCH_TRAIN_DATASET", "cutthebill")
    train_split = os.environ.get("RLLM_DEEPRESEARCH_TRAIN_SPLIT", "train")
    val_name = os.environ.get("RLLM_DEEPRESEARCH_VAL_DATASET", "browsecomp-plus")
    val_split = os.environ.get("RLLM_DEEPRESEARCH_VAL_SPLIT", "test")

    train_dataset = DatasetRegistry.load_dataset(train_name, train_split)
    val_dataset = DatasetRegistry.load_dataset(val_name, val_split)
    if train_dataset is None:
        raise RuntimeError(f"Dataset {train_name}/{train_split} not found under RLLM_HOME={os.environ.get('RLLM_HOME')}")
    if val_dataset is None:
        raise RuntimeError(f"Dataset {val_name}/{val_split} not found under RLLM_HOME={os.environ.get('RLLM_HOME')}")

    trainer = AgentTrainer(
        backend=config.rllm.get("backend", "verl"),
        agent_flow=deepresearch_flow,
        evaluator=deepresearch_evaluator,
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
