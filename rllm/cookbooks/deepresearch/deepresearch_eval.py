"""DeepResearch evaluator using the search F1 reward."""

from __future__ import annotations

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.rewards.reward_types import RewardConfig, RewardInput
from rllm.rewards.search_reward import RewardSearchFn
from rllm.types import Episode, Task


def _task_info(task: Task | dict) -> dict:
    return (task.metadata or {}) if isinstance(task, Task) else (task or {})


@rllm.evaluator
def deepresearch_evaluator(task: Task | dict, episode: Episode) -> EvalOutput:
    info = _task_info(task)
    answer = str(episode.artifacts.get("answer", ""))
    reward_fn = RewardSearchFn(RewardConfig())
    out = reward_fn(RewardInput(task_info=info, action=answer))
    meta = dict(out.metadata or {})
    for key, value in (episode.metrics or {}).items():
        if isinstance(value, (int, float, bool)):
            meta[f"flow/{key}"] = float(value)
    return EvalOutput(reward=float(out.reward), is_correct=bool(out.is_correct), signals=[Signal(name="f1", value=float(meta.get("f1_score", 0.0))), Signal(name="accuracy", value=1.0 if out.is_correct else 0.0)], metadata=meta)
