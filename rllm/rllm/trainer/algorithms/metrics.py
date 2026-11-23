"""
Common metric utilities for rLLM. Work with TrajectoryGroups and Episodes.
For backend-dependent metrics, please implement them in the backend-specific modules.
TODO(listar2000): think hard about what are the actually important metrics in agentic RL settings.
"""

from collections import defaultdict
from collections.abc import Callable
from numbers import Number
from typing import Any

import numpy as np

from rllm.types import Episode, TrajectoryGroup


_FULLY_ASYNC_EPISODE_METRICS = {
    "avg_completion_tokens_per_turn",
    "avg_query_length",
    "avg_refine_time",
    "avg_tool_tokens_per_call",
    "best_match",
    "duplicate_search_detected",
    "exact_match",
    "excessive_parallel_calls",
    "extracted_answer",
    "f1_score",
    "f1_threshold",
    "is_correct",
    "merged_step",
    "num_turns",
    "overlong",
    "precision",
    "raw_reward",
    "recall",
    "refine_error_detected",
    "tool_error_detected",
    "total_completion_tokens",
    "total_generation_time",
    "total_parse_tool_args_error",
    "total_query_length",
    "total_refine_time",
    "total_tool_calls",
    "total_tool_return_error",
    "total_tool_tokens",
    "total_tool_wait_time",
}


def _compat_float(value: Any) -> float | None:
    if isinstance(value, Number):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _episode_reward(episode: Episode) -> float:
    rewards = [
        float(trajectory.reward)
        for trajectory in episode.trajectories
        if trajectory.reward is not None
    ]
    return float(np.mean(rewards)) if rewards else 0.0


def episode_fully_async_metrics(
    episode: Episode,
    tokenizer: Any | None = None,
) -> dict[str, float]:
    """Build legacy Fully Async metrics from one unified Episode."""

    values: dict[str, float] = {}
    for key, value in episode.metrics.items():
        key = key.removeprefix("flow/")
        if key not in _FULLY_ASYNC_EPISODE_METRICS:
            continue
        numeric = _compat_float(value)
        if numeric is not None:
            values.setdefault(key, numeric)

    refine_errors = _compat_float(
        episode.metrics.get(
            "total_refine_error",
            episode.metrics.get("flow/total_refine_error", 0.0),
        )
    )
    num_turns = sum(len(trajectory.steps) for trajectory in episode.trajectories)
    completion_tokens = sum(
        len(step.response_ids)
        for trajectory in episode.trajectories
        for step in trajectory.steps
    )
    tool_calls = values.get("total_tool_calls", 0.0)
    values.update(
        {
            "raw_reward": _episode_reward(episode),
            "is_correct": float(episode.is_correct),
            "num_turns": float(num_turns),
            "merged_step": float(num_turns),
            "total_completion_tokens": float(completion_tokens),
            "avg_completion_tokens_per_turn": completion_tokens
            / max(num_turns, 1),
            "tool_error_detected": float(
                values.get("total_tool_return_error", 0.0) > 0
            ),
            "refine_error_detected": float(
                (refine_errors or 0.0) > 0
            ),
            "is_partial": 0.0,
        }
    )
    for key in (
        "duplicate_search_detected",
        "excessive_parallel_calls",
        "total_parse_tool_args_error",
        "total_query_length",
        "total_refine_time",
        "total_tool_calls",
        "total_tool_return_error",
        "total_tool_wait_time",
    ):
        values.setdefault(key, 0.0)

    reason = getattr(episode.termination_reason, "value", episode.termination_reason)
    values["overlong"] = float(
        reason
        in {
            "max_prompt_length_exceeded",
            "max_response_length_exceeded",
            "max_turns_exceeded",
        }
    )
    generation_time = _compat_float(
        episode.metrics.get("time/agentflow_llm_sum_s")
    )
    if generation_time is not None:
        values["total_generation_time"] = generation_time
    else:
        values["total_generation_time"] = 0.0
    if tool_calls:
        values.setdefault(
            "avg_query_length",
            values.get("total_query_length", 0.0) / tool_calls,
        )
        values.setdefault(
            "avg_refine_time",
            values.get("total_refine_time", 0.0) / tool_calls,
        )
    values.setdefault("avg_query_length", 0.0)
    values.setdefault("avg_refine_time", 0.0)

    messages = episode.artifacts.get("messages")
    if tokenizer is not None and isinstance(messages, list):
        tool_tokens = sum(
            len(
                tokenizer.encode(
                    str(message.get("content") or ""),
                    add_special_tokens=False,
                )
            )
            for message in messages
            if isinstance(message, dict) and message.get("role") == "tool"
        )
        values["total_tool_tokens"] = float(tool_tokens)
        values["avg_tool_tokens_per_call"] = tool_tokens / max(tool_calls, 1.0)
    values.setdefault("total_tool_tokens", 0.0)
    values.setdefault("avg_tool_tokens_per_call", 0.0)
    return values


def add_fully_async_training_aliases(
    metrics: dict[str, Any],
    episodes: list[Episode],
    tokenizer: Any | None = None,
) -> None:
    """Add comparable legacy names while retaining unified-trainer metrics."""

    aliases = {
        "actor/grad_norm": "grad_norm",
        "actor/lr": "lr",
        "actor/pg_clipfrac": "lsd/hard/pg_clipfrac",
        "actor/pg_clipfrac_lower": "lsd/hard/pg_clipfrac_lower",
        "actor/ppo_kl": "lsd/hard/ppo_kl",
        "perf/mfu/actor": "mfu",
        "timing_s/old_log_prob": "timing_s/old_log_probs",
        "timing_s/param_sync": "timing_s/update_weights",
    }
    for target, source in aliases.items():
        if source in metrics:
            metrics.setdefault(target, metrics[source])
    metrics.setdefault("actor/kl_loss", metrics.get("kl_loss", 0.0))

    for key, value in list(metrics.items()):
        if key.startswith("offpolicy/"):
            metrics.setdefault(
                f"rollout_corr/{key.removeprefix('offpolicy/')}", value
            )

    aggregated: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        for key, value in episode_fully_async_metrics(
            episode, tokenizer
        ).items():
            aggregated[key].append(value)
    for key, values in aggregated.items():
        metrics[f"custom/{key}/avg"] = float(np.mean(values))
        metrics[f"custom/{key}/min"] = float(np.min(values))
        metrics[f"custom/{key}/max"] = float(np.max(values))

    rewards = [_episode_reward(episode) for episode in episodes]
    if rewards:
        metrics.setdefault("rejection_sample/enabled", 1.0)
        metrics.setdefault(
            "rejection_sample/unfiltered_num_trajectories", len(rewards)
        )
        metrics.setdefault(
            "rejection_sample/unfiltered_reward_count", len(rewards)
        )
        metrics.setdefault(
            "rejection_sample/unfiltered_reward_sum", float(np.sum(rewards))
        )
        metrics.setdefault(
            "rejection_sample/unfiltered_reward_mean", float(np.mean(rewards))
        )
        metrics.setdefault(
            "rejection_sample/unfiltered_reward_std", float(np.std(rewards))
        )
        metrics.setdefault(
            "rejection_sample/unfiltered_reward_min", float(np.min(rewards))
        )
        metrics.setdefault(
            "rejection_sample/unfiltered_reward_max", float(np.max(rewards))
        )

    rejection_aliases = {
        "rejection_sample/solve_all": "batch/solve_all",
        "rejection_sample/solve_none": "batch/solve_none",
        "rejection_sample/solve_partial": "batch/solve_partial",
        "rejection_sample/total_groups": "batch/groups_before_filter",
        "rejection_sample/filtered_groups": "batch/groups_after_filter",
    }
    for target, source in rejection_aliases.items():
        if source in metrics:
            metrics.setdefault(target, metrics[source])


def add_fully_async_validation_aliases(
    metrics: dict[str, Any],
    episodes: list[Episode],
    *,
    tokenizer: Any | None = None,
    weight_version: int = 0,
) -> None:
    """Add legacy unscoped validation names used by Fully Async."""

    aggregated: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        for key, value in episode_fully_async_metrics(
            episode, tokenizer
        ).items():
            aggregated[key].append(value)
    for key, values in aggregated.items():
        metrics[f"val/{key}"] = float(np.mean(values))

    metrics["val/num_samples"] = len(episodes)
    metrics["val/param_version"] = weight_version
    metrics["val/param_version_start"] = weight_version
    metrics["val/param_version_end"] = weight_version
    metrics["val/validated_version"] = weight_version
    if "val/raw_reward" in metrics:
        metrics["val/avg_reward"] = metrics["val/raw_reward"]
    if "time/testing" in metrics:
        metrics["timing_s/validation"] = metrics["time/testing"]


def reduce_metrics_by_trajectory_name(trajectory_groups: list[TrajectoryGroup], prefix: str = "reward", include_fraction_zero: bool = False) -> dict:
    """
    Reduce reward metrics by trajectory name.

    Args:
        trajectory_groups: List of TrajectoryGroup objects
        prefix: Prefix for the metric keys
        include_fraction_zero: Whether to include the fraction of zero summary
    Returns:
        Dictionary of metrics by trajectory name
    """
    metrics, rewards_by_traj_name = {}, {}
    for group in trajectory_groups:
        for traj in group.trajectories:
            if traj.name not in rewards_by_traj_name:
                rewards_by_traj_name[traj.name] = []
            if traj.reward is not None:
                rewards_by_traj_name[traj.name].append(traj.reward)

    for traj_name, rewards in rewards_by_traj_name.items():
        if not rewards:
            continue
        metrics[f"{prefix}/{traj_name}/mean"] = np.mean(rewards)
        metrics[f"{prefix}/{traj_name}/max"] = np.max(rewards)
        metrics[f"{prefix}/{traj_name}/min"] = np.min(rewards)
        metrics[f"{prefix}/{traj_name}/std"] = np.std(rewards)
        if include_fraction_zero:
            metrics[f"{prefix}/{traj_name}/fraction_zero"] = np.sum(np.abs(rewards) < 1e-8) / len(rewards)

    return metrics


def reduce_metrics_lists(metrics: dict, reduce_fn: Callable = np.mean):
    """A simple helper function that average any metric value that is a list into a scalar.

    Args:
        metrics: Dictionary of metrics
    """
    for key, value in metrics.items():
        if isinstance(value, list):
            metrics[key] = reduce_fn(value)
