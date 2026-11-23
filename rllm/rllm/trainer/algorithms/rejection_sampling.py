"""
Rejection sampling utilities for multi-trajectory workflows.

This module provides rejection sampling that works with TrajectoryGroups,
tracking episode-level correctness metrics while filtering based on reward variance.
"""

from dataclasses import dataclass, field

import numpy as np

from rllm.trainer.algorithms.config import RejectionSamplingConfig
from rllm.types import Episode, TrajectoryGroup


@dataclass
class RejectionSamplingMetrics:
    """Metrics tracked during rejection sampling."""

    # Episode-level correctness counts (for logging)
    solve_none: int = 0  # Tasks where ALL episodes are incorrect
    solve_all: int = 0  # Tasks where ALL episodes are correct
    solve_partial: int = 0  # Tasks where SOME episodes are correct

    # Group-level filtering counts
    groups_before_filter: int = 0
    groups_after_filter: int = 0
    groups_dropped_insufficient_trajs: int = 0

    def reset(self):
        """Reset all metrics to zero."""
        self.solve_none = 0
        self.solve_all = 0
        self.solve_partial = 0
        self.groups_before_filter = 0
        self.groups_after_filter = 0
        self.groups_dropped_insufficient_trajs = 0

    def to_dict(self, prefix: str = "batch/") -> dict:
        """Convert metrics to a dictionary for logging."""
        total_tasks = max(self.solve_none + self.solve_all + self.solve_partial, 1)

        return {
            f"{prefix}num_tasks": total_tasks,
            f"{prefix}solve_none": self.solve_none / total_tasks,
            f"{prefix}solve_all": self.solve_all / total_tasks,
            f"{prefix}solve_partial": self.solve_partial / total_tasks,
            f"{prefix}groups_before_filter": self.groups_before_filter,
            f"{prefix}groups_after_filter": self.groups_after_filter,
            f"{prefix}groups_dropped_insufficient_trajs": self.groups_dropped_insufficient_trajs,
        }


@dataclass
class RejectionSamplingState:
    """State maintained across batches for episode-level rejection sampling."""

    # Accumulated groups waiting to be processed
    accumulated_groups: list[TrajectoryGroup] = field(default_factory=list)
    # Accumulated episodes for reference
    accumulated_episodes: list[Episode] = field(default_factory=list)
    # Running metrics across accumulation
    metrics: RejectionSamplingMetrics = field(default_factory=RejectionSamplingMetrics)

    def reset(self):
        """Reset state after a batch is processed."""
        self.accumulated_groups = []
        self.accumulated_episodes = []
        self.metrics.reset()


def update_episode_metrics(
    episodes: list[Episode],
    metrics: RejectionSamplingMetrics,
) -> None:
    """
    Update episode-level metrics (solve_none/all/partial).

    Groups episodes by task_id (extracted from episode.id) and checks if
    all/none/some episodes for each task are correct.

    Args:
        episodes: List of episodes to analyze
        metrics: Metrics object to update (in-place)
    """
    # Group episodes by task_id
    episodes_by_task: dict[str, list[Episode]] = {}
    for episode in episodes:
        if len(episode.trajectories) == 0:
            continue
        # Extract task_id from episode.id format "task_id:rollout_idx"
        task_id = episode.task_id
        if task_id not in episodes_by_task:
            episodes_by_task[task_id] = []
        episodes_by_task[task_id].append(episode)

    # Check correctness for each task
    for task_id, task_episodes in episodes_by_task.items():
        correct_mask = [ep.is_correct for ep in task_episodes]
        if all(correct_mask):
            metrics.solve_all += 1
        elif any(correct_mask):
            metrics.solve_partial += 1
        else:
            metrics.solve_none += 1


def filter_groups(
    groups: list[TrajectoryGroup],
    config: RejectionSamplingConfig,
    metrics: RejectionSamplingMetrics,
) -> tuple[list[TrajectoryGroup], list[TrajectoryGroup]]:
    """
    Filter trajectory groups based on configuration.

    Args:
        groups: List of trajectory groups to filter
        config: Rejection sampling configuration
        metrics: Metrics object to update (in-place)

    Returns:
        Tuple of (filtered groups, dropped groups)
    """
    metrics.groups_before_filter += len(groups)
    filtered, dropped = [], []
    all_correct_groups = 0
    all_incorrect_groups = 0

    for group in groups:
        # Check minimum trajectories
        if len(group.trajectories) < config.min_trajs_per_group:
            metrics.groups_dropped_insufficient_trajs += 1
            dropped.append(group)
            continue

        if config.filter_uniform_groups:
            correct_mask = []
            for index, trajectory in enumerate(group.trajectories):
                metadata = group.metadata[index] if index < len(group.metadata) else {}
                is_correct = metadata.get("is_correct")
                if is_correct is None:
                    is_correct = bool(trajectory.reward and trajectory.reward > 0)
                correct_mask.append(bool(is_correct))
            all_correct = bool(correct_mask) and all(correct_mask)
            all_incorrect = bool(correct_mask) and not any(correct_mask)
            all_correct_groups += int(all_correct)
            all_incorrect_groups += int(all_incorrect)
            keep_all_correct = bool(getattr(config, "keep_all_correct_groups", False))
            if all_incorrect or (all_correct and not keep_all_correct):
                dropped.append(group)
                continue

        filtered.append(group)

    metrics.groups_after_filter += len(filtered)

    # #region debug-point A:uniform-group-filter
    try:
        import json
        import urllib.request

        urllib.request.urlopen(
            urllib.request.Request(
                "http://127.0.0.1:7777/event",
                data=json.dumps(
                    {
                        "sessionId": "lsd-routing-relaunch",
                        "runId": "post-fix",
                        "hypothesisId": "A",
                        "location": "rllm/trainer/algorithms/rejection_sampling.py:filter_groups",
                        "msg": "[DEBUG] uniform-group filtering result",
                        "data": {
                            "filter_uniform_groups": bool(config.filter_uniform_groups),
                            "keep_all_correct_groups": bool(getattr(config, "keep_all_correct_groups", False)),
                            "all_correct_groups": all_correct_groups,
                            "all_incorrect_groups": all_incorrect_groups,
                            "groups_before": len(groups),
                            "groups_after": len(filtered),
                            "groups_dropped": len(dropped),
                        },
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=2.0,
        ).read()
    except Exception:
        pass
    # #endregion
    return filtered, dropped


def filter_episodes(
    episodes: list[Episode],
    dropped_groups: list[TrajectoryGroup],
) -> list[Episode]:
    """
    Filter episodes based on dropped groups.
    """
    # obtain a flattened list of trajectory uids (unique identifier)
    trajectory_uids = [traj.uid for group in dropped_groups for traj in group.trajectories]
    filtered_episodes = []
    for episode in episodes:
        filtered_trajectories = []
        for trajectory in episode.trajectories:
            if trajectory.uid in trajectory_uids:
                continue
            filtered_trajectories.append(trajectory)
        if filtered_trajectories:
            episode.trajectories = filtered_trajectories
            filtered_episodes.append(episode)
    return filtered_episodes


def _reward_metrics(
    groups: list[TrajectoryGroup],
    filtered_groups: list[TrajectoryGroup],
    config: RejectionSamplingConfig,
    metrics: RejectionSamplingMetrics,
) -> dict:
    result = metrics.to_dict()
    result.update(
        {
            "rejection_sample/enabled": float(config.mode != "none"),
            "rejection_sample/solve_none": metrics.solve_none,
            "rejection_sample/solve_all": metrics.solve_all,
            "rejection_sample/solve_partial": metrics.solve_partial,
            "rejection_sample/total_groups": metrics.groups_before_filter,
            "rejection_sample/filtered_groups": metrics.groups_after_filter,
        }
    )

    for prefix, source_groups in (
        ("unfiltered", groups),
        ("filtered", filtered_groups),
    ):
        rewards = np.asarray(
            [
                float(trajectory.reward or 0.0)
                for group in source_groups
                for trajectory in group.trajectories
            ],
            dtype=np.float64,
        )
        if rewards.size == 0:
            continue
        result.update(
            {
                f"rejection_sample/{prefix}_reward_sum": float(rewards.sum()),
                f"rejection_sample/{prefix}_reward_count": int(rewards.size),
                f"rejection_sample/{prefix}_reward_mean": float(rewards.mean()),
                f"rejection_sample/{prefix}_reward_std": float(rewards.std()),
                f"rejection_sample/{prefix}_num_trajectories": int(rewards.size),
            }
        )
        if prefix == "unfiltered":
            result[f"rejection_sample/{prefix}_reward_min"] = float(rewards.min())
            result[f"rejection_sample/{prefix}_reward_max"] = float(rewards.max())
    return result


def apply_rejection_sampling_and_filtering(
    episodes: list[Episode],
    groups: list[TrajectoryGroup],
    config: RejectionSamplingConfig,
    state: RejectionSamplingState,
) -> tuple[list[TrajectoryGroup], list[Episode], dict]:
    """
    Apply rejection sampling to trajectory groups and episodes.

    Args:
        episodes: List of episodes (for correctness metrics)
        groups: List of trajectory groups to filter
        config: Rejection sampling configuration
        state: State object for episode-level accumulation (required for "episode" mode)

    Returns:
        Tuple of (filtered groups, filtered episodes, metrics dictionary)
        filtered groups: list of trajectory groups that pass the rejection sampling criteria
        filtered episodes: list of episodes that pass the rejection sampling criteria
        metrics: metrics dictionary for logging
    """
    if config.mode == "group":
        raise NotImplementedError("Group-level rejection sampling is not implemented yet")

    metrics = state.metrics

    # Count task outcomes before rejection removes uniform groups.
    update_episode_metrics(episodes, metrics)

    # Step 1: Filter groups and episodes based on config
    filtered_groups, dropped_groups = filter_groups(groups, config, metrics)
    filtered_episodes = filter_episodes(episodes, dropped_groups)
    rs_metrics = _reward_metrics(groups, filtered_groups, config, metrics)

    # Step 3: Apply mode-specific logic (TODO(listar2000): implement a group-level rejection sampling)
    if config.mode == "none":
        # No rejection, just return filtered groups with metrics
        return filtered_groups, filtered_episodes, rs_metrics
    elif config.mode == "episode":  # Episode-level: accumulate until we have enough partial solves
        state.accumulated_groups.extend(filtered_groups)
        state.accumulated_episodes.extend(filtered_episodes)

        # Check if we have enough partial solves
        if metrics.solve_partial >= config.min_partial_solve_tasks:
            # Ready to proceed - return accumulated groups
            return state.accumulated_groups.copy(), state.accumulated_episodes.copy(), rs_metrics
        else:  # Not enough - skip this batch
            return [], [], rs_metrics
    else:
        raise ValueError(f"Unknown rejection sampling mode: {config.mode}")
