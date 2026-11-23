import pytest

from rllm.trainer.algorithms.config import RejectionSamplingConfig
from rllm.trainer.algorithms.metrics import (
    add_fully_async_training_aliases,
    add_fully_async_validation_aliases,
    episode_fully_async_metrics,
)
from rllm.trainer.algorithms.rejection_sampling import (
    RejectionSamplingState,
    apply_rejection_sampling_and_filtering,
)
from rllm.types import Episode, Step, Trajectory, TrajectoryGroup
from rllm.workflows.workflow import TerminationReason


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return text.split()


def _episode():
    return Episode(
        id="task:0",
        is_correct=True,
        termination_reason=TerminationReason.ENV_DONE,
        trajectories=[
            Trajectory(
                name="deepresearch",
                reward=0.75,
                steps=[
                    Step(prompt_ids=[1], response_ids=[2, 3]),
                    Step(prompt_ids=[1, 2], response_ids=[4]),
                ],
            )
        ],
        artifacts={
            "messages": [
                {"role": "tool", "content": "one two three"},
                {"role": "assistant", "content": "answer"},
            ]
        },
        metrics={
            "total_tool_calls": 2.0,
            "total_query_length": 10.0,
            "total_refine_time": 4.0,
            "total_refine_error": 1.0,
            "total_tool_return_error": 0.0,
            "f1_score": 0.75,
            "time/agentflow_llm_sum_s": 3.0,
        },
    )


def test_episode_fully_async_metrics_matches_legacy_names():
    metrics = episode_fully_async_metrics(_episode(), _Tokenizer())

    assert metrics["raw_reward"] == 0.75
    assert metrics["is_correct"] == 1.0
    assert metrics["num_turns"] == 2.0
    assert metrics["total_completion_tokens"] == 3.0
    assert metrics["avg_completion_tokens_per_turn"] == 1.5
    assert metrics["total_tool_tokens"] == 3.0
    assert metrics["avg_query_length"] == 5.0
    assert metrics["avg_refine_time"] == 2.0
    assert metrics["refine_error_detected"] == 1.0


def test_training_aliases_keep_native_metrics_and_add_legacy_names():
    metrics = {
        "grad_norm": 0.5,
        "lr": 1e-6,
        "lsd/hard/pg_clipfrac": 0.1,
        "lsd/hard/pg_clipfrac_lower": 0.2,
        "lsd/hard/ppo_kl": 0.3,
        "offpolicy/kl": 0.4,
        "batch/solve_partial": 1.0,
    }

    add_fully_async_training_aliases(metrics, [_episode()], _Tokenizer())

    assert metrics["actor/grad_norm"] == 0.5
    assert metrics["actor/lr"] == 1e-6
    assert metrics["actor/pg_clipfrac"] == 0.1
    assert metrics["actor/pg_clipfrac_lower"] == 0.2
    assert metrics["actor/ppo_kl"] == 0.3
    assert metrics["rollout_corr/kl"] == 0.4
    assert metrics["custom/raw_reward/avg"] == 0.75
    assert metrics["custom/total_tool_tokens/max"] == 3.0
    assert metrics["rejection_sample/solve_partial"] == 1.0


def test_validation_aliases_use_unscoped_fully_async_keys():
    metrics = {"time/testing": 12.0}

    add_fully_async_validation_aliases(
        metrics,
        [_episode()],
        tokenizer=_Tokenizer(),
        weight_version=7,
    )

    assert metrics["val/avg_reward"] == 0.75
    assert metrics["val/num_samples"] == 1
    assert metrics["val/f1_score"] == 0.75
    assert metrics["val/param_version"] == 7
    assert metrics["timing_s/validation"] == 12.0


def _task_group(task_id: str, outcomes: list[bool]):
    episodes = []
    trajectories = []
    metadata = []
    for index, is_correct in enumerate(outcomes):
        trajectory = Trajectory(
            name="deepresearch",
            reward=float(is_correct),
        )
        episodes.append(
            Episode(
                id=f"{task_id}:{index}",
                is_correct=is_correct,
                trajectories=[trajectory],
            )
        )
        trajectories.append(trajectory)
        metadata.append({"is_correct": is_correct})
    return episodes, TrajectoryGroup(
        group_id=f"{task_id}:deepresearch",
        trajectories=trajectories,
        metadata=metadata,
    )


def test_uniform_group_filter_matches_fully_async_rejection_sampling():
    none_episodes, none_group = _task_group("none", [False, False])
    all_episodes, all_group = _task_group("all", [True, True])
    partial_episodes, partial_group = _task_group(
        "partial", [False, True]
    )
    episodes = none_episodes + all_episodes + partial_episodes
    groups = [none_group, all_group, partial_group]
    config = RejectionSamplingConfig(
        mode="episode",
        min_partial_solve_tasks=1,
        filter_uniform_groups=True,
    )

    filtered_groups, filtered_episodes, metrics = (
        apply_rejection_sampling_and_filtering(
            episodes, groups, config, RejectionSamplingState()
        )
    )

    assert [group.task_id for group in filtered_groups] == ["partial"]
    assert {episode.task_id for episode in filtered_episodes} == {"partial"}
    assert metrics["rejection_sample/solve_none"] == 1
    assert metrics["rejection_sample/solve_all"] == 1
    assert metrics["rejection_sample/solve_partial"] == 1
    assert metrics["rejection_sample/unfiltered_num_trajectories"] == 6
    assert metrics["rejection_sample/filtered_num_trajectories"] == 2
    assert metrics["rejection_sample/unfiltered_reward_mean"] == pytest.approx(0.5)
    assert metrics["rejection_sample/filtered_reward_mean"] == pytest.approx(0.5)


def test_uniform_group_filter_can_keep_solve_all_for_lsd():
    none_episodes, none_group = _task_group("none", [False, False])
    all_episodes, all_group = _task_group("all", [True, True])
    partial_episodes, partial_group = _task_group(
        "partial", [False, True]
    )
    episodes = none_episodes + all_episodes + partial_episodes
    groups = [none_group, all_group, partial_group]
    config = RejectionSamplingConfig(
        mode="episode",
        min_partial_solve_tasks=1,
        filter_uniform_groups=True,
        keep_all_correct_groups=True,
    )

    filtered_groups, filtered_episodes, metrics = (
        apply_rejection_sampling_and_filtering(
            episodes, groups, config, RejectionSamplingState()
        )
    )

    assert [group.task_id for group in filtered_groups] == ["all", "partial"]
    assert {episode.task_id for episode in filtered_episodes} == {"all", "partial"}
    assert metrics["rejection_sample/solve_none"] == 1
    assert metrics["rejection_sample/solve_all"] == 1
    assert metrics["rejection_sample/solve_partial"] == 1
    assert metrics["rejection_sample/unfiltered_num_trajectories"] == 6
    assert metrics["rejection_sample/filtered_num_trajectories"] == 4
    assert metrics["rejection_sample/filtered_reward_mean"] == pytest.approx(0.75)
