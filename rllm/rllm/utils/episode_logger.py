"""Episode JSON logger for saving rollout records."""

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rllm.types import Episode


_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+(?:\.\d+)?|\\boxed|\\frac|\\sqrt|[=+\-*/^<>]+")

_REFLECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern, re.IGNORECASE))
    for name, pattern in (
        ("wait", r"\bwait\b|等等"),
        ("hmm", r"\bhm+\b|\bhmm+\b|嗯"),
        ("actually", r"\bactually\b|实际上"),
        ("recheck", r"\bre-?check\b|\bdouble-?check\b|\bcheck again\b|重新检查"),
        ("reconsider", r"\breconsider\b|\brethink\b|重新考虑"),
        ("mistake", r"\bmistake\b|\berror\b|\bwrong\b|错误"),
        ("however", r"\bhowever\b|\bbut\b|但是|然而"),
        ("alternatively", r"\balternatively\b|\bon the other hand\b|另一方面"),
        ("verify", r"\bverify\b|\bverification\b|\bconfirm\b|验证|确认"),
        ("lets_check", r"\blet'?s check\b|\blet me check\b"),
        ("lets_verify", r"\blet'?s verify\b|\blet me verify\b"),
        ("maybe", r"\bmaybe\b|\bperhaps\b|\bseems\b|\blikely\b|也许|可能"),
    )
)


class EpisodeLogger:
    """Logger to save full episodes or lightweight per-episode summaries."""

    def __init__(
        self,
        base_dir: str,
        subdirectory: str = "episodes",
        *,
        modes: Iterable[str] | None = None,
        summary_only: bool = False,
        include_text: bool = True,
        file_format: str = "json",
    ):
        """Initialize the episode logger.

        Args:
            base_dir: Base directory for episode logs. Can be configured via
                     config.trainer.episode_log_dir
                     (default: "logs/${trainer.project_name}/${trainer.experiment_name}")
            subdirectory: Subdirectory within base_dir for episodes (default: "episodes")
                         Final path will be: {base_dir}/{subdirectory}/
            modes: Modes to log, such as {"train", "val"}. None logs all modes.
            summary_only: Save compact metadata instead of full trajectories.
            include_text: Include response text fields when writing full trajectories.
            file_format: "json" for per-episode files or "jsonl" for one file per step.
        """
        if file_format not in {"json", "jsonl"}:
            raise ValueError(f"Unsupported episode log file_format: {file_format}")

        self.log_dir = Path(base_dir) / subdirectory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.modes = set(modes) if modes is not None else None
        self.summary_only = summary_only
        self.include_text = include_text
        self.file_format = file_format

    def should_log_mode(self, mode: str) -> bool:
        return self.modes is None or mode in self.modes

    @staticmethod
    def compute_task_hash(task: Any, length: int = 8) -> str:
        """Compute a hash from the task data.

        Args:
            task: The task dictionary or data
            length: Length of the hash to use (default 8 chars)

        Returns:
            Hash string
        """
        # Convert task to a stable string representation
        task_str = json.dumps(task, sort_keys=True, default=str)
        # Compute SHA256 hash
        hash_obj = hashlib.sha256(task_str.encode("utf-8"))
        # Return first `length` characters of hex digest
        return hash_obj.hexdigest()[:length]

    def get_step_dir(self, step: int, mode: str = "train", epoch: int = 0) -> Path:
        """Get the directory path for a specific training or validation step.

        Args:
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0

        Returns:
            Path object for the step directory
        """
        step_dir = self.log_dir / f"{mode}_step_{step}_epoch_{epoch}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def get_episode_filename(self, episode: Episode, step: int) -> str:
        """Generate filename for an episode.

        Format: episode_hash{task_hash}_id{episode_id}.json

        Args:
            episode: The episode to save
            step: Current training step (not used in filename, but kept for compatibility)

        Returns:
            Filename string
        """
        task_hash = self.compute_task_hash(episode.task)
        # Clean episode_id to make it filesystem-safe
        episode_id_safe = str(episode.id).replace(":", "_").replace("/", "_")

        suffix = "jsonl" if self.file_format == "jsonl" else "json"
        filename = f"episode_hash{task_hash}_id{episode_id_safe}.{suffix}"
        return filename

    @staticmethod
    def _task_value(task: Any, key: str, default: Any = None) -> Any:
        if isinstance(task, dict):
            return task.get(key, default)
        return getattr(task, key, default)

    @staticmethod
    def _termination_reason_value(termination_reason: Any) -> Any:
        return termination_reason.value if termination_reason is not None and hasattr(termination_reason, "value") else termination_reason

    @staticmethod
    def _split_episode_id(episode: Episode) -> tuple[str, str | None]:
        parts = str(episode.id).split(":", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return str(episode.id), None

    @staticmethod
    def _episode_reward(episode: Episode) -> float:
        reward = 0.0
        for trajectory in episode.trajectories:
            if trajectory.reward is not None:
                reward += float(trajectory.reward)
                continue
            reward += sum(float(step.reward) for step in trajectory.steps)
        return reward

    @staticmethod
    def _episode_lengths(episode: Episode) -> dict[str, Any]:
        prompt_total = 0
        response_total = 0
        step_prompt_lengths = []
        step_response_lengths = []

        for trajectory in episode.trajectories:
            for step in trajectory.steps:
                prompt_ids = step.prompt_ids
                response_ids = step.response_ids
                model_output = getattr(step, "model_output", None)

                if not prompt_ids and model_output is not None:
                    prompt_ids = getattr(model_output, "prompt_ids", None) or []
                if not response_ids and model_output is not None:
                    response_ids = getattr(model_output, "completion_ids", None) or []

                prompt_len = len(prompt_ids)
                response_len = len(response_ids)
                prompt_total += prompt_len
                response_total += response_len
                step_prompt_lengths.append(prompt_len)
                step_response_lengths.append(response_len)

        return {
            "prompt_length": prompt_total,
            "response_length": response_total,
            "step_prompt_lengths": step_prompt_lengths,
            "step_response_lengths": step_response_lengths,
            "n_turns": len(step_response_lengths),
        }

    @staticmethod
    def _step_response_text(step: Any) -> str:
        if getattr(step, "model_response", None):
            return str(step.model_response)

        model_output = getattr(step, "model_output", None)
        if model_output is not None and getattr(model_output, "content", None):
            return str(model_output.content)

        chat_completions = getattr(step, "chat_completions", None) or []
        for message in reversed(chat_completions):
            if isinstance(message, dict) and message.get("role") == "assistant":
                content = message.get("content")
                if content:
                    return str(content)
        return ""

    @classmethod
    def _episode_response_text(cls, episode: Episode) -> str:
        texts = []
        for trajectory in episode.trajectories:
            for step in trajectory.steps:
                text = cls._step_response_text(step)
                if text:
                    texts.append(text)
        return "\n".join(texts)

    @staticmethod
    def _tokenize_response_text(text: str) -> list[str]:
        return [token.lower() for token in _TOKEN_RE.findall(text or "")]

    @staticmethod
    def _ngram_stats(tokens: list[str], n: int) -> dict[str, int | float]:
        if len(tokens) < n:
            return {
                "total": 0,
                "unique": 0,
                "repeated": 0,
                "repetition_rate": 0.0,
                "distinct_ratio": 0.0,
                "max_count": 0,
            }

        grams = list(zip(*(tokens[i:] for i in range(n))))
        counter = Counter(grams)
        total = len(grams)
        unique = len(counter)
        repeated = total - unique
        return {
            "total": total,
            "unique": unique,
            "repeated": repeated,
            "repetition_rate": repeated / total if total else 0.0,
            "distinct_ratio": unique / total if total else 0.0,
            "max_count": max(counter.values()) if counter else 0,
        }

    @classmethod
    def _response_pattern_stats(cls, episode: Episode) -> dict[str, Any]:
        text = cls._episode_response_text(episode)
        tokens = cls._tokenize_response_text(text)
        reflection_word_counts = {
            name: sum(1 for _ in pattern.finditer(text))
            for name, pattern in _REFLECTION_PATTERNS
        }
        reflection_total = sum(reflection_word_counts.values())
        ngram_stats = {
            f"{n}gram": cls._ngram_stats(tokens, n)
            for n in (1, 2, 3, 4, 20)
        }

        return {
            "response_char_length": len(text),
            "response_word_count": len(tokens),
            "reflection_word_counts": reflection_word_counts,
            "reflection_word_total": reflection_total,
            "reflection_per_1k_words": reflection_total * 1000.0 / max(1, len(tokens)),
            "ngram_stats": ngram_stats,
            "repetition_ratio_20gram": ngram_stats["20gram"]["repetition_rate"],
            "most_repeated_20gram": ngram_stats["20gram"]["max_count"],
        }

    def _build_summary_episode_data(self, episode: Episode, step: int, mode: str, epoch: int) -> dict[str, Any]:
        task = episode.task
        task_row_id = self._task_value(task, "id")
        data_source = episode.info.get("data_source") or self._task_value(task, "data_source", "unknown")
        task_id, rollout_idx = self._split_episode_id(episode)

        return {
            "training_step": step,
            "epoch": epoch,
            "mode": mode,
            "episode_id": episode.id,
            "task_id": task_id,
            "rollout_idx": rollout_idx,
            "source_id": task_row_id,
            "data_source": data_source,
            "task_hash": self.compute_task_hash(task),
            "is_correct": bool(episode.is_correct),
            "reward": self._episode_reward(episode),
            "termination_reason": self._termination_reason_value(episode.termination_reason),
            **self._episode_lengths(episode),
            **self._response_pattern_stats(episode),
        }

    def _build_full_episode_data(self, episode: Episode, step: int, mode: str, epoch: int) -> dict[str, Any]:
        episode_data = {
            "training_step": step,
            "epoch": epoch,
            "mode": mode,
            "episode_id": episode.id,
            "session_id": episode.session_id,
            "task": episode.task,
            "task_hash": self.compute_task_hash(episode.task),
            "is_correct": episode.is_correct,
            "termination_reason": self._termination_reason_value(episode.termination_reason),
            "metrics": episode.metrics,
            "metadata": episode.metadata,
            "timing": episode.info.get("timing", {}),
            "trajectories": [],
        }

        for traj in episode.trajectories:
            steps = []
            for step_obj in traj.steps:
                step_data = {
                    "reward": step_obj.reward,
                    "done": step_obj.done,
                    "prompt_length": len(step_obj.prompt_ids),
                    "response_length": len(step_obj.response_ids),
                    "timing": step_obj.info.get("timing", {}),
                }
                if self.include_text:
                    step_data.update(
                        {
                            "observation": step_obj.observation,
                            "thought": step_obj.thought,
                            "action": step_obj.action,
                            "model_response": step_obj.model_response,
                            "chat_completions": step_obj.chat_completions,
                        }
                    )
                steps.append(step_data)

            traj_data = {
                "name": traj.name,
                "uid": traj.uid,
                "reward": traj.reward,
                "num_steps": len(traj.steps),
                "timing": traj.info.get("timing", {}),
                "steps": steps,
            }
            episode_data["trajectories"].append(traj_data)

        return episode_data

    def _build_episode_data(self, episode: Episode, step: int, mode: str, epoch: int) -> dict[str, Any]:
        if self.summary_only:
            return self._build_summary_episode_data(episode, step, mode, epoch)
        return self._build_full_episode_data(episode, step, mode, epoch)

    def log_episode(self, episode: Episode, step: int, mode: str = "train", epoch: int = 0):
        """Log a single episode to its own JSON file in a step-specific directory.

        Args:
            episode: The episode to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        if not self.should_log_mode(mode):
            return

        episode_data = self._build_episode_data(episode, step, mode, epoch)

        # Write to individual file in step-specific directory
        step_dir = self.get_step_dir(step, mode, epoch)
        if self.file_format == "jsonl":
            filepath = step_dir / "episodes.jsonl"
            write_mode = "a"
            payload = json.dumps(episode_data, default=str) + "\n"
        else:
            filename = self.get_episode_filename(episode, step)
            filepath = step_dir / filename
            write_mode = "w"
            payload = json.dumps(episode_data, indent=2, default=str) + "\n"

        try:
            with open(filepath, write_mode) as f:
                f.write(payload)
                f.flush()  # Ensure data is written to disk
        except Exception as e:
            print(f"Error writing episode to {filepath}: {e}")
            import traceback

            traceback.print_exc()
            raise

    def log_episodes(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0):
        """Log multiple episodes, each to its own file.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
        """
        if not self.should_log_mode(mode):
            return

        print(f"[EpisodeLogger] Logging {len(episodes)} episodes for step={step}, mode={mode}, epoch={epoch}")
        for i, episode in enumerate(episodes):
            try:
                self.log_episode(episode, step, mode, epoch)
                print(f"[EpisodeLogger] Successfully logged episode {i + 1}/{len(episodes)}: {episode.id}")
            except Exception as e:
                print(f"[EpisodeLogger] Failed to log episode {i + 1}/{len(episodes)}: {e}")
                raise

    def log_episodes_batch(self, episodes: list[Episode], step: int, mode: str = "train", epoch: int = 0, batch_summary: bool = True):
        """Log multiple episodes and optionally create a batch summary in step-specific directory.

        Args:
            episodes: List of episodes to log
            step: Current training/validation step
            mode: Mode identifier ('train' or 'val'), defaults to 'train'
            epoch: Current epoch number, defaults to 0
            batch_summary: Whether to create a summary file for the batch
        """
        if not self.should_log_mode(mode):
            return

        # Log individual episodes
        self.log_episodes(episodes, step, mode, epoch)

        # Optionally create batch summary in step-specific directory
        if batch_summary and episodes:
            summary_data = {
                "training_step": step,
                "epoch": epoch,
                "mode": mode,
                "num_episodes": len(episodes),
                "episode_files": ["episodes.jsonl"] if self.file_format == "jsonl" else [self.get_episode_filename(ep, step) for ep in episodes],
                "summary_stats": {
                    "total_correct": sum(1 for ep in episodes if ep.is_correct),
                    "total_incorrect": sum(1 for ep in episodes if not ep.is_correct),
                    "accuracy": sum(1 for ep in episodes if ep.is_correct) / len(episodes) if episodes else 0,
                    "avg_trajectories_per_episode": sum(len(ep.trajectories) for ep in episodes) / len(episodes) if episodes else 0,
                },
            }

            step_dir = self.get_step_dir(step, mode, epoch)
            summary_file = step_dir / "batch_summary.json"
            with open(summary_file, "w") as f:
                json.dump(summary_data, f, indent=2)
