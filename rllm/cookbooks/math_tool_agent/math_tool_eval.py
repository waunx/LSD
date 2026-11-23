"""Math tool agent evaluator: checks if the final answer matches ground truth.

Uses rllm's math grading utilities for robust comparison that handles LaTeX,
symbolic expressions, and numeric formats (e.g. ``\\frac{1}{2}`` vs ``0.5``).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

import rllm
from rllm.eval.types import EvalOutput, Signal
from rllm.rewards.math_utils.utils import grade_answer_mathd, grade_answer_sympy
from rllm.types import Episode, Task


# #region debug-point D:debug-event
def _debug_event(hypothesis_id: str, location: str, msg: str, data: dict) -> None:
    try:
        url = os.environ.get("DEBUG_SERVER_URL", "http://127.0.0.1:7777/event")
        session_id = os.environ.get("DEBUG_SESSION_ID", "math-tool-val-low")
        run_id = os.environ.get("TRAE_DEBUG_RUN_ID", "pre-fix")
        payload = {
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "msg": f"[DEBUG] {msg}",
            "data": data,
            "ts": int(time.time() * 1000),
        }
        urllib.request.urlopen(
            urllib.request.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=0.2,
        ).read()
    except Exception:
        pass
# #endregion


@rllm.evaluator
def math_tool_evaluator(task: Task | dict, episode: Episode) -> EvalOutput:
    """Grade the agent's answer against ground truth using symbolic math comparison."""
    task_data = task.metadata if isinstance(task, Task) else task
    answer_text = str(episode.artifacts.get("answer", ""))
    # math500 carries the answer in `answer` and the full solution in `ground_truth`;
    # gsm8k and others carry the answer in `ground_truth`. Prefer `answer`.
    ground_truth = str(task_data.get("answer") or task_data.get("ground_truth") or "")

    used_tool = episode.artifacts.get("used_tool")
    is_correct = (used_tool is not False) and (
        grade_answer_mathd(answer_text, ground_truth) or grade_answer_sympy(answer_text, ground_truth)
    )

    reward = 1.0 if is_correct else 0.0
    # #region debug-point D:evaluator
    _debug_event(
        "D",
        "math_tool_eval.py:evaluator",
        "episode evaluated",
        {
            "answer": answer_text[:200],
            "ground_truth": ground_truth[:200],
            "used_tool": used_tool,
            "is_correct": bool(is_correct),
            "reward": reward,
        },
    )
    # #endregion
    return EvalOutput(
        reward=reward,
        is_correct=is_correct,
        signals=[Signal(name="accuracy", value=reward)],
    )
