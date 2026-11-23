"""Multi-turn math agent with calculator tool.

A multi-turn agent that solves math problems step by step using a calculator
tool via native OpenAI function-calling.  Works identically for eval and
training (the gateway handles trace capture).
"""

from __future__ import annotations

import ast
import json
import logging
import math
import os
import re
import time
import urllib.request

from openai import AsyncOpenAI

import rllm
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)

MAX_TURNS = 8


# #region debug-point A:debug-event
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

SYSTEM_PROMPT = """\
You are a math assistant that solves competition math problems step by step.
You have access to a calculator tool and you MUST use it.

IMPORTANT rules you must follow:
1. You MUST call the calculator tool at least once before giving your final answer. \
Answers given without any prior tool call will be marked wrong.
2. Do NOT perform arithmetic in your head — every computation must go through the calculator.
3. Break the problem into small steps. Make one tool call per step, then reason about the result.
4. When you have the final answer, put it in \\boxed{ANSWER} in your response.
"""

_CALCULATOR_DESCRIPTION = (
    "Evaluate a mathematical expression. Supports operators +, -, *, /, //, ** (power), "
    "% (modulo), and parentheses. Functions: abs, round, min, max, sum, pow, sqrt, exp, "
    "log, log2, log10, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh, floor, "
    "ceil, trunc, factorial, gcd, lcm, comb (aka binom), perm, degrees, radians. "
    "Constants: pi, e, tau."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": _CALCULATOR_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The expression to evaluate, e.g. 'sqrt(64 + 225)' or 'binom(10, 3) * pi'",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


# Whitelisted names exposed inside _safe_eval. Anything not here raises an error.
_SAFE_NAMES: dict[str, object] = {
    # constants
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
    # builtins
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "pow": pow,
    # math
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "floor": math.floor,
    "ceil": math.ceil,
    "trunc": math.trunc,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "comb": math.comb,
    "binom": math.comb,
    "perm": math.perm,
    "degrees": math.degrees,
    "radians": math.radians,
}

_ALLOWED_NODES: tuple[type, ...] = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.List,
    ast.Tuple,
    # binary operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    # unary operators
    ast.USub,
    ast.UAdd,
)


def _safe_eval(expression: str) -> str:
    """Safely evaluate a mathematical expression.

    Parses the expression into an AST and rejects anything outside a fixed
    whitelist of node types and names (see ``_SAFE_NAMES``). Attribute access,
    imports, comparisons, comprehensions, and statements are not allowed.
    """
    if len(expression) > 200:
        return "Error: expression too long"
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return f"Error: {e.msg}"
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return f"Error: disallowed syntax ({type(node).__name__})"
        if isinstance(node, ast.Name) and node.id not in _SAFE_NAMES:
            return f"Error: unknown name '{node.id}'"
    try:
        result = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, _SAFE_NAMES)
    except Exception as e:
        return f"Error: {e}"
    if isinstance(result, bool):
        return str(result)
    if isinstance(result, int):
        return str(result)
    if isinstance(result, float):
        if result == int(result):
            return str(int(result))
        return str(round(result, 6))
    return str(result)


def _extract_boxed(text: str) -> str | None:
    """Extract the contents of the last ``\\boxed{...}`` with balanced braces."""
    idx = text.rfind(r"\boxed{")
    if idx < 0:
        return None
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def _extract_answer(text: str) -> str:
    """Try multiple patterns to extract the final answer.

    Order: \\boxed{} → <answer> tags → #### → "answer is X" → last number.
    """

    # 1. \boxed{ANSWER}  (balanced braces — handles nested LaTeX like \frac{a}{b})
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed

    # 2. <answer>ANSWER</answer>
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # 3. #### ANSWER
    m = re.search(r"####\s*(.+?)(?:\n|$)", text)
    if m:
        return m.group(1).strip()

    # 4. "the answer is NUMBER" / "the final answer is NUMBER"
    m = re.search(r"(?:the\s+)?(?:final\s+)?answer\s+is[:\s]*([+-]?\d[\d,]*(?:\.\d+)?)", text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")

    # 5. Last standalone number in the text (aggressive fallback)
    numbers = re.findall(r"(?<!\w)([+-]?\d[\d,]*\.?\d*)(?!\w)", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def _msg_to_dict(msg) -> dict:
    """Convert an OpenAI message object to a plain dict for serialization."""
    if isinstance(msg, dict):
        return msg
    d: dict = {"role": msg.role}
    if msg.content:
        d["content"] = msg.content
    if getattr(msg, "tool_calls", None):
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    if getattr(msg, "tool_call_id", None):
        d["tool_call_id"] = msg.tool_call_id
    return d


@rllm.rollout(name="math-tool-agent")
async def math_tool_agent(task: Task, config: AgentConfig) -> Episode:
    """Multi-turn agent that solves math problems using a calculator tool."""
    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY")
    question = task.instruction

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    steps = []
    final_answer = ""
    used_tool = False

    for turn in range(MAX_TURNS):
        try:
            tool_choice = "auto" if used_tool else "required"
            response = await client.chat.completions.create(
                model=config.model,
                messages=messages,
                tools=TOOLS,
                tool_choice=tool_choice,
                timeout=120,
            )
        except Exception as e:
            # #region debug-point A:llm-error
            _debug_event(
                "A",
                "math_tool_agent.py:llm_call",
                "llm call failed",
                {"turn": turn, "used_tool": used_tool, "error": str(e), "model": config.model},
            )
            # #endregion
            logger.warning("Task %s turn %d: LLM call failed: %s", question[:40], turn, e)
            break

        msg = response.choices[0].message
        content = msg.content or ""
        tool_calls = msg.tool_calls or []
        # #region debug-point A:llm-response
        _debug_event(
            "A",
            "math_tool_agent.py:llm_response",
            "llm response received",
            {
                "turn": turn,
                "tool_choice": tool_choice,
                "used_tool_before": used_tool,
                "content_chars": len(content),
                "tool_calls_count": len(tool_calls),
                "finish_reason": getattr(response.choices[0], "finish_reason", None),
                "model": config.model,
            },
        )
        # #endregion

        # Append the assistant message (may include tool_calls)
        assistant_msg = _msg_to_dict(msg)
        messages.append(assistant_msg)

        # Record this LLM call as a step
        steps.append(
            Step(
                chat_completions=[_msg_to_dict(m) if not isinstance(m, dict) else m for m in messages],
                model_response=content,
                action=content,
            )
        )

        # Execute any tool calls
        if tool_calls:
            used_tool = True
            for tc in tool_calls:
                args = json.loads(tc.function.arguments)
                expr = args.get("expression", "")
                result = _safe_eval(expr)
                # #region debug-point D:tool-result
                _debug_event(
                    "D",
                    "math_tool_agent.py:tool_execute",
                    "calculator executed",
                    {
                        "turn": turn,
                        "tool_name": tc.function.name,
                        "expression": expr[:500],
                        "result": result[:500],
                        "is_error": result.startswith("Error:"),
                    },
                )
                # #endregion
                logger.info("Task %s turn %d: tool(%s) = %s", question[:40], turn, expr, result)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            # No tool calls — model produced a text response; extract answer and stop.
            final_answer = _extract_answer(content)
            # #region debug-point D:answer-extract
            _debug_event(
                "D",
                "math_tool_agent.py:answer_extract",
                "final answer extracted from text response",
                {"turn": turn, "answer": final_answer[:200], "content_tail": content[-300:]},
            )
            # #endregion
            break

    # If loop ended without extracting (e.g. hit MAX_TURNS), try last content
    if not final_answer and steps:
        last_content = steps[-1].action or ""
        final_answer = _extract_answer(str(last_content))
        # #region debug-point C:loop-end-extract
        _debug_event(
            "C",
            "math_tool_agent.py:loop_end_extract",
            "final answer extracted after loop end",
            {"steps": len(steps), "used_tool": used_tool, "answer": final_answer[:200]},
        )
        # #endregion

    # #region debug-point C:episode-summary
    _debug_event(
        "C",
        "math_tool_agent.py:episode_return",
        "episode returning",
        {"steps": len(steps), "used_tool": used_tool, "final_answer": final_answer[:200]},
    )
    # #endregion

    return Episode(
        trajectories=[Trajectory(name="solver", steps=steps)],
        artifacts={"answer": final_answer, "used_tool": used_tool},
    )
