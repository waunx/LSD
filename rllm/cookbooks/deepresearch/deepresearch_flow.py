"""DeepResearch AgentFlow for regular rLLM/VERL LSD training."""

from __future__ import annotations

import json
import logging
import os
import re
import time

import httpx
from openai import AsyncOpenAI

import rllm
from examples.fully_async.deepresearch.refine_agent import refine
from examples.fully_async.deepresearch.tool import LocalRetrievalTool
from rllm.parser.tool_parser import QwenToolParser
from rllm.types import AgentConfig, Episode, Step, Task, Trajectory

logger = logging.getLogger(__name__)

DEFAULT_TRAIN_RAG_URL = "http://[2605:340:cd51:7700:7739:4273:3005:67c1]:19002"
DEFAULT_VAL_RAG_URL = "http://[2605:340:cd51:7700:7739:4273:3005:67c1]:19003"

SEARCH_SYSTEM_PROMPT = """You are a helpful AI assistant that can search for information to answer questions accurately.

When answering questions:
1. Use the available search tools to find relevant and reliable information
2. Synthesize information from multiple sources when needed
3. Provide accurate and comprehensive answers based on your search results
4. Always put your final answer in \\boxed{} format

For example:
- If the answer is "American", write: \\boxed{American}
- If the answer is "yes", write: \\boxed{yes}
- If the answer is a year like "1985", write: \\boxed{1985}

Remember to search thoroughly and provide your final answer clearly within the \\boxed{} format."""

_TOOL_PARSER = QwenToolParser()


def _split_urls(value: str) -> list[str]:
    return [url.strip().rstrip("/") for url in value.split(",") if url.strip()]


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _rag_urls(is_validation: bool) -> list[str]:
    env_name = "DEEPRESEARCH_VAL_RAG_URLS" if is_validation else "DEEPRESEARCH_TRAIN_RAG_URLS"
    default = DEFAULT_VAL_RAG_URL if is_validation else DEFAULT_TRAIN_RAG_URL
    return _split_urls(os.environ.get(env_name, default))


def _message_to_dict(message) -> dict:
    if isinstance(message, dict):
        return message
    tool_call_dicts = None
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        tool_call_dicts = []
        for tc in tool_calls:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            tool_call_dicts.append(
                {"id": getattr(tc, "id", None), "type": getattr(tc, "type", "function"), "function": {"name": getattr(fn, "name", None), "arguments": getattr(fn, "arguments", None)}}
            )
    return {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", None),
        "reasoning_content": getattr(message, "reasoning_content", None),
        "tool_calls": tool_call_dicts,
    }


def _assistant_answer_text(message: dict) -> str:
    """Return the visible answer, falling back for Base models without </think>."""

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning_content = message.get("reasoning_content")
    return reasoning_content if isinstance(reasoning_content, str) else ""


def _recover_reasoning_tool_calls(message: dict) -> dict:
    """Match Fully Async by parsing tool tags from the complete decoded text."""

    if message.get("tool_calls"):
        return message

    text = "\n".join(
        value
        for value in (
            message.get("reasoning_content"),
            message.get("content"),
        )
        if isinstance(value, str) and value
    )
    has_tool_call_markup = _TOOL_PARSER.tool_call_begin in text
    try:
        parsed = _TOOL_PARSER.parse(text)
    except (KeyError, TypeError) as exc:
        # #region debug-point A:malformed-tool-call
        try:
            debug_url = "http://127.0.0.1:7777/event"
            debug_session = "deepresearch-training-stopped"
            with open(".dbg/deepresearch-training-stopped.env", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("DEBUG_SERVER_URL="):
                        debug_url = line.strip().split("=", 1)[1]
                    elif line.startswith("DEBUG_SESSION_ID="):
                        debug_session = line.strip().split("=", 1)[1]
            httpx.post(
                debug_url,
                json={
                    "sessionId": debug_session,
                    "runId": "post-fix",
                    "hypothesisId": "A",
                    "location": "cookbooks/deepresearch/deepresearch_flow.py:_recover_reasoning_tool_calls",
                    "msg": "[DEBUG] malformed tool call was contained",
                    "data": {
                        "exception_type": type(exc).__name__,
                        "exception": repr(exc),
                        "text_prefix": text[:500],
                    },
                },
                timeout=2.0,
            )
        except Exception:
            pass
        # #endregion
        message["_tool_parse_error"] = True
        return message
    if not parsed:
        if has_tool_call_markup:
            message["_tool_parse_error"] = True
        return message

    message["tool_calls"] = [
        {
            "id": f"call_{index}",
            "type": "function",
            "function": {
                "name": tool_call.name,
                "arguments": json.dumps(
                    tool_call.arguments,
                    ensure_ascii=False,
                )
                if isinstance(tool_call.arguments, dict)
                else tool_call.arguments,
            },
        }
        for index, tool_call in enumerate(parsed)
    ]
    pattern = re.compile(
        re.escape(_TOOL_PARSER.tool_call_begin) + r".*?" + re.escape(_TOOL_PARSER.tool_call_end),
        re.DOTALL,
    )
    message["content"] = pattern.sub("", text).strip()
    message["reasoning_content"] = None
    return message


async def _run_tool_calls(tool: LocalRetrievalTool, tool_calls: list[dict]) -> tuple[list[dict], dict[str, float]]:
    messages = []
    metrics = {
        "total_parse_tool_args_error": 0.0,
        "total_tool_return_error": 0.0,
        "total_refine_error": 0.0,
        "total_tool_calls": 0.0,
        "total_tool_wait_time": 0.0,
        "total_refine_time": 0.0,
        "total_query_length": 0.0,
    }
    for tool_call in tool_calls:
        metrics["total_tool_calls"] += 1.0
        try:
            fn = tool_call.get("function", {})
            if fn.get("name") != tool.name:
                raise ValueError(f"unexpected tool name: {fn.get('name')}")
            args = json.loads(fn.get("arguments") or "{}")
            query = str(args.get("query") or "")
        except Exception:
            metrics["total_parse_tool_args_error"] += 1.0
            query = ""
        metrics["total_query_length"] += len(query)
        start = time.time()
        result = await tool.run(query=query)
        metrics["total_tool_wait_time"] += time.time() - start
        if result.startswith("Error:"):
            metrics["total_tool_return_error"] += 1.0
        elif _env_bool("DEEPRESEARCH_USE_REFINE"):
            raw_result = result
            refine_start = time.time()
            result = await refine(query=query, result=result)
            metrics["total_refine_time"] += time.time() - refine_start
            if result == raw_result:
                metrics["total_refine_error"] += 1.0
        messages.append({"role": "tool", "tool_call_id": tool_call.get("id"), "name": tool.name, "content": result})
    return messages, metrics


@rllm.rollout(name="deepresearch")
async def deepresearch_flow(task: Task, config: AgentConfig) -> Episode:
    meta = task.metadata or {}
    question = str(meta.get("question") or meta.get("prompt") or task.instruction or "")
    max_turns = int(os.environ.get("DEEPRESEARCH_MAX_TURNS", "48"))
    tool = LocalRetrievalTool(server_urls=_rag_urls(config.is_validation), url_file=None, max_results=10, timeout=90.0, format_style="original", max_content_length=4000, max_connections=2000)
    http_client = httpx.AsyncClient(trust_env=False)
    client = AsyncOpenAI(base_url=config.base_url, api_key="EMPTY", http_client=http_client)
    messages: list[dict] = [{"role": "system", "content": SEARCH_SYSTEM_PROMPT}, {"role": "user", "content": question}]
    steps: list[Step] = []
    metrics: dict[str, float] = {
        "total_parse_tool_args_error": 0.0,
        "total_tool_return_error": 0.0,
        "total_refine_error": 0.0,
        "total_tool_calls": 0.0,
        "total_tool_wait_time": 0.0,
        "total_refine_time": 0.0,
        "total_query_length": 0.0,
        "duplicate_search_detected": 0.0,
        "excessive_parallel_calls": 0.0,
        "tool_parse_error_detected": 0.0,
    }
    seen_tool_calls: set[tuple[str, str]] = set()
    final_answer = ""

    for turn in range(max_turns):
        try:
            response = await client.chat.completions.create(model=config.model, messages=messages, tools=[tool.json], timeout=300)
        except Exception as exc:
            logger.warning("deepresearch task %s: LLM call failed at turn %d: %s", task.id, turn + 1, exc)
            metrics["llm_call_error"] = 1.0
            break

        assistant = _recover_reasoning_tool_calls(_message_to_dict(response.choices[0].message))
        tool_parse_error = bool(assistant.pop("_tool_parse_error", False))
        messages.append(assistant)
        final_answer = "" if tool_parse_error else _assistant_answer_text(assistant)
        steps.append(Step(chat_completions=list(messages), model_response=final_answer, action=final_answer, thought=final_answer))

        if tool_parse_error:
            metrics["total_parse_tool_args_error"] += 1.0
            metrics["tool_parse_error_detected"] = 1.0
            break

        tool_calls = assistant.get("tool_calls") or []
        if not tool_calls or turn >= max_turns - 1:
            break
        if len(tool_calls) >= 9:
            metrics["excessive_parallel_calls"] = 1.0
            break
        duplicated = False
        for tool_call in tool_calls:
            fn = tool_call.get("function", {})
            key = (str(fn.get("name")), str(fn.get("arguments")))
            if key in seen_tool_calls:
                duplicated = True
                break
            seen_tool_calls.add(key)
        if duplicated:
            metrics["duplicate_search_detected"] = 1.0
            break

        tool_messages, tool_metrics = await _run_tool_calls(tool, tool_calls)
        messages.extend(tool_messages)
        for key, value in tool_metrics.items():
            metrics[key] = metrics.get(key, 0.0) + float(value)
        if metrics.get("total_tool_return_error", 0.0) > 0:
            break

    await client.close()
    metrics["num_turns"] = float(len(steps))
    return Episode(task=meta, trajectories=[Trajectory(name="deepresearch", steps=steps)], artifacts={"answer": final_answer, "messages": messages}, metrics=metrics)
