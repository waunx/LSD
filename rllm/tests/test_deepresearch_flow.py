from cookbooks.deepresearch.deepresearch_flow import (
    SEARCH_SYSTEM_PROMPT,
    _assistant_answer_text,
    _recover_reasoning_tool_calls,
)
from examples.fully_async.deepresearch.search_agent import (
    SEARCH_SYSTEM_PROMPT as FULLY_ASYNC_SEARCH_SYSTEM_PROMPT,
)


def test_search_system_prompt_matches_fully_async():
    assert SEARCH_SYSTEM_PROMPT == FULLY_ASYNC_SEARCH_SYSTEM_PROMPT
    assert "\b" not in SEARCH_SYSTEM_PROMPT


def test_assistant_answer_text_prefers_content():
    message = {
        "content": "final answer",
        "reasoning_content": "internal reasoning",
    }

    assert _assistant_answer_text(message) == "final answer"


def test_assistant_answer_text_falls_back_to_reasoning_content():
    message = {
        "content": None,
        "reasoning_content": r"The answer is \\boxed{Paris}.",
    }

    assert _assistant_answer_text(message) == r"The answer is \\boxed{Paris}."


def test_assistant_answer_text_handles_empty_message():
    assert _assistant_answer_text({}) == ""


def test_recover_reasoning_tool_calls_matches_fully_async_parser():
    message = {
        "content": None,
        "reasoning_content": ('I will search now.\n<tool_call>{"name":"local_search","arguments":{"query":"Qwen tool parser"}}</tool_call>'),
        "tool_calls": None,
    }

    normalized = _recover_reasoning_tool_calls(message)

    assert normalized["content"] == "I will search now."
    assert normalized["reasoning_content"] is None
    assert normalized["tool_calls"] == [
        {
            "id": "call_0",
            "type": "function",
            "function": {
                "name": "local_search",
                "arguments": '{"query": "Qwen tool parser"}',
            },
        }
    ]


def test_recover_reasoning_tool_calls_preserves_structured_response():
    message = {
        "content": None,
        "reasoning_content": "reasoning",
        "tool_calls": [{"id": "server_call"}],
    }

    assert _recover_reasoning_tool_calls(message) is message
    assert message["reasoning_content"] == "reasoning"


def test_recover_reasoning_tool_calls_ignores_unwrapped_json():
    message = {
        "content": None,
        "reasoning_content": ('```json\n{"name":"local_search","arguments":{"query":"x"}}\n```'),
        "tool_calls": None,
    }

    normalized = _recover_reasoning_tool_calls(message)

    assert normalized["tool_calls"] is None
    assert normalized["reasoning_content"].startswith("```json")


def test_recover_reasoning_tool_calls_marks_missing_name_as_parse_error():
    message = {
        "content": None,
        "reasoning_content": ('<tool_call>{"arguments":{"query":"x"}}</tool_call>'),
        "tool_calls": None,
    }

    normalized = _recover_reasoning_tool_calls(message)

    assert normalized["tool_calls"] is None
    assert normalized["_tool_parse_error"] is True


def test_recover_reasoning_tool_calls_marks_invalid_json_as_parse_error():
    message = {
        "content": None,
        "reasoning_content": ('<tool_call>{"name":"local_search","arguments":invalid}</tool_call>'),
        "tool_calls": None,
    }

    normalized = _recover_reasoning_tool_calls(message)

    assert normalized["tool_calls"] is None
    assert normalized["_tool_parse_error"] is True
