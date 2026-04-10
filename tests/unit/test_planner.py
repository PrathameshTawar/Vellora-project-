"""
Unit tests for PlannerAgent.

Tests:
1.  Valid query + mocked LLM returning 3 valid subtasks → PlannerOutput with 3 subtasks
2.  Valid query + mocked LLM returning 5 valid subtasks → PlannerOutput with 5 subtasks
3.  Empty query → dict with error="INVALID_QUERY"
4.  Whitespace-only query → dict with error="INVALID_QUERY"
5.  Query of exactly 4 chars → dict with error="INVALID_QUERY"
6.  Query of exactly 5 chars → does NOT return INVALID_QUERY (proceeds to LLM)
7.  LLM returns non-JSON → AgentError with error_code="INVALID_JSON"
8.  LLM returns JSON array with 2 subtasks (too few) → AgentError with error_code="INVALID_COUNT"
9.  LLM returns JSON array with 6 subtasks (too many) → AgentError with error_code="INVALID_COUNT"
10. LLM returns subtask with invalid type → AgentError with error_code="SCHEMA_VALIDATION_ERROR"
11. LLM call times out → AgentError with error_code="TIMEOUT"
12. LLM returns subtask without subtask_id → auto-generates UUID (result is PlannerOutput)
13. Each subtask in valid output has all required fields

Requirements: 1.1, 1.2, 1.3, 1.4
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
import uuid
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Stub out autogen before importing any swarmiq modules
# ---------------------------------------------------------------------------
mod = types.ModuleType("autogen")
mod.AssistantAgent = MagicMock  # type: ignore[attr-defined]
sys.modules["autogen"] = mod

# ---------------------------------------------------------------------------
# Now safe to import swarmiq
# ---------------------------------------------------------------------------
from swarmiq.agents.planner import PlannerAgent  # noqa: E402
from swarmiq.core.models import AgentError, PlannerOutput, SubTask  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_TYPES = [
    "literature_review",
    "summarization",
    "visualization",
    "conflict_resolution",
    "synthesis",
]


def _make_subtask_dict(i: int = 0, include_id: bool = True) -> dict:
    """Return a valid subtask dict."""
    d: dict = {
        "type": _VALID_TYPES[i % len(_VALID_TYPES)],
        "description": f"Subtask description {i}",
        "search_keywords": [f"keyword{i}", "research"],
    }
    if include_id:
        d["subtask_id"] = str(uuid.uuid4())
    return d


def _make_llm_response(n: int, include_id: bool = True) -> str:
    """Return a JSON string with *n* valid subtask dicts."""
    return json.dumps([_make_subtask_dict(i, include_id=include_id) for i in range(n)])


def _make_agent() -> PlannerAgent:
    """Return a PlannerAgent with a dummy llm_config."""
    return PlannerAgent(llm_config={"model": "gpt-4o", "api_key": "test"})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_valid_query_3_subtasks():
    """Valid query with mocked LLM returning 3 subtasks → PlannerOutput with 3 subtasks."""
    agent = _make_agent()
    agent._call_llm = AsyncMock(return_value=PlannerOutput(
        subtasks=[
            SubTask(
                subtask_id=str(uuid.uuid4()),
                type=_VALID_TYPES[i % len(_VALID_TYPES)],
                description=f"Subtask {i}",
                search_keywords=["kw"],
            )
            for i in range(3)
        ]
    ))

    result = asyncio.run(agent.decompose("What is quantum computing?", "research"))

    assert isinstance(result, PlannerOutput)
    assert len(result.subtasks) == 3


def test_valid_query_5_subtasks():
    """Valid query with mocked LLM returning 5 subtasks → PlannerOutput with 5 subtasks."""
    agent = _make_agent()
    agent._call_llm = AsyncMock(return_value=PlannerOutput(
        subtasks=[
            SubTask(
                subtask_id=str(uuid.uuid4()),
                type=_VALID_TYPES[i % len(_VALID_TYPES)],
                description=f"Subtask {i}",
                search_keywords=["kw"],
            )
            for i in range(5)
        ]
    ))

    result = asyncio.run(agent.decompose("Explain the impact of AI on healthcare", "research"))

    assert isinstance(result, PlannerOutput)
    assert len(result.subtasks) == 5


def test_empty_query_returns_invalid_query_error():
    """Empty query → dict with error='INVALID_QUERY'."""
    agent = _make_agent()

    result = asyncio.run(agent.decompose("", "research"))

    assert isinstance(result, dict)
    assert result["error"] == "INVALID_QUERY"


def test_whitespace_only_query_returns_invalid_query_error():
    """Whitespace-only query → dict with error='INVALID_QUERY'."""
    agent = _make_agent()

    result = asyncio.run(agent.decompose("     ", "research"))

    assert isinstance(result, dict)
    assert result["error"] == "INVALID_QUERY"


def test_query_exactly_4_chars_returns_invalid_query_error():
    """Query of exactly 4 non-whitespace chars → dict with error='INVALID_QUERY'."""
    agent = _make_agent()

    result = asyncio.run(agent.decompose("abcd", "research"))

    assert isinstance(result, dict)
    assert result["error"] == "INVALID_QUERY"


def test_query_exactly_5_chars_proceeds_to_llm():
    """Query of exactly 5 chars → does NOT return INVALID_QUERY (proceeds to LLM)."""
    agent = _make_agent()
    agent._call_llm = AsyncMock(return_value=PlannerOutput(
        subtasks=[
            SubTask(
                subtask_id=str(uuid.uuid4()),
                type="literature_review",
                description="Subtask",
                search_keywords=["kw"],
            )
            for _ in range(3)
        ]
    ))

    result = asyncio.run(agent.decompose("abcde", "research"))

    # Must NOT be an INVALID_QUERY error
    if isinstance(result, dict):
        assert result.get("error") != "INVALID_QUERY"
    # _call_llm should have been called
    agent._call_llm.assert_called_once()


def test_llm_returns_non_json_returns_invalid_json_error():
    """LLM returns non-JSON text → AgentError with error_code='INVALID_JSON'."""
    agent = _make_agent()

    async def _side_effect(q, d):
        return agent._parse_and_validate("not valid json at all")

    agent._call_llm = AsyncMock(side_effect=_side_effect)

    result = asyncio.run(agent.decompose("What is machine learning?", "research"))

    assert isinstance(result, AgentError)
    assert result.error_code == "INVALID_JSON"


def test_llm_returns_2_subtasks_returns_invalid_count_error():
    """LLM returns JSON array with 2 subtasks (too few) → AgentError with error_code='INVALID_COUNT'."""
    agent = _make_agent()

    async def _side_effect(q, d):
        return agent._parse_and_validate(_make_llm_response(2))

    agent._call_llm = AsyncMock(side_effect=_side_effect)

    result = asyncio.run(agent.decompose("What is machine learning?", "research"))

    assert isinstance(result, AgentError)
    assert result.error_code == "INVALID_COUNT"


def test_llm_returns_6_subtasks_returns_invalid_count_error():
    """LLM returns JSON array with 6 subtasks (too many) → AgentError with error_code='INVALID_COUNT'."""
    agent = _make_agent()

    async def _side_effect(q, d):
        return agent._parse_and_validate(_make_llm_response(6))

    agent._call_llm = AsyncMock(side_effect=_side_effect)

    result = asyncio.run(agent.decompose("What is machine learning?", "research"))

    assert isinstance(result, AgentError)
    assert result.error_code == "INVALID_COUNT"


def test_llm_returns_invalid_type_returns_schema_validation_error():
    """LLM returns subtask with invalid type → AgentError with error_code='SCHEMA_VALIDATION_ERROR'."""
    bad_subtask = {
        "subtask_id": str(uuid.uuid4()),
        "type": "invalid_type_value",  # not in the allowed enum
        "description": "Some description",
        "search_keywords": ["kw"],
    }
    raw = json.dumps([bad_subtask, _make_subtask_dict(1), _make_subtask_dict(2)])

    agent = _make_agent()

    async def _side_effect(q, d):
        return agent._parse_and_validate(raw)

    agent._call_llm = AsyncMock(side_effect=_side_effect)

    result = asyncio.run(agent.decompose("What is machine learning?", "research"))

    assert isinstance(result, AgentError)
    assert result.error_code == "SCHEMA_VALIDATION_ERROR"


def test_llm_timeout_returns_timeout_error():
    """LLM call times out → AgentError with error_code='TIMEOUT'."""
    agent = _make_agent()
    agent._call_llm = AsyncMock(side_effect=asyncio.TimeoutError())

    result = asyncio.run(agent.decompose("What is machine learning?", "research"))

    assert isinstance(result, AgentError)
    assert result.error_code == "TIMEOUT"


def test_llm_returns_subtask_without_id_auto_generates_uuid():
    """LLM returns subtask without subtask_id → auto-generates UUID; result is PlannerOutput."""
    raw = _make_llm_response(3, include_id=False)

    agent = _make_agent()

    async def _side_effect(q, d):
        return agent._parse_and_validate(raw)

    agent._call_llm = AsyncMock(side_effect=_side_effect)

    result = asyncio.run(agent.decompose("What is machine learning?", "research"))

    assert isinstance(result, PlannerOutput)
    assert len(result.subtasks) == 3
    for subtask in result.subtasks:
        # Each subtask_id should be a valid UUID
        uuid.UUID(subtask.subtask_id)  # raises ValueError if not valid


def test_valid_output_subtasks_have_all_required_fields():
    """Each subtask in valid output has all required fields."""
    raw = _make_llm_response(3)

    agent = _make_agent()

    async def _side_effect(q, d):
        return agent._parse_and_validate(raw)

    agent._call_llm = AsyncMock(side_effect=_side_effect)

    result = asyncio.run(agent.decompose("What is machine learning?", "research"))

    assert isinstance(result, PlannerOutput)
    for subtask in result.subtasks:
        assert isinstance(subtask, SubTask)
        assert subtask.subtask_id, "subtask_id must be non-empty"
        assert subtask.type, "type must be non-empty"
        assert subtask.description, "description must be non-empty"
        assert isinstance(subtask.search_keywords, list)
        assert len(subtask.search_keywords) > 0, "search_keywords must be non-empty"
