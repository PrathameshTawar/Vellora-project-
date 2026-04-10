"""
Property-based tests for PlannerAgent.

Feature: swarmiq-v2
Validates: Requirements 1.1, 1.2, 1.3
"""
from __future__ import annotations

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Stub out autogen before importing PlannerAgent
# ---------------------------------------------------------------------------
mod = types.ModuleType("autogen")
mod.AssistantAgent = MagicMock
sys.modules["autogen"] = mod

from swarmiq.agents.planner import PlannerAgent  # noqa: E402
from swarmiq.core.models import PlannerOutput, SubTask  # noqa: E402

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_DUMMY_LLM_CONFIG = {"config_list": [{"model": "gpt-4o", "api_key": "test"}]}

_VALID_SUBTASKS = [
    SubTask(
        subtask_id="00000000-0000-0000-0000-000000000001",
        type="literature_review",
        description="Review existing literature on the topic.",
        search_keywords=["topic", "review"],
    ),
    SubTask(
        subtask_id="00000000-0000-0000-0000-000000000002",
        type="summarization",
        description="Summarize key findings from retrieved documents.",
        search_keywords=["summary", "findings"],
    ),
    SubTask(
        subtask_id="00000000-0000-0000-0000-000000000003",
        type="synthesis",
        description="Synthesize results into a coherent report.",
        search_keywords=["synthesis", "report"],
    ),
]

_VALID_PLANNER_OUTPUT = PlannerOutput(subtasks=_VALID_SUBTASKS)

_ALLOWED_TYPES = {
    "literature_review",
    "summarization",
    "visualization",
    "conflict_resolution",
    "synthesis",
}

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid queries: at least 5 printable non-whitespace characters (so strip() >= 5)
valid_query = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P")),
    min_size=5,
)
short_query = st.text(max_size=4)
whitespace_query = st.just("") | st.text(
    alphabet=" \t\n", min_size=1, max_size=4
)

# ---------------------------------------------------------------------------
# Property 1: Planner output count
# ---------------------------------------------------------------------------


class TestPlannerOutputCount:
    """
    # Feature: swarmiq-v2, Property 1: Planner output count —
    # for any valid query (length >= 5), the PlannerAgent output must be a
    # PlannerOutput containing between 3 and 5 subtasks.
    """

    @given(valid_query)
    @settings(max_examples=100)
    def test_planner_output_count(self, query: str):
        # Feature: swarmiq-v2, Property 1: Planner output count
        agent = PlannerAgent(llm_config=_DUMMY_LLM_CONFIG)
        agent._call_llm = AsyncMock(return_value=_VALID_PLANNER_OUTPUT)

        result = asyncio.run(agent.decompose(query, "research"))

        assert isinstance(result, PlannerOutput)
        assert 3 <= len(result.subtasks) <= 5


# ---------------------------------------------------------------------------
# Property 2: Planner subtask type labels
# ---------------------------------------------------------------------------


class TestPlannerSubtaskTypeLabels:
    """
    # Feature: swarmiq-v2, Property 2: Planner subtask type labels —
    # for any valid query, every subtask must have a `type` field from the
    # allowed enum.
    """

    @given(valid_query)
    @settings(max_examples=100)
    def test_planner_subtask_type_labels(self, query: str):
        # Feature: swarmiq-v2, Property 2: Planner subtask type labels
        agent = PlannerAgent(llm_config=_DUMMY_LLM_CONFIG)
        agent._call_llm = AsyncMock(return_value=_VALID_PLANNER_OUTPUT)

        result = asyncio.run(agent.decompose(query, "research"))

        assert isinstance(result, PlannerOutput)
        for subtask in result.subtasks:
            assert subtask.type in _ALLOWED_TYPES


# ---------------------------------------------------------------------------
# Property 3: Planner rejects short or empty queries
# ---------------------------------------------------------------------------


class TestPlannerRejectsShortQueries:
    """
    # Feature: swarmiq-v2, Property 3: Planner rejects short or empty queries —
    # for any query of length 0–4 (including empty and whitespace-only), the
    # PlannerAgent must return a dict with "error": "INVALID_QUERY".
    """

    @given(short_query)
    @settings(max_examples=100)
    def test_planner_rejects_short_query(self, query: str):
        # Feature: swarmiq-v2, Property 3: Planner rejects short or empty queries
        agent = PlannerAgent(llm_config=_DUMMY_LLM_CONFIG)

        result = asyncio.run(agent.decompose(query, "research"))

        assert isinstance(result, dict)
        assert result.get("error") == "INVALID_QUERY"

    @given(whitespace_query)
    @settings(max_examples=100)
    def test_planner_rejects_whitespace_only_query(self, query: str):
        # Feature: swarmiq-v2, Property 3: Planner rejects short or empty queries
        agent = PlannerAgent(llm_config=_DUMMY_LLM_CONFIG)

        result = asyncio.run(agent.decompose(query, "research"))

        assert isinstance(result, dict)
        assert result.get("error") == "INVALID_QUERY"
