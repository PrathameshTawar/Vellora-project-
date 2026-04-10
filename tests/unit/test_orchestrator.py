"""
Unit tests for SwarmOrchestrator.

Tests:
- Parallel Literature_Agent dispatch count (3–5)
- Schema validation failure halts subtask pipeline
- AgentError routing to activity queue
- Full happy-path pipeline produces PipelineResult with status "complete"
- Planner INVALID_QUERY error → status "failed"
"""
from __future__ import annotations

import asyncio
import dataclasses
import sys
import types
import uuid
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Async test helper — pytest-asyncio may not be installed
# ---------------------------------------------------------------------------

def run_async(coro):
    """Run a coroutine synchronously using asyncio.run()."""
    import asyncio
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Stub out optional heavy dependencies before any swarmiq imports
# ---------------------------------------------------------------------------

def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


for _dep in [
    "autogen",
    "sentence_transformers",
    "pinecone",
    "tavily",
    "serpapi",
    "transformers",
    "plotly",
    "plotly.graph_objects",
    "matplotlib",
    "matplotlib.pyplot",
]:
    if _dep not in sys.modules:
        _stub_module(_dep)

# Provide AssistantAgent stub so planner/summarizer/etc. can be imported
sys.modules["autogen"].AssistantAgent = MagicMock  # type: ignore[attr-defined]

# Provide SentenceTransformer stub
sys.modules["sentence_transformers"].SentenceTransformer = MagicMock  # type: ignore[attr-defined]

# Provide Pinecone stub
sys.modules["pinecone"].Pinecone = MagicMock  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------

from swarmiq.agents.literature import LiteratureOutput
from swarmiq.core.models import (
    ActivityEvent,
    AgentError,
    Claim,
    ConflictResolverOutput,
    Document,
    EvaluatorOutput,
    Figure,
    PlannerOutput,
    Resolution,
    ScoredClaim,
    SubTask,
    SummarizerOutput,
    SynthesizerOutput,
)
from swarmiq.core.orchestrator import PipelineResult, SwarmOrchestrator


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_subtask(i: int = 0) -> SubTask:
    return SubTask(
        subtask_id=str(uuid.uuid4()),
        type="literature_review",
        description=f"Subtask {i}",
        search_keywords=["keyword"],
    )


def _make_claim(subtask_id: str) -> Claim:
    return Claim(
        claim_id=str(uuid.uuid4()),
        claim_text="Some factual claim.",
        confidence=0.8,
        source_url="https://example.com/paper",
        subtask_id=subtask_id,
    )


def _make_scored_claim(subtask_id: str) -> ScoredClaim:
    return ScoredClaim(
        claim_id=str(uuid.uuid4()),
        claim_text="Some factual claim.",
        confidence=0.8,
        source_url="https://example.com/paper",
        subtask_id=subtask_id,
        credibility_score=0.7,
    )


def _make_resolution(claim_id: str) -> Resolution:
    return Resolution(
        claim_id=claim_id,
        status="accepted",
        rationale="No contradictions found.",
        credibility_score=0.7,
    )


def _make_evaluator_output() -> EvaluatorOutput:
    return EvaluatorOutput(
        coherence=0.95,
        factuality=0.95,
        citation_coverage=0.95,
        composite_score=0.95,
        passed=True,
        deficiencies=[],
    )


def _make_planner_output(n: int = 3) -> PlannerOutput:
    return PlannerOutput(subtasks=[_make_subtask(i) for i in range(n)])


def _build_orchestrator(
    planner_return=None,
    lit_return=None,
    summarizer_return=None,
    conflict_return=None,
    synthesizer_return=None,
    evaluator_return=None,
    visualization_return=None,
    n_subtasks: int = 3,
):
    """Build a SwarmOrchestrator with all agents mocked."""
    subtasks = [_make_subtask(i) for i in range(n_subtasks)]
    planner_output = planner_return or PlannerOutput(subtasks=subtasks)

    # Planner mock
    planner = MagicMock()
    planner.decompose = AsyncMock(return_value=planner_output)

    # Literature agent factory
    lit_output = lit_return or LiteratureOutput(
        subtask_id=subtasks[0].subtask_id,
        documents=[
            Document(
                url="https://example.com/doc",
                title="Doc",
                content="Content",
                retrieved_at=datetime.now(timezone.utc).isoformat(),
                subtask_id=subtasks[0].subtask_id,
            )
        ],
    )

    def _factory():
        agent = MagicMock()
        agent.run = AsyncMock(return_value=lit_output)
        return agent

    # Summarizer mock
    claims = [_make_claim(st.subtask_id) for st in subtasks]
    summ_output = summarizer_return or SummarizerOutput(claims=claims[:1])
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value=summ_output)

    # Conflict resolver mock
    scored = [_make_scored_claim(st.subtask_id) for st in subtasks]
    resolutions = [_make_resolution(sc.claim_id) for sc in scored]
    cr_output = conflict_return or ConflictResolverOutput(resolutions=resolutions)
    conflict_resolver = MagicMock()
    conflict_resolver.resolve = MagicMock(return_value=cr_output)

    # Synthesizer mock
    synth_output = synthesizer_return or SynthesizerOutput(
        report_markdown="# Report\n\nContent [1].\n\nhttps://example.com/paper",
        references=[],
    )
    synthesizer = MagicMock()
    synthesizer.synthesize = AsyncMock(return_value=synth_output)

    # Evaluator mock
    eval_output = evaluator_return or _make_evaluator_output()
    evaluator = MagicMock()
    evaluator.evaluate = AsyncMock(return_value=eval_output)

    # Visualization mock
    from swarmiq.agents.visualization import VisualizationOutput
    viz_output = visualization_return or VisualizationOutput(figures=[])
    visualization = MagicMock()
    visualization.generate = MagicMock(return_value=viz_output)

    orchestrator = SwarmOrchestrator(
        planner=planner,
        literature_agent_factory=_factory,
        summarizer=summarizer,
        conflict_resolver=conflict_resolver,
        synthesizer=synthesizer,
        evaluator=evaluator,
        visualization=visualization,
    )
    return orchestrator, subtasks


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_happy_path_returns_complete_status():
    """Full happy-path pipeline should return status 'complete'."""
    async def _run():
        orchestrator, _ = _build_orchestrator(n_subtasks=3)
        return await orchestrator.run_pipeline(
            query="What are the effects of climate change?",
            domain_mode="research",
            session_id=str(uuid.uuid4()),
        )
    result = run_async(_run())
    assert isinstance(result, PipelineResult)
    assert result.status == "complete"
    assert result.report_markdown != ""
    assert result.evaluator_output is not None
    assert result.evaluator_output.passed is True


def test_pipeline_result_contains_session_and_query():
    """PipelineResult should echo back session_id, query, and domain_mode."""
    async def _run():
        orchestrator, _ = _build_orchestrator()
        session_id = str(uuid.uuid4())
        result = await orchestrator.run_pipeline(
            query="Test query for research",
            domain_mode="business",
            session_id=session_id,
        )
        return result, session_id
    result, session_id = run_async(_run())
    assert result.session_id == session_id
    assert result.query == "Test query for research"
    assert result.domain_mode == "business"


def test_literature_agents_dispatched_for_each_subtask():
    """One LiteratureAgent should be dispatched per subtask (up to 5)."""
    n = 4
    orchestrator, subtasks = _build_orchestrator(n_subtasks=n)

    call_count = 0
    original_factory = orchestrator._literature_agent_factory

    def counting_factory():
        nonlocal call_count
        call_count += 1
        return original_factory()

    orchestrator._literature_agent_factory = counting_factory

    async def _run():
        return await orchestrator.run_pipeline(
            query="Parallel dispatch test query",
            domain_mode="research",
            session_id=str(uuid.uuid4()),
        )
    run_async(_run())
    assert call_count == n


def test_literature_agents_capped_at_5():
    """Even if planner returns 5 subtasks, no more than 5 agents are spawned."""
    orchestrator, _ = _build_orchestrator(n_subtasks=5)

    call_count = 0
    original_factory = orchestrator._literature_agent_factory

    def counting_factory():
        nonlocal call_count
        call_count += 1
        return original_factory()

    orchestrator._literature_agent_factory = counting_factory

    async def _run():
        return await orchestrator.run_pipeline(
            query="Cap test query for research",
            domain_mode="research",
            session_id=str(uuid.uuid4()),
        )
    run_async(_run())
    assert call_count <= 5


def test_planner_invalid_query_returns_failed_status():
    """When planner returns INVALID_QUERY, pipeline status should be 'failed'."""
    async def _run():
        orchestrator, _ = _build_orchestrator()
        orchestrator._planner.decompose = AsyncMock(
            return_value={"error": "INVALID_QUERY", "message": "Query too short."}
        )
        return await orchestrator.run_pipeline(
            query="hi",
            domain_mode="research",
            session_id=str(uuid.uuid4()),
        )
    result = run_async(_run())
    assert result.status == "failed"
    assert any(e.error_code == "INVALID_QUERY" for e in result.errors)


def test_planner_agent_error_returns_failed_status():
    """When planner returns AgentError, pipeline status should be 'failed'."""
    async def _run():
        orchestrator, _ = _build_orchestrator()
        orchestrator._planner.decompose = AsyncMock(
            return_value=AgentError(
                agent_type="planner_agent",
                error_code="TIMEOUT",
                message="Timed out.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )
        return await orchestrator.run_pipeline(
            query="Some valid query here",
            domain_mode="research",
            session_id=str(uuid.uuid4()),
        )
    result = run_async(_run())
    assert result.status == "failed"
    assert any(e.error_code == "TIMEOUT" for e in result.errors)


def test_literature_agent_error_is_collected_pipeline_continues():
    """AgentError from a LiteratureAgent should be collected; pipeline continues."""
    async def _run():
        orchestrator, subtasks = _build_orchestrator(n_subtasks=3)

        def error_factory():
            agent = MagicMock()
            agent.run = AsyncMock(
                return_value=AgentError(
                    agent_type="literature_agent",
                    subtask_id=subtasks[0].subtask_id,
                    error_code="TIMEOUT",
                    message="Literature agent timed out.",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
            )
            return agent

        orchestrator._literature_agent_factory = error_factory
        return await orchestrator.run_pipeline(
            query="Error propagation test query",
            domain_mode="research",
            session_id=str(uuid.uuid4()),
        )
    result = run_async(_run())
    assert result is not None
    assert any(e.error_code == "TIMEOUT" for e in result.errors)


def test_activity_events_emitted_for_each_agent():
    """Activity events should be emitted for every agent dispatch and completion."""
    async def _run():
        orchestrator, _ = _build_orchestrator(n_subtasks=3)
        return await orchestrator.run_pipeline(
            query="Activity event tracking test query",
            domain_mode="research",
            session_id=str(uuid.uuid4()),
        )
    result = run_async(_run())
    agent_types = {e.agent_type for e in result.activity_events}
    assert "planner_agent" in agent_types
    assert "literature_agent" in agent_types
    assert "summarizer_agent" in agent_types
    assert "credibility_scorer" in agent_types
    assert "conflict_resolver_agent" in agent_types
    assert "synthesizer_agent" in agent_types
    assert "evaluator_agent" in agent_types
    assert "visualization_agent" in agent_types


def test_activity_events_have_dispatched_and_completed_statuses():
    """Each agent should have both 'dispatched' and 'completed' events."""
    async def _run():
        orchestrator, _ = _build_orchestrator(n_subtasks=3)
        return await orchestrator.run_pipeline(
            query="Status tracking test query here",
            domain_mode="research",
            session_id=str(uuid.uuid4()),
        )
    result = run_async(_run())
    statuses = {e.status for e in result.activity_events}
    assert "dispatched" in statuses
    assert "completed" in statuses


def test_activity_queue_receives_events():
    """The per-session activity queue should contain all emitted events."""
    async def _run():
        orchestrator, _ = _build_orchestrator(n_subtasks=3)
        session_id = str(uuid.uuid4())
        queue = await orchestrator.create_queue(session_id)
        result = await orchestrator.run_pipeline(
            query="Queue population test query here",
            domain_mode="research",
            session_id=session_id,
        )
        queued_events: list[ActivityEvent] = []
        while not queue.empty():
            queued_events.append(queue.get_nowait())
        await orchestrator.remove_queue(session_id)
        return result, queued_events
    result, queued_events = run_async(_run())
    assert len(queued_events) == len(result.activity_events)


def test_schema_validation_error_on_summarizer_halts_subtask():
    """A SchemaValidationError from summarizer output should halt that subtask."""
    async def _run():
        orchestrator, subtasks = _build_orchestrator(n_subtasks=3)
        bad_claim = Claim(
            claim_id=str(uuid.uuid4()),
            claim_text="Bad claim",
            confidence=1.5,  # invalid — exceeds maximum 1.0
            source_url="https://example.com/bad",
            subtask_id=subtasks[0].subtask_id,
        )
        orchestrator._summarizer.summarize = AsyncMock(
            return_value=SummarizerOutput(claims=[bad_claim])
        )
        return await orchestrator.run_pipeline(
            query="Schema validation failure test query",
            domain_mode="research",
            session_id=str(uuid.uuid4()),
        )
    result = run_async(_run())
    assert any(e.error_code == "SCHEMA_VALIDATION_ERROR" for e in result.errors)
    assert result is not None


def test_evaluator_agent_error_does_not_crash_pipeline():
    """AgentError from EvaluatorAgent should be recorded; pipeline returns partial."""
    async def _run():
        orchestrator, _ = _build_orchestrator()
        orchestrator._evaluator.evaluate = AsyncMock(
            return_value=AgentError(
                agent_type="evaluator_agent",
                error_code="UNHANDLED_EXCEPTION",
                message="Evaluator crashed.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )
        return await orchestrator.run_pipeline(
            query="Evaluator error test query here",
            domain_mode="research",
            session_id=str(uuid.uuid4()),
        )
    result = run_async(_run())
    assert result is not None
    assert result.evaluator_output is None
    assert any(e.agent_type == "evaluator_agent" for e in result.errors)


def test_synthesizer_agent_error_results_in_empty_report():
    """AgentError from SynthesizerAgent should leave report_markdown empty."""
    async def _run():
        orchestrator, _ = _build_orchestrator()
        orchestrator._synthesizer.synthesize = AsyncMock(
            return_value=AgentError(
                agent_type="synthesizer_agent",
                error_code="UNHANDLED_EXCEPTION",
                message="Synthesizer crashed.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )
        return await orchestrator.run_pipeline(
            query="Synthesizer error test query here",
            domain_mode="research",
            session_id=str(uuid.uuid4()),
        )
    result = run_async(_run())
    assert result.report_markdown == ""
    assert any(e.agent_type == "synthesizer_agent" for e in result.errors)


def test_pipeline_result_subtasks_match_planner_output():
    """result.subtasks should match what the planner returned."""
    async def _run():
        orchestrator, subtasks = _build_orchestrator(n_subtasks=3)
        result = await orchestrator.run_pipeline(
            query="Subtask tracking test query here",
            domain_mode="research",
            session_id=str(uuid.uuid4()),
        )
        return result, subtasks
    result, subtasks = run_async(_run())
    assert len(result.subtasks) == 3
    assert result.subtasks[0].subtask_id == subtasks[0].subtask_id


def test_activity_events_have_timestamps():
    """Every ActivityEvent should have a non-empty timestamp."""
    async def _run():
        orchestrator, _ = _build_orchestrator(n_subtasks=3)
        return await orchestrator.run_pipeline(
            query="Timestamp validation test query here",
            domain_mode="research",
            session_id=str(uuid.uuid4()),
        )
    result = run_async(_run())
    for event in result.activity_events:
        assert event.timestamp, f"Event {event.event_id} has empty timestamp"
        assert event.agent_type, f"Event {event.event_id} has empty agent_type"
