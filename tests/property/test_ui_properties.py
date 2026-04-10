"""
Property-based tests for Orchestrator activity feed and UI data structures.

Feature: swarmiq-v2
Validates: Requirements 11.1, 11.3, 12.1, 12.2, 12.4
"""
from __future__ import annotations

import asyncio
import sys
import types
import uuid
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Stub out heavy dependencies before any swarmiq imports
# ---------------------------------------------------------------------------

def _stub(name: str) -> types.ModuleType:
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
    "gradio",
    "gradio.themes",
]:
    if _dep not in sys.modules:
        _stub(_dep)

# Provide required stubs
sys.modules["autogen"].AssistantAgent = MagicMock  # type: ignore[attr-defined]
sys.modules["sentence_transformers"].SentenceTransformer = MagicMock  # type: ignore[attr-defined]
sys.modules["pinecone"].Pinecone = MagicMock  # type: ignore[attr-defined]

# Stub gradio components used at module level in app.py
_gr = sys.modules["gradio"]
for _attr in [
    "Blocks", "Row", "Column", "Textbox", "Radio", "Button", "HTML",
    "Markdown", "Dataframe", "Plot", "File", "State", "themes",
]:
    setattr(_gr, _attr, MagicMock)
_gr.themes.Soft = MagicMock  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------

from swarmiq.agents.literature import LiteratureOutput  # noqa: E402
from swarmiq.agents.visualization import VisualizationOutput  # noqa: E402
from swarmiq.core.models import (  # noqa: E402
    AgentError,
    Claim,
    ConflictResolverOutput,
    Document,
    EvaluatorOutput,
    PlannerOutput,
    Resolution,
    ScoredClaim,
    SubTask,
    SummarizerOutput,
    SynthesizerOutput,
)
from swarmiq.core.orchestrator import PipelineResult, SwarmOrchestrator  # noqa: E402
from swarmiq.ui.app import _evaluator_to_metrics, _resolutions_to_rows  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _run_async(coro):
    return asyncio.run(coro)


def _make_subtask(i: int = 0) -> SubTask:
    return SubTask(
        subtask_id=str(uuid.uuid4()),
        type="literature_review",
        description=f"Subtask {i}",
        search_keywords=["keyword"],
    )


def _make_scored_claim(subtask_id: str, claim_id: str | None = None) -> ScoredClaim:
    return ScoredClaim(
        claim_id=claim_id or str(uuid.uuid4()),
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


def _build_orchestrator(n_subtasks: int = 3):
    """Build a SwarmOrchestrator with all agents mocked."""
    subtasks = [_make_subtask(i) for i in range(n_subtasks)]
    planner_output = PlannerOutput(subtasks=subtasks)

    planner = MagicMock()
    planner.decompose = AsyncMock(return_value=planner_output)

    lit_output = LiteratureOutput(
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

    claims = [
        Claim(
            claim_id=str(uuid.uuid4()),
            claim_text="Some factual claim.",
            confidence=0.8,
            source_url="https://example.com/paper",
            subtask_id=subtask.subtask_id,
        )
        for subtask in subtasks
    ]
    summ_output = SummarizerOutput(claims=claims[:1])
    summarizer = MagicMock()
    summarizer.summarize = AsyncMock(return_value=summ_output)

    scored = [_make_scored_claim(st.subtask_id) for st in subtasks]
    resolutions = [_make_resolution(sc.claim_id) for sc in scored]
    cr_output = ConflictResolverOutput(resolutions=resolutions)
    conflict_resolver = MagicMock()
    conflict_resolver.resolve = MagicMock(return_value=cr_output)

    synth_output = SynthesizerOutput(
        report_markdown="# Report\n\nContent [1].",
        references=[],
    )
    synthesizer = MagicMock()
    synthesizer.synthesize = AsyncMock(return_value=synth_output)

    eval_output = EvaluatorOutput(
        coherence=0.95,
        factuality=0.95,
        citation_coverage=0.95,
        composite_score=0.95,
        passed=True,
        deficiencies=[],
    )
    evaluator = MagicMock()
    evaluator.evaluate = AsyncMock(return_value=eval_output)

    viz_output = VisualizationOutput(figures=[])
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
# Property 24: Activity feed contains entry for every dispatched event
# ---------------------------------------------------------------------------

_AGENT_TYPES = [
    "planner_agent",
    "literature_agent",
    "summarizer_agent",
    "credibility_scorer",
    "conflict_resolver_agent",
    "synthesizer_agent",
    "evaluator_agent",
    "visualization_agent",
]


class TestActivityFeedContainsEntryForEveryDispatchedEvent:
    """
    # Feature: swarmiq-v2, Property 24: Activity feed contains entry for every dispatched event
    """

    @given(n_subtasks=st.integers(min_value=3, max_value=5))
    @settings(max_examples=100, deadline=None)
    def test_activity_feed_has_entry_per_agent_type(self, n_subtasks: int):
        # Feature: swarmiq-v2, Property 24: Activity feed contains entry for every dispatched event
        orchestrator, _ = _build_orchestrator(n_subtasks=n_subtasks)

        result = _run_async(
            orchestrator.run_pipeline(
                query="Research query for activity feed test",
                domain_mode="research",
                session_id=str(uuid.uuid4()),
            )
        )

        assert isinstance(result, PipelineResult)
        agent_types_in_feed = {e.agent_type for e in result.activity_events}

        for agent_type in _AGENT_TYPES:
            assert agent_type in agent_types_in_feed, (
                f"Expected agent_type '{agent_type}' in activity feed, "
                f"but only found: {agent_types_in_feed}"
            )

    @given(n_subtasks=st.integers(min_value=3, max_value=5))
    @settings(max_examples=100, deadline=None)
    def test_every_activity_event_has_non_empty_timestamp_and_agent_type(
        self, n_subtasks: int
    ):
        # Feature: swarmiq-v2, Property 24: Activity feed contains entry for every dispatched event
        orchestrator, _ = _build_orchestrator(n_subtasks=n_subtasks)

        result = _run_async(
            orchestrator.run_pipeline(
                query="Timestamp and agent type validation query",
                domain_mode="research",
                session_id=str(uuid.uuid4()),
            )
        )

        assert len(result.activity_events) > 0, "Expected at least one activity event"

        for event in result.activity_events:
            assert event.timestamp, (
                f"Event {event.event_id} has empty timestamp"
            )
            assert event.agent_type, (
                f"Event {event.event_id} has empty agent_type"
            )

    @given(n_subtasks=st.integers(min_value=3, max_value=5))
    @settings(max_examples=100, deadline=None)
    def test_completion_and_failure_events_update_status(self, n_subtasks: int):
        # Feature: swarmiq-v2, Property 24: Activity feed contains entry for every dispatched event
        orchestrator, _ = _build_orchestrator(n_subtasks=n_subtasks)

        result = _run_async(
            orchestrator.run_pipeline(
                query="Status update validation query for orchestrator",
                domain_mode="research",
                session_id=str(uuid.uuid4()),
            )
        )

        statuses = {e.status for e in result.activity_events}
        # Every dispatched agent should also have a completed or failed event
        assert "dispatched" in statuses, "Expected 'dispatched' status in activity events"
        assert "completed" in statuses or "failed" in statuses, (
            "Expected at least one 'completed' or 'failed' status in activity events"
        )

        # For each agent type that was dispatched, verify a terminal event exists
        dispatched_agents = {
            e.agent_type for e in result.activity_events if e.status == "dispatched"
        }
        terminal_agents = {
            e.agent_type
            for e in result.activity_events
            if e.status in ("completed", "failed")
        }
        for agent_type in dispatched_agents:
            assert agent_type in terminal_agents, (
                f"Agent '{agent_type}' was dispatched but has no completed/failed event"
            )


# ---------------------------------------------------------------------------
# Property 25: Explainability panel completeness
# ---------------------------------------------------------------------------

_STATUS_STRATEGY = st.sampled_from(["accepted", "rejected", "uncertain"])


@st.composite
def resolution_with_scored_claim(draw):
    """Generate a matching (Resolution, ScoredClaim) pair sharing the same claim_id."""
    claim_id = draw(st.uuids().map(str))
    subtask_id = draw(st.uuids().map(str))
    status = draw(_STATUS_STRATEGY)
    credibility_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    confidence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))

    resolution = Resolution(
        claim_id=claim_id,
        status=status,
        rationale="Some rationale text.",
        credibility_score=credibility_score,
    )
    scored_claim = ScoredClaim(
        claim_id=claim_id,
        claim_text="Some claim text.",
        confidence=confidence,
        source_url="https://example.com/paper",
        subtask_id=subtask_id,
        credibility_score=credibility_score,
    )
    return resolution, scored_claim


class TestExplainabilityPanelCompleteness:
    """
    # Feature: swarmiq-v2, Property 25: Explainability panel completeness
    """

    @given(
        pairs=st.lists(resolution_with_scored_claim(), min_size=1, max_size=10)
    )
    @settings(max_examples=100)
    def test_one_row_per_claim_with_all_required_fields(
        self, pairs: list[tuple[Resolution, ScoredClaim]]
    ):
        # Feature: swarmiq-v2, Property 25: Explainability panel completeness
        resolutions = [p[0] for p in pairs]
        scored_claims = [p[1] for p in pairs]

        rows = _resolutions_to_rows(resolutions, scored_claims)

        # One row per resolution
        assert len(rows) == len(resolutions), (
            f"Expected {len(resolutions)} rows, got {len(rows)}"
        )

        for i, row in enumerate(rows):
            # Each row must have exactly 5 columns:
            # [claim_id, status, rationale, credibility_score, confidence]
            assert len(row) == 5, (
                f"Row {i} has {len(row)} columns, expected 5: "
                "[claim_id, status, rationale, credibility_score, confidence]"
            )
            claim_id, status, rationale, credibility_score, confidence = row

            assert claim_id, f"Row {i}: claim_id is empty"
            assert status in ("accepted", "rejected", "uncertain"), (
                f"Row {i}: status '{status}' is not valid"
            )
            assert rationale is not None, f"Row {i}: rationale is None"
            assert isinstance(credibility_score, float), (
                f"Row {i}: credibility_score is not a float"
            )
            assert isinstance(confidence, float), (
                f"Row {i}: confidence is not a float"
            )

    @given(
        pairs=st.lists(resolution_with_scored_claim(), min_size=1, max_size=10)
    )
    @settings(max_examples=100)
    def test_claim_id_matches_resolution(
        self, pairs: list[tuple[Resolution, ScoredClaim]]
    ):
        # Feature: swarmiq-v2, Property 25: Explainability panel completeness
        resolutions = [p[0] for p in pairs]
        scored_claims = [p[1] for p in pairs]

        rows = _resolutions_to_rows(resolutions, scored_claims)

        for i, (row, resolution) in enumerate(zip(rows, resolutions)):
            assert row[0] == resolution.claim_id, (
                f"Row {i}: claim_id '{row[0]}' does not match "
                f"resolution.claim_id '{resolution.claim_id}'"
            )

    @given(
        pairs=st.lists(resolution_with_scored_claim(), min_size=1, max_size=10)
    )
    @settings(max_examples=100)
    def test_confidence_comes_from_scored_claim(
        self, pairs: list[tuple[Resolution, ScoredClaim]]
    ):
        # Feature: swarmiq-v2, Property 25: Explainability panel completeness
        resolutions = [p[0] for p in pairs]
        scored_claims = [p[1] for p in pairs]

        rows = _resolutions_to_rows(resolutions, scored_claims)

        confidence_map = {sc.claim_id: sc.confidence for sc in scored_claims}

        for i, (row, resolution) in enumerate(zip(rows, resolutions)):
            expected_confidence = round(confidence_map.get(resolution.claim_id, 0.0), 4)
            assert row[4] == expected_confidence, (
                f"Row {i}: confidence {row[4]} does not match "
                f"expected {expected_confidence} from scored_claim"
            )


# ---------------------------------------------------------------------------
# Property 26: Metrics panel contains all score fields
# ---------------------------------------------------------------------------

@st.composite
def evaluator_output_strategy(draw) -> EvaluatorOutput:
    """Generate a valid EvaluatorOutput with scores in [0.0, 1.0]."""
    coherence = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    factuality = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    citation_coverage = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    composite_score = draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
    passed = composite_score >= 0.90
    return EvaluatorOutput(
        coherence=coherence,
        factuality=factuality,
        citation_coverage=citation_coverage,
        composite_score=composite_score,
        passed=passed,
        deficiencies=[],
    )


class TestMetricsPanelContainsAllScoreFields:
    """
    # Feature: swarmiq-v2, Property 26: Metrics panel contains all score fields
    """

    @given(evaluator_output=evaluator_output_strategy())
    @settings(max_examples=100)
    def test_metrics_panel_contains_all_required_fields(
        self, evaluator_output: EvaluatorOutput
    ):
        # Feature: swarmiq-v2, Property 26: Metrics panel contains all score fields
        rows = _evaluator_to_metrics(evaluator_output)

        metric_names = {row[0] for row in rows}
        required_fields = {"coherence", "factuality", "citation_coverage", "composite_score"}

        for field in required_fields:
            assert field in metric_names, (
                f"Required field '{field}' missing from metrics panel. "
                f"Found: {metric_names}"
            )

    @given(evaluator_output=evaluator_output_strategy())
    @settings(max_examples=100)
    def test_metrics_values_match_evaluator_output(
        self, evaluator_output: EvaluatorOutput
    ):
        # Feature: swarmiq-v2, Property 26: Metrics panel contains all score fields
        rows = _evaluator_to_metrics(evaluator_output)

        metrics_dict = {row[0]: row[1] for row in rows}

        assert metrics_dict["coherence"] == round(evaluator_output.coherence, 4), (
            f"coherence mismatch: {metrics_dict['coherence']} != "
            f"{round(evaluator_output.coherence, 4)}"
        )
        assert metrics_dict["factuality"] == round(evaluator_output.factuality, 4), (
            f"factuality mismatch: {metrics_dict['factuality']} != "
            f"{round(evaluator_output.factuality, 4)}"
        )
        assert metrics_dict["citation_coverage"] == round(
            evaluator_output.citation_coverage, 4
        ), (
            f"citation_coverage mismatch: {metrics_dict['citation_coverage']} != "
            f"{round(evaluator_output.citation_coverage, 4)}"
        )
        assert metrics_dict["composite_score"] == round(
            evaluator_output.composite_score, 4
        ), (
            f"composite_score mismatch: {metrics_dict['composite_score']} != "
            f"{round(evaluator_output.composite_score, 4)}"
        )

    def test_none_evaluator_output_returns_empty_list(self):
        # Feature: swarmiq-v2, Property 26: Metrics panel contains all score fields
        rows = _evaluator_to_metrics(None)
        assert rows == [], f"Expected empty list for None input, got {rows}"
