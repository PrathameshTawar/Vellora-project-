"""
Property-based tests for Literature_Agent (Properties 4, 5, 6).

Feature: swarmiq-v2
Validates: Requirements 2.1, 2.2, 2.3
"""
from __future__ import annotations

import asyncio
import sys
import types
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

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
]:
    if _dep not in sys.modules:
        _stub(_dep)

sys.modules["autogen"].AssistantAgent = MagicMock  # type: ignore[attr-defined]
sys.modules["sentence_transformers"].SentenceTransformer = MagicMock  # type: ignore[attr-defined]

# CrossEncoder mock: predict() returns a list of 0.5 scores for any input
_ce_instance = MagicMock()
_ce_instance.predict.side_effect = lambda pairs: [0.5] * len(pairs)
sys.modules["sentence_transformers"].CrossEncoder = MagicMock(return_value=_ce_instance)  # type: ignore[attr-defined]

sys.modules["pinecone"].Pinecone = MagicMock  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------

from swarmiq.agents.literature import LiteratureAgent, LiteratureOutput  # noqa: E402
from swarmiq.core.models import Document, SubTask  # noqa: E402
from swarmiq.core.orchestrator import SwarmOrchestrator  # noqa: E402
from swarmiq.core.models import (  # noqa: E402
    ConflictResolverOutput,
    EvaluatorOutput,
    PlannerOutput,
    Resolution,
    ScoredClaim,
    SummarizerOutput,
    SynthesizerOutput,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

subtask_count = st.integers(min_value=3, max_value=5)
session_id_st = st.uuids().map(str)
subtask_id_st = st.uuids().map(str)


def _make_subtask(i: int = 0, subtask_id: str | None = None) -> SubTask:
    return SubTask(
        subtask_id=subtask_id or str(uuid.uuid4()),
        type="literature_review",
        description=f"Subtask {i}",
        search_keywords=["keyword"],
    )


def _make_document(url: str, subtask_id: str) -> Document:
    return Document(
        url=url,
        title="Test Document",
        content="Some content about the topic.",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        subtask_id=subtask_id,
    )


def _make_lit_output(subtask_id: str, n_docs: int = 5) -> LiteratureOutput:
    docs = [
        _make_document(f"https://example.com/doc{i}", subtask_id)
        for i in range(n_docs)
    ]
    return LiteratureOutput(subtask_id=subtask_id, documents=docs)


# ---------------------------------------------------------------------------
# Orchestrator builder (mirrors test_orchestrator.py pattern)
# ---------------------------------------------------------------------------

def _build_orchestrator_with_counter(n_subtasks: int):
    """Build a SwarmOrchestrator with a counting literature_agent_factory."""
    subtasks = [_make_subtask(i) for i in range(n_subtasks)]
    planner_output = PlannerOutput(subtasks=subtasks)

    planner = MagicMock()

    async def _decompose(*args, **kwargs):
        return planner_output

    planner.decompose = _decompose

    call_count = [0]

    def counting_factory():
        call_count[0] += 1
        agent = MagicMock()

        async def _run(subtask, session_id, query_fingerprint):
            return _make_lit_output(subtask.subtask_id)

        agent.run = _run
        return agent

    # Summarizer
    summarizer = MagicMock()

    async def _summarize(subtask_id, documents):
        return SummarizerOutput(claims=[])

    summarizer.summarize = _summarize

    # Conflict resolver
    conflict_resolver = MagicMock()
    conflict_resolver.resolve = MagicMock(
        return_value=ConflictResolverOutput(resolutions=[])
    )

    # Synthesizer
    synthesizer = MagicMock()

    async def _synthesize(*args, **kwargs):
        return SynthesizerOutput(
            report_markdown="# Report\n\nContent.",
            references=[],
        )

    synthesizer.synthesize = _synthesize

    # Evaluator
    evaluator = MagicMock()

    async def _evaluate(*args, **kwargs):
        return EvaluatorOutput(
            coherence=0.95,
            factuality=0.95,
            citation_coverage=0.95,
            composite_score=0.95,
            passed=True,
            deficiencies=[],
        )

    evaluator.evaluate = _evaluate

    # Visualization
    from swarmiq.agents.visualization import VisualizationOutput
    visualization = MagicMock()
    visualization.generate = MagicMock(return_value=VisualizationOutput(figures=[]))

    orchestrator = SwarmOrchestrator(
        planner=planner,
        literature_agent_factory=counting_factory,
        summarizer=summarizer,
        conflict_resolver=conflict_resolver,
        synthesizer=synthesizer,
        evaluator=evaluator,
        visualization=visualization,
    )
    return orchestrator, call_count


# ---------------------------------------------------------------------------
# Property 4: Orchestrator spawns 3–5 Literature_Agent instances
# ---------------------------------------------------------------------------


class TestOrchestratorSpawnsCorrectAgentCount:
    """
    # Feature: swarmiq-v2, Property 4: Orchestrator spawns 3–5 Literature_Agent instances —
    # for any valid subtask list (3–5 subtasks), the Orchestrator must spawn between
    # 3 and 5 LiteratureAgent instances.
    """

    @given(subtask_count)
    @settings(max_examples=100)
    def test_orchestrator_spawns_between_3_and_5_agents(self, n: int):
        # Feature: swarmiq-v2, Property 4: Orchestrator spawns 3–5 Literature_Agent instances
        orchestrator, call_count = _build_orchestrator_with_counter(n)

        asyncio.run(
            orchestrator.run_pipeline(
                query="Research query for property test",
                domain_mode="research",
                session_id=str(uuid.uuid4()),
            )
        )

        assert 3 <= call_count[0] <= 5
        assert call_count[0] == n


# ---------------------------------------------------------------------------
# Property 5: Literature_Agent retrieves at least 5 documents per subtask
# ---------------------------------------------------------------------------


class TestLiteratureAgentRetrievesMinDocuments:
    """
    # Feature: swarmiq-v2, Property 5: Literature_Agent retrieves at least 5 documents per subtask —
    # for any subtask dispatched to a LiteratureAgent (when web search API is available),
    # the returned document list must contain at least 5 documents.
    """

    @given(subtask_id_st, session_id_st)
    @settings(max_examples=100)
    def test_literature_agent_returns_at_least_5_documents(
        self, subtask_id: str, session_id: str
    ):
        # Feature: swarmiq-v2, Property 5: Literature_Agent retrieves at least 5 documents per subtask
        subtask = SubTask(
            subtask_id=subtask_id,
            type="literature_review",
            description="Test subtask",
            search_keywords=["test", "query"],
        )

        mock_store = MagicMock()
        # Cache returns nothing — force web search path
        mock_store.query.return_value = []
        mock_store.upsert = MagicMock()

        agent = LiteratureAgent(
            knowledge_store=mock_store,
            tavily_api_key="fake-key",
        )

        # Mock _search_sync to return 5+ documents
        five_docs = [
            _make_document(f"https://example.com/doc{i}", subtask_id)
            for i in range(5)
        ]

        with patch.object(agent, "_search_sync", return_value=five_docs):
            with patch.object(agent, "_embed_text", return_value=[0.1] * 384):
                output = asyncio.run(
                    agent.run(subtask, session_id, "fingerprint-abc")
                )

        assert isinstance(output, LiteratureOutput)
        assert len(output.documents) >= 5


# ---------------------------------------------------------------------------
# Property 6: Upserted vectors contain required metadata
# ---------------------------------------------------------------------------

_REQUIRED_METADATA_KEYS = {"url", "retrieved_at", "subtask_id", "session_id", "query_fingerprint"}


class TestUpsertedVectorsContainRequiredMetadata:
    """
    # Feature: swarmiq-v2, Property 6: Upserted vectors contain required metadata —
    # for any document retrieved and upserted, the metadata must contain non-null
    # values for url, retrieved_at, subtask_id, session_id, and query_fingerprint.
    """

    @given(subtask_id_st, session_id_st)
    @settings(max_examples=100)
    def test_upserted_vectors_have_required_metadata(
        self, subtask_id: str, session_id: str
    ):
        # Feature: swarmiq-v2, Property 6: Upserted vectors contain required metadata
        subtask = SubTask(
            subtask_id=subtask_id,
            type="literature_review",
            description="Test subtask for metadata",
            search_keywords=["metadata", "test"],
        )

        captured_vectors: list[dict] = []

        mock_store = MagicMock()
        mock_store.query.return_value = []

        def capture_upsert(vectors, namespace=""):
            captured_vectors.extend(vectors)

        mock_store.upsert = capture_upsert

        agent = LiteratureAgent(
            knowledge_store=mock_store,
            tavily_api_key="fake-key",
        )

        docs = [
            _make_document(f"https://example.com/doc{i}", subtask_id)
            for i in range(3)
        ]
        query_fingerprint = "fp-" + subtask_id[:8]

        with patch.object(agent, "_search_sync", return_value=docs):
            with patch.object(agent, "_embed_text", return_value=[0.0] * 384):
                asyncio.run(agent.run(subtask, session_id, query_fingerprint))

        assert len(captured_vectors) > 0, "Expected at least one upserted vector"

        for vec in captured_vectors:
            meta = vec.get("metadata", {})
            for key in _REQUIRED_METADATA_KEYS:
                assert key in meta, f"Missing metadata key: {key}"
                assert meta[key] is not None, f"Metadata key '{key}' is None"
                assert meta[key] != "", f"Metadata key '{key}' is empty string"
