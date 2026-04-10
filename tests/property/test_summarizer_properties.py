"""
Property-based tests for Summarizer_Agent (Property 7).

Feature: swarmiq-v2
Validates: Requirements 3.2, 3.3, 3.4
"""
from __future__ import annotations

import asyncio
import sys
import types
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock

from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Stub out autogen before any swarmiq imports
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

# ---------------------------------------------------------------------------

from swarmiq.agents.summarizer import SummarizerAgent  # noqa: E402
from swarmiq.core.models import Claim, Document, SummarizerOutput  # noqa: E402

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_DUMMY_LLM_CONFIG = {"config_list": [{"model": "gpt-4o", "api_key": "test"}]}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


url_st = st.from_regex(r"https://example\.com/[a-z]{3,10}", fullmatch=True)

document_st = st.builds(
    Document,
    url=url_st,
    title=st.text(min_size=1, max_size=50, alphabet=st.characters(whitelist_categories=("L", "N", "Zs"))),
    content=st.text(min_size=1, max_size=200, alphabet=st.characters(whitelist_categories=("L", "N", "Zs", "P"))),
    retrieved_at=st.just(_now_iso()),
    subtask_id=st.uuids().map(str),
)

document_list_st = st.lists(document_st, min_size=1, max_size=5)


# ---------------------------------------------------------------------------
# Property 7: Claim schema invariant
# ---------------------------------------------------------------------------


class TestClaimSchemaInvariant:
    """
    # Feature: swarmiq-v2, Property 7: Claim schema invariant —
    # for any non-empty document list processed by the SummarizerAgent, every
    # claim in the output must contain non-empty claim_id, claim_text,
    # source_url, and subtask_id fields, and the confidence field must be a
    # number in the closed interval [0.0, 1.0].
    """

    @given(document_list_st)
    @settings(max_examples=100)
    def test_claim_schema_invariant(self, documents: list[Document]):
        # Feature: swarmiq-v2, Property 7: Claim schema invariant
        subtask_id = str(uuid.uuid4())

        agent = SummarizerAgent(llm_config=_DUMMY_LLM_CONFIG)

        # Mock _call_llm to return one claim per document
        async def _mock_call_llm(sid: str, docs: list[Document]) -> SummarizerOutput:
            claims = [
                Claim(
                    claim_id=str(uuid.uuid4()),
                    claim_text=f"Claim extracted from {doc.url}",
                    confidence=0.8,
                    source_url=doc.url,
                    subtask_id=sid,
                )
                for doc in docs
            ]
            return SummarizerOutput(claims=claims)

        agent._call_llm = _mock_call_llm  # type: ignore[method-assign]

        result = asyncio.run(agent.summarize(subtask_id, documents))

        assert isinstance(result, SummarizerOutput)
        assert len(result.claims) > 0

        for claim in result.claims:
            assert isinstance(claim.claim_id, str) and claim.claim_id.strip() != "", \
                f"claim_id must be a non-empty string, got: {claim.claim_id!r}"
            assert isinstance(claim.claim_text, str) and claim.claim_text.strip() != "", \
                f"claim_text must be a non-empty string, got: {claim.claim_text!r}"
            assert isinstance(claim.source_url, str) and claim.source_url.strip() != "", \
                f"source_url must be a non-empty string, got: {claim.source_url!r}"
            assert isinstance(claim.subtask_id, str) and claim.subtask_id.strip() != "", \
                f"subtask_id must be a non-empty string, got: {claim.subtask_id!r}"
            assert isinstance(claim.confidence, (int, float)), \
                f"confidence must be a number, got: {type(claim.confidence)}"
            assert 0.0 <= claim.confidence <= 1.0, \
                f"confidence must be in [0.0, 1.0], got: {claim.confidence}"
