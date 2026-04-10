"""
Property-based tests for Visualization_Agent (Property 18).

Feature: swarmiq-v2
Validates: Requirements 8.1
"""
from __future__ import annotations

import sys
import types
import uuid
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

# AssistantAgent must be a real class so agent __init__ calls don't fail
sys.modules["autogen"].AssistantAgent = MagicMock  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------

from swarmiq.agents.visualization import VisualizationAgent, VisualizationOutput  # noqa: E402
from swarmiq.core.models import ScoredClaim  # noqa: E402

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_FAKE_PLOTLY_JSON = '{"data": [], "layout": {}}'


@st.composite
def structured_claim_strategy(draw) -> ScoredClaim:
    """Generate a ScoredClaim whose text contains numerical and/or temporal patterns."""
    n = draw(st.integers(min_value=1, max_value=100))
    year = draw(st.integers(min_value=2000, max_value=2030))
    claim_text = f"The value increased by {n}% in {year}"
    return ScoredClaim(
        claim_id=str(draw(st.uuids())),
        claim_text=claim_text,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        source_url="https://example.com/source",
        subtask_id=str(draw(st.uuids())),
        credibility_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    )


@st.composite
def unstructured_claim_strategy(draw) -> ScoredClaim:
    """Generate a ScoredClaim whose text contains no numerical or temporal patterns."""
    # Use only alphabetic words — no digits, no year-like sequences
    words = draw(
        st.lists(
            st.from_regex(r"[a-zA-Z]{3,10}", fullmatch=True),
            min_size=3,
            max_size=10,
        )
    )
    claim_text = " ".join(words)
    return ScoredClaim(
        claim_id=str(draw(st.uuids())),
        claim_text=claim_text,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
        source_url="https://example.com/source",
        subtask_id=str(draw(st.uuids())),
        credibility_score=draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False)),
    )


# ---------------------------------------------------------------------------
# Property 18: Visualization agent produces figures for structured data
# ---------------------------------------------------------------------------


class TestVisualizationProducesFiguresForStructuredData:
    """
    # Feature: swarmiq-v2, Property 18: Visualization agent produces figures for structured data
    """

    @given(
        structured_claims=st.lists(structured_claim_strategy(), min_size=1, max_size=10),
        extra_unstructured=st.lists(unstructured_claim_strategy(), min_size=0, max_size=5),
    )
    @settings(max_examples=100)
    def test_figures_produced_for_structured_data(
        self,
        structured_claims: list[ScoredClaim],
        extra_unstructured: list[ScoredClaim],
    ):
        # Feature: swarmiq-v2, Property 18: Visualization agent produces figures for structured data
        claims = structured_claims + extra_unstructured

        agent = VisualizationAgent()

        with (
            patch(
                "swarmiq.agents.visualization._build_plotly_bar",
                return_value=_FAKE_PLOTLY_JSON,
            ),
            patch(
                "swarmiq.agents.visualization._build_plotly_line",
                return_value=_FAKE_PLOTLY_JSON,
            ),
        ):
            result = agent.generate(claims)

        assert isinstance(result, VisualizationOutput)
        assert len(result.figures) >= 1, (
            f"Expected at least one figure for claims with structured data, "
            f"got {len(result.figures)}. Claims: {[c.claim_text for c in structured_claims]}"
        )

    # ------------------------------------------------------------------
    # Edge case: claims with no structured data → empty figures list
    # ------------------------------------------------------------------

    @given(
        claims=st.lists(unstructured_claim_strategy(), min_size=0, max_size=10),
    )
    @settings(max_examples=100)
    def test_no_figures_for_unstructured_data(self, claims: list[ScoredClaim]):
        # Feature: swarmiq-v2, Property 18: Visualization agent produces figures for structured data
        agent = VisualizationAgent()

        with (
            patch(
                "swarmiq.agents.visualization._build_plotly_bar",
                return_value=_FAKE_PLOTLY_JSON,
            ),
            patch(
                "swarmiq.agents.visualization._build_plotly_line",
                return_value=_FAKE_PLOTLY_JSON,
            ),
        ):
            result = agent.generate(claims)

        assert isinstance(result, VisualizationOutput)
        assert result.figures == [], (
            f"Expected empty figures list for claims with no structured data, "
            f"got {len(result.figures)} figure(s). Claims: {[c.claim_text for c in claims]}"
        )
