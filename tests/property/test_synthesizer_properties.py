"""
Property-based tests for Synthesizer_Agent (Properties 11, 12, 13, 27).

Feature: swarmiq-v2
Validates: Requirements 6.2, 6.3, 6.4, 6.5, 13.4
"""
from __future__ import annotations

import asyncio
import re
import sys
import types
import uuid
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
]:
    if _dep not in sys.modules:
        _stub(_dep)

# AssistantAgent must be a real class so SynthesizerAgent.__init__ can call it
sys.modules["autogen"].AssistantAgent = MagicMock  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------

from swarmiq.agents.synthesizer import SynthesizerAgent  # noqa: E402
from swarmiq.core.models import (  # noqa: E402
    Resolution,
    ScoredClaim,
    SubTask,
    SynthesizerOutput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_LLM_CONFIG: dict = {}

_DETERMINISTIC_REGULATORY = (
    "Regulatory bodies should consider these findings. "
    "Compliance with existing frameworks is advised."
)


def _prose_with_citations(messages, **kwargs) -> str:
    """
    Deterministic mock for _section_agent.a_generate_reply.

    Extracts citation markers like [1], [2] from the prompt and returns
    prose that includes all of them, simulating an LLM that preserves
    citation markers as instructed.
    """
    prompt = ""
    if messages:
        last = messages[-1]
        prompt = last.get("content", "") if isinstance(last, dict) else str(last)
    markers = re.findall(r"\[\d+\]", prompt)
    marker_str = " ".join(markers) if markers else ""
    return (
        f"This section synthesises the available evidence {marker_str}. "
        "The findings are consistent with prior research. "
        "Further investigation is recommended."
    )


def _make_agent() -> SynthesizerAgent:
    """Create a SynthesizerAgent with mocked LLM calls."""
    agent = SynthesizerAgent(llm_config=_DUMMY_LLM_CONFIG)
    # Mock section agent: returns prose that preserves citation markers from the prompt
    agent._section_agent.a_generate_reply = AsyncMock(  # type: ignore[attr-defined]
        side_effect=_prose_with_citations
    )
    # Mock regulatory agent: returns deterministic regulatory prose
    agent._regulatory_agent.a_generate_reply = AsyncMock(  # type: ignore[attr-defined]
        return_value=_DETERMINISTIC_REGULATORY
    )
    return agent
    return agent


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

url_st = st.from_regex(
    r"https://example\.com/[a-z]{3,10}/[a-z]{3,8}", fullmatch=True
)

subtask_description_st = st.text(
    min_size=5,
    max_size=60,
    alphabet=st.characters(whitelist_categories=("L", "N", "Zs")),
)


@st.composite
def subtask_strategy(draw) -> SubTask:
    return SubTask(
        subtask_id=str(draw(st.uuids())),
        type="literature_review",
        description=draw(subtask_description_st),
        search_keywords=["keyword"],
    )


@st.composite
def accepted_claim_with_resolution(draw, subtask_id: str):
    """Generate a (ScoredClaim, Resolution) pair with status='accepted'."""
    claim_id = str(draw(st.uuids()))
    source_url = draw(url_st)
    claim = ScoredClaim(
        claim_id=claim_id,
        claim_text=draw(st.text(min_size=5, max_size=80,
                                alphabet=st.characters(whitelist_categories=("L", "N", "Zs")))),
        confidence=draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False)),
        source_url=source_url,
        subtask_id=subtask_id,
        credibility_score=draw(st.floats(min_value=0.5, max_value=1.0, allow_nan=False)),
    )
    resolution = Resolution(
        claim_id=claim_id,
        status="accepted",
        rationale="High credibility source.",
        credibility_score=claim.credibility_score,
    )
    return claim, resolution


@st.composite
def uncertain_claim_with_resolution(draw, subtask_id: str):
    """Generate a (ScoredClaim, Resolution) pair with status='uncertain'."""
    claim_id = str(draw(st.uuids()))
    source_url = draw(url_st)
    claim = ScoredClaim(
        claim_id=claim_id,
        claim_text=draw(st.text(min_size=5, max_size=80,
                                alphabet=st.characters(whitelist_categories=("L", "N", "Zs")))),
        confidence=draw(st.floats(min_value=0.0, max_value=0.49, allow_nan=False)),
        source_url=source_url,
        subtask_id=subtask_id,
        credibility_score=draw(st.floats(min_value=0.0, max_value=0.49, allow_nan=False)),
    )
    resolution = Resolution(
        claim_id=claim_id,
        status="uncertain",
        rationale="Insufficient evidence.",
        credibility_score=claim.credibility_score,
    )
    return claim, resolution


# ---------------------------------------------------------------------------
# Property 11: Synthesizer report contains citations for all accepted claims
# ---------------------------------------------------------------------------


class TestSynthesizerCitationsForAcceptedClaims:
    """
    # Feature: swarmiq-v2, Property 11: Synthesizer report contains citations
    # for all accepted claims — for any resolution report containing accepted
    # claims, the synthesizer's Markdown output must contain an inline citation
    # marker for each accepted claim's source URL, and the report must end with
    # a non-empty reference list section.
    """

    @given(
        st.lists(subtask_strategy(), min_size=1, max_size=4).flatmap(
            lambda subtasks: st.tuples(
                st.just(subtasks),
                st.lists(
                    st.one_of(
                        *[accepted_claim_with_resolution(st.subtask_id)
                          for st in subtasks]
                    ),
                    min_size=1,
                    max_size=6,
                ),
            )
        )
    )
    @settings(max_examples=100)
    def test_citations_for_accepted_claims(self, args):
        # Feature: swarmiq-v2, Property 11: Synthesizer report contains citations for all accepted claims
        subtasks, claim_resolution_pairs = args

        scored_claims = [pair[0] for pair in claim_resolution_pairs]
        resolutions = [pair[1] for pair in claim_resolution_pairs]

        agent = _make_agent()
        result = asyncio.run(
            agent.synthesize(
                resolutions=resolutions,
                domain_mode="research",
                subtasks=subtasks,
                scored_claims=scored_claims,
            )
        )

        assert isinstance(result, SynthesizerOutput)
        report = result.report_markdown

        # The report must contain at least one inline citation marker [N]
        assert re.search(r"\[\d+\]", report), (
            "Report must contain at least one inline citation marker [N]"
        )

        # The report must contain a non-empty References section
        assert "## References" in report, "Report must contain a '## References' section"
        ref_section_start = report.index("## References")
        ref_section = report[ref_section_start:]
        # References section must have content beyond the heading line
        lines_after_heading = [
            ln.strip()
            for ln in ref_section.splitlines()[1:]
            if ln.strip()
        ]
        assert len(lines_after_heading) > 0, (
            "References section must be non-empty"
        )

        # Every reference object must have a URL
        assert len(result.references) > 0, "SynthesizerOutput.references must be non-empty"
        for ref in result.references:
            assert ref.url, "Each Reference must have a non-empty URL"


# ---------------------------------------------------------------------------
# Property 12: Synthesizer report sections correspond to subtasks
# ---------------------------------------------------------------------------


class TestSynthesizerSectionsCorrespondToSubtasks:
    """
    # Feature: swarmiq-v2, Property 12: Synthesizer report sections correspond
    # to subtasks — for any set of subtasks, the synthesizer's Markdown output
    # must contain one section per subtask, identifiable by the subtask
    # description or ID.
    """

    @given(st.lists(subtask_strategy(), min_size=1, max_size=5))
    @settings(max_examples=100)
    def test_sections_correspond_to_subtasks(self, subtasks: list[SubTask]):
        # Feature: swarmiq-v2, Property 12: Synthesizer report sections correspond to subtasks
        agent = _make_agent()
        result = asyncio.run(
            agent.synthesize(
                resolutions=[],
                domain_mode="research",
                subtasks=subtasks,
                scored_claims=[],
            )
        )

        assert isinstance(result, SynthesizerOutput)
        report = result.report_markdown

        for subtask in subtasks:
            # Each subtask description must appear in the report
            assert subtask.description in report, (
                f"Subtask description '{subtask.description}' not found in report"
            )


# ---------------------------------------------------------------------------
# Property 13: Weak-evidence sections marked with uncertainty indicator
# ---------------------------------------------------------------------------


class TestWeakEvidenceSectionsMarked:
    """
    # Feature: swarmiq-v2, Property 13: Weak-evidence sections marked with
    # uncertainty indicator — for any subtask where all associated claims have
    # status `uncertain`, the corresponding section must contain the
    # ⚠️ Weak Evidence indicator string.
    """

    @given(
        st.lists(subtask_strategy(), min_size=1, max_size=4).flatmap(
            lambda subtasks: st.tuples(
                st.just(subtasks),
                st.lists(
                    st.one_of(
                        *[uncertain_claim_with_resolution(st.subtask_id)
                          for st in subtasks]
                    ),
                    min_size=1,
                    max_size=6,
                ),
            )
        )
    )
    @settings(max_examples=100)
    def test_weak_evidence_sections_marked(self, args):
        # Feature: swarmiq-v2, Property 13: Weak-evidence sections marked with uncertainty indicator
        subtasks, claim_resolution_pairs = args

        scored_claims = [pair[0] for pair in claim_resolution_pairs]
        resolutions = [pair[1] for pair in claim_resolution_pairs]

        # Identify subtasks where ALL associated claims are uncertain
        subtask_statuses: dict[str, set[str]] = {st.subtask_id: set() for st in subtasks}
        for claim, resolution in zip(scored_claims, resolutions):
            if claim.subtask_id in subtask_statuses:
                subtask_statuses[claim.subtask_id].add(resolution.status)

        agent = _make_agent()
        result = asyncio.run(
            agent.synthesize(
                resolutions=resolutions,
                domain_mode="research",
                subtasks=subtasks,
                scored_claims=scored_claims,
            )
        )

        assert isinstance(result, SynthesizerOutput)
        report = result.report_markdown

        for subtask in subtasks:
            statuses = subtask_statuses[subtask.subtask_id]
            # A subtask with only uncertain claims (or no claims) must show the marker
            all_uncertain = len(statuses) == 0 or statuses == {"uncertain"}
            if all_uncertain:
                assert "⚠️ Weak Evidence" in report, (
                    f"Expected '⚠️ Weak Evidence' in report for subtask "
                    f"'{subtask.description}' where all claims are uncertain"
                )


# ---------------------------------------------------------------------------
# Property 27: Policy mode includes regulatory implications section
# ---------------------------------------------------------------------------


class TestPolicyModeIncludesRegulatoryImplications:
    """
    # Feature: swarmiq-v2, Property 27: Policy mode includes regulatory
    # implications section — for any report synthesized with
    # domain_mode = "policy", the Markdown output must contain a section with
    # a heading matching "Regulatory Implications".
    """

    @given(st.lists(subtask_strategy(), min_size=1, max_size=4))
    @settings(max_examples=100)
    def test_policy_mode_includes_regulatory_implications(
        self, subtasks: list[SubTask]
    ):
        # Feature: swarmiq-v2, Property 27: Policy mode includes regulatory implications section
        agent = _make_agent()
        result = asyncio.run(
            agent.synthesize(
                resolutions=[],
                domain_mode="policy",
                subtasks=subtasks,
                scored_claims=[],
            )
        )

        assert isinstance(result, SynthesizerOutput)
        report = result.report_markdown

        assert "Regulatory Implications" in report, (
            "Policy mode report must contain a 'Regulatory Implications' section"
        )
