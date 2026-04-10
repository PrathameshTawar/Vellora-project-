"""
Unit tests for SynthesizerAgent.

Tests:
1.  Accepted claims produce inline citation markers [1], [2] in the report
2.  Reference list is non-empty when there are accepted claims
3.  APA format (research mode) uses "Retrieved from" in reference list
4.  Business mode uses numbered format [1] in reference list
5.  Section with all uncertain claims contains ⚠️ Weak Evidence
6.  Section with no claims contains ⚠️ Weak Evidence
7.  Policy mode report contains "Regulatory Implications" heading
8.  Non-policy mode report does NOT contain "Regulatory Implications"
9.  Each subtask description appears in the report
10. Empty resolutions and claims → report still has subtask sections

Requirements: 6.2, 6.3, 6.4, 6.5, 13.4
"""
from __future__ import annotations

import asyncio
import re
import sys
import types
import uuid
from unittest.mock import AsyncMock, MagicMock

# ---------------------------------------------------------------------------
# Stub out autogen before any swarmiq imports
# ---------------------------------------------------------------------------
_autogen_mod = types.ModuleType("autogen")
_autogen_mod.AssistantAgent = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("autogen", _autogen_mod)

# ---------------------------------------------------------------------------
# Now safe to import swarmiq
# ---------------------------------------------------------------------------
from swarmiq.agents.synthesizer import SynthesizerAgent  # noqa: E402
from swarmiq.core.models import (  # noqa: E402
    Resolution,
    ScoredClaim,
    SubTask,
    SynthesizerOutput,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_LLM_CONFIG: dict = {"model": "gpt-4o", "api_key": "fake"}

_PROSE_REPLY = (
    "This section synthesises the available evidence [1]. "
    "The findings are consistent with prior research."
)
_REGULATORY_REPLY = (
    "Regulatory bodies should consider these findings. "
    "Compliance with existing frameworks is advised."
)


def _make_agent() -> SynthesizerAgent:
    """Create a SynthesizerAgent with mocked LLM calls."""
    agent = SynthesizerAgent(llm_config=_LLM_CONFIG)
    agent._section_agent.a_generate_reply = AsyncMock(return_value=_PROSE_REPLY)
    agent._regulatory_agent.a_generate_reply = AsyncMock(return_value=_REGULATORY_REPLY)
    return agent


def _subtask(description: str = "Subtask A") -> SubTask:
    return SubTask(
        subtask_id=str(uuid.uuid4()),
        type="literature_review",
        description=description,
        search_keywords=["keyword"],
    )


def _accepted_pair(subtask_id: str, url: str = "https://example.com/paper1"):
    """Return (ScoredClaim, Resolution) with status='accepted'."""
    claim_id = str(uuid.uuid4())
    claim = ScoredClaim(
        claim_id=claim_id,
        claim_text="The sky is blue.",
        confidence=0.9,
        source_url=url,
        subtask_id=subtask_id,
        credibility_score=0.85,
    )
    resolution = Resolution(
        claim_id=claim_id,
        status="accepted",
        rationale="High credibility.",
        credibility_score=0.85,
    )
    return claim, resolution


def _uncertain_pair(subtask_id: str, url: str = "https://example.com/paper2"):
    """Return (ScoredClaim, Resolution) with status='uncertain'."""
    claim_id = str(uuid.uuid4())
    claim = ScoredClaim(
        claim_id=claim_id,
        claim_text="Results are inconclusive.",
        confidence=0.3,
        source_url=url,
        subtask_id=subtask_id,
        credibility_score=0.3,
    )
    resolution = Resolution(
        claim_id=claim_id,
        status="uncertain",
        rationale="Insufficient evidence.",
        credibility_score=0.3,
    )
    return claim, resolution


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Test 1: Accepted claims produce inline citation markers [1], [2] in the report
# ---------------------------------------------------------------------------


def test_accepted_claims_produce_inline_citation_markers():
    """Accepted claims produce inline citation markers [1], [2] in the report."""
    st = _subtask("Climate change overview")
    claim1, res1 = _accepted_pair(st.subtask_id, "https://example.com/paper1")
    claim2, res2 = _accepted_pair(st.subtask_id, "https://example.com/paper2")

    # Make section agent echo back the citation markers from the prompt
    def _echo_citations(messages, **kwargs):
        prompt = messages[-1].get("content", "") if messages else ""
        markers = re.findall(r"\[\d+\]", prompt)
        return " ".join(markers) + " Evidence supports the claim."

    agent = _make_agent()
    agent._section_agent.a_generate_reply = AsyncMock(side_effect=_echo_citations)

    result = _run(
        agent.synthesize(
            resolutions=[res1, res2],
            domain_mode="research",
            subtasks=[st],
            scored_claims=[claim1, claim2],
        )
    )

    assert isinstance(result, SynthesizerOutput)
    markers = re.findall(r"\[\d+\]", result.report_markdown)
    assert len(markers) >= 2, f"Expected at least 2 citation markers, got: {markers}"
    assert "[1]" in result.report_markdown
    assert "[2]" in result.report_markdown


# ---------------------------------------------------------------------------
# Test 2: Reference list is non-empty when there are accepted claims
# ---------------------------------------------------------------------------


def test_reference_list_non_empty_with_accepted_claims():
    """Reference list is non-empty when there are accepted claims."""
    st = _subtask("AI safety research")
    claim, res = _accepted_pair(st.subtask_id)

    agent = _make_agent()
    result = _run(
        agent.synthesize(
            resolutions=[res],
            domain_mode="research",
            subtasks=[st],
            scored_claims=[claim],
        )
    )

    assert isinstance(result, SynthesizerOutput)
    assert len(result.references) > 0, "references list must be non-empty"
    assert "## References" in result.report_markdown
    # Ensure there's actual content after the heading
    ref_idx = result.report_markdown.index("## References")
    ref_section = result.report_markdown[ref_idx:]
    content_lines = [ln.strip() for ln in ref_section.splitlines()[1:] if ln.strip()]
    assert len(content_lines) > 0, "References section must have content"


# ---------------------------------------------------------------------------
# Test 3: APA format (research mode) uses "Retrieved from" in reference list
# ---------------------------------------------------------------------------


def test_research_mode_uses_apa_format():
    """APA format (research mode) uses 'Retrieved from' in reference list."""
    st = _subtask("Quantum computing")
    claim, res = _accepted_pair(st.subtask_id)

    agent = _make_agent()
    result = _run(
        agent.synthesize(
            resolutions=[res],
            domain_mode="research",
            subtasks=[st],
            scored_claims=[claim],
        )
    )

    assert isinstance(result, SynthesizerOutput)
    assert "Retrieved from" in result.report_markdown, (
        "Research mode should use APA format with 'Retrieved from'"
    )


# ---------------------------------------------------------------------------
# Test 4: Business mode uses numbered format [1] in reference list
# ---------------------------------------------------------------------------


def test_business_mode_uses_numbered_format():
    """Business mode uses numbered format [1] in reference list."""
    st = _subtask("Market analysis")
    claim, res = _accepted_pair(st.subtask_id)

    agent = _make_agent()
    result = _run(
        agent.synthesize(
            resolutions=[res],
            domain_mode="business",
            subtasks=[st],
            scored_claims=[claim],
        )
    )

    assert isinstance(result, SynthesizerOutput)
    # Business mode reference list entries start with [N]
    ref_idx = result.report_markdown.index("## References")
    ref_section = result.report_markdown[ref_idx:]
    assert re.search(r"\[\d+\]", ref_section), (
        "Business mode should use numbered format [N] in reference list"
    )
    assert "Retrieved from" not in ref_section, (
        "Business mode should NOT use APA 'Retrieved from' format"
    )


# ---------------------------------------------------------------------------
# Test 5: Section with all uncertain claims contains ⚠️ Weak Evidence
# ---------------------------------------------------------------------------


def test_all_uncertain_claims_shows_weak_evidence_marker():
    """Section with all uncertain claims contains ⚠️ Weak Evidence."""
    st = _subtask("Speculative technology")
    claim, res = _uncertain_pair(st.subtask_id)

    agent = _make_agent()
    result = _run(
        agent.synthesize(
            resolutions=[res],
            domain_mode="research",
            subtasks=[st],
            scored_claims=[claim],
        )
    )

    assert isinstance(result, SynthesizerOutput)
    assert "⚠️ Weak Evidence" in result.report_markdown, (
        "Section with all uncertain claims must contain ⚠️ Weak Evidence marker"
    )


# ---------------------------------------------------------------------------
# Test 6: Section with no claims contains ⚠️ Weak Evidence
# ---------------------------------------------------------------------------


def test_section_with_no_claims_shows_weak_evidence_marker():
    """Section with no claims contains ⚠️ Weak Evidence."""
    st = _subtask("Empty research area")

    agent = _make_agent()
    result = _run(
        agent.synthesize(
            resolutions=[],
            domain_mode="research",
            subtasks=[st],
            scored_claims=[],
        )
    )

    assert isinstance(result, SynthesizerOutput)
    assert "⚠️ Weak Evidence" in result.report_markdown, (
        "Section with no claims must contain ⚠️ Weak Evidence marker"
    )


# ---------------------------------------------------------------------------
# Test 7: Policy mode report contains "Regulatory Implications" heading
# ---------------------------------------------------------------------------


def test_policy_mode_contains_regulatory_implications_heading():
    """Policy mode report contains 'Regulatory Implications' heading."""
    st = _subtask("Healthcare policy")

    agent = _make_agent()
    result = _run(
        agent.synthesize(
            resolutions=[],
            domain_mode="policy",
            subtasks=[st],
            scored_claims=[],
        )
    )

    assert isinstance(result, SynthesizerOutput)
    assert "Regulatory Implications" in result.report_markdown, (
        "Policy mode report must contain 'Regulatory Implications' heading"
    )


# ---------------------------------------------------------------------------
# Test 8: Non-policy mode report does NOT contain "Regulatory Implications"
# ---------------------------------------------------------------------------


def test_non_policy_mode_does_not_contain_regulatory_implications():
    """Non-policy mode report does NOT contain 'Regulatory Implications'."""
    st = _subtask("General research")

    for mode in ("research", "business"):
        agent = _make_agent()
        result = _run(
            agent.synthesize(
                resolutions=[],
                domain_mode=mode,
                subtasks=[st],
                scored_claims=[],
            )
        )

        assert isinstance(result, SynthesizerOutput)
        assert "Regulatory Implications" not in result.report_markdown, (
            f"Mode '{mode}' report must NOT contain 'Regulatory Implications'"
        )


# ---------------------------------------------------------------------------
# Test 9: Each subtask description appears in the report
# ---------------------------------------------------------------------------


def test_each_subtask_description_appears_in_report():
    """Each subtask description appears in the report."""
    subtasks = [
        _subtask("Introduction to neural networks"),
        _subtask("Deep learning applications"),
        _subtask("Future of AI research"),
    ]

    agent = _make_agent()
    result = _run(
        agent.synthesize(
            resolutions=[],
            domain_mode="research",
            subtasks=subtasks,
            scored_claims=[],
        )
    )

    assert isinstance(result, SynthesizerOutput)
    for st in subtasks:
        assert st.description in result.report_markdown, (
            f"Subtask description '{st.description}' must appear in the report"
        )


# ---------------------------------------------------------------------------
# Test 10: Empty resolutions and claims → report still has subtask sections
# ---------------------------------------------------------------------------


def test_empty_resolutions_and_claims_still_produces_subtask_sections():
    """Empty resolutions and claims → report still has subtask sections."""
    subtasks = [
        _subtask("Background context"),
        _subtask("Methodology overview"),
    ]

    agent = _make_agent()
    result = _run(
        agent.synthesize(
            resolutions=[],
            domain_mode="research",
            subtasks=subtasks,
            scored_claims=[],
        )
    )

    assert isinstance(result, SynthesizerOutput)
    assert result.report_markdown, "Report markdown must not be empty"
    # Each subtask should have a <details> section
    assert result.report_markdown.count("<details>") == len(subtasks), (
        "Report must have one <details> section per subtask"
    )
    for st in subtasks:
        assert st.description in result.report_markdown, (
            f"Subtask '{st.description}' must appear even with no claims"
        )
