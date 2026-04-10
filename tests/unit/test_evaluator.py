"""
Unit tests for EvaluatorAgent.

Tests:
1.  coherence=1.0, factuality=1.0, citation_coverage=1.0 → composite_score=1.0, passed=True
2.  coherence=0.0, factuality=0.0, citation_coverage=0.0 → composite_score=0.0, passed=False
3.  coherence=0.9, factuality=0.9, citation_coverage=0.9 → composite_score=0.9, passed=True (boundary)
4.  coherence=0.89, factuality=0.9, citation_coverage=0.9 → composite_score < 0.9, passed=False
5.  When passed=False, deficiencies is a non-empty list
6.  When passed=True, deficiencies is an empty list
7.  citation_coverage: 1 claim with URL in report → 1.0
8.  citation_coverage: 1 claim with URL NOT in report → 0.0
9.  citation_coverage with 0 claims → 1.0 (vacuous)
10. _parse_llm_scores('{"coherence": 0.8, "factuality": 0.7}') returns (0.8, 0.7)
11. _parse_llm_scores('not json') returns (0.5, 0.5) fallback
12. _parse_llm_scores('{"coherence": 1.5, "factuality": -0.1}') clamps to (1.0, 0.0)

Requirements: 7.2, 7.3, 7.5
"""
from __future__ import annotations

import asyncio
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
from swarmiq.agents.evaluator import (  # noqa: E402
    EvaluatorAgent,
    _compute_citation_coverage,
)
from swarmiq.core.models import EvaluatorOutput, Reference, ScoredClaim  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LLM_CONFIG: dict = {"model": "gpt-4o", "api_key": "fake"}


def _make_agent(coherence: float = 1.0, factuality: float = 1.0) -> EvaluatorAgent:
    """Create an EvaluatorAgent with _llm_score mocked to return fixed values."""
    agent = EvaluatorAgent(llm_config=_LLM_CONFIG)
    agent._llm_score = AsyncMock(return_value=(coherence, factuality))
    return agent


def _claim(url: str = "https://example.com/paper1") -> ScoredClaim:
    return ScoredClaim(
        claim_id=str(uuid.uuid4()),
        claim_text="Some claim text.",
        confidence=0.9,
        source_url=url,
        subtask_id="st-1",
        credibility_score=0.85,
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Test 1: All perfect scores → composite=1.0, passed=True
# ---------------------------------------------------------------------------


def test_all_perfect_scores_composite_one_passed():
    """coherence=1.0, factuality=1.0, citation_coverage=1.0 → composite=1.0, passed=True."""
    url = "https://example.com/paper1"
    report = f"Report text. {url}"
    claims = [_claim(url)]

    agent = _make_agent(coherence=1.0, factuality=1.0)
    result = _run(agent.evaluate(report, claims, []))

    assert isinstance(result, EvaluatorOutput)
    assert result.composite_score == 1.0
    assert result.passed is True


# ---------------------------------------------------------------------------
# Test 2: All zero scores → composite=0.0, passed=False
# ---------------------------------------------------------------------------


def test_all_zero_scores_composite_zero_failed():
    """coherence=0.0, factuality=0.0, citation_coverage=0.0 → composite=0.0, passed=False."""
    report = "Report with no cited URLs."
    claims = [_claim("https://not-in-report.com/paper")]

    agent = _make_agent(coherence=0.0, factuality=0.0)
    result = _run(agent.evaluate(report, claims, []))

    assert isinstance(result, EvaluatorOutput)
    assert result.composite_score == 0.0
    assert result.passed is False


# ---------------------------------------------------------------------------
# Test 3: Boundary — composite exactly 0.9 → passed=True
# ---------------------------------------------------------------------------


def test_boundary_composite_exactly_0_9_passes():
    """coherence=0.9, factuality=0.9, citation_coverage=0.9 → composite=0.9, passed=True."""
    url = "https://example.com/paper1"
    # Build a report where exactly 9 out of 10 claims have their URL present
    claims = [_claim(url) for _ in range(9)] + [_claim("https://missing.com/paper")]
    report = " ".join(url for _ in range(9))  # only the first URL appears

    agent = _make_agent(coherence=0.9, factuality=0.9)
    result = _run(agent.evaluate(report, claims, []))

    assert isinstance(result, EvaluatorOutput)
    assert abs(result.composite_score - 0.9) < 1e-9
    assert result.passed is True


# ---------------------------------------------------------------------------
# Test 4: Just below boundary → passed=False
# ---------------------------------------------------------------------------


def test_just_below_boundary_fails():
    """coherence=0.89, factuality=0.9, citation_coverage=0.9 → composite < 0.9, passed=False."""
    url = "https://example.com/paper1"
    claims = [_claim(url) for _ in range(9)] + [_claim("https://missing.com/paper")]
    report = " ".join(url for _ in range(9))

    agent = _make_agent(coherence=0.89, factuality=0.9)
    result = _run(agent.evaluate(report, claims, []))

    assert isinstance(result, EvaluatorOutput)
    assert result.composite_score < 0.9
    assert result.passed is False


# ---------------------------------------------------------------------------
# Test 5: When passed=False, deficiencies is non-empty
# ---------------------------------------------------------------------------


def test_failed_result_has_non_empty_deficiencies():
    """When passed=False, deficiencies is a non-empty list."""
    report = "Report with no cited URLs."
    claims = [_claim("https://not-in-report.com/paper")]

    agent = _make_agent(coherence=0.0, factuality=0.0)
    result = _run(agent.evaluate(report, claims, []))

    assert isinstance(result, EvaluatorOutput)
    assert result.passed is False
    assert isinstance(result.deficiencies, list)
    assert len(result.deficiencies) > 0


# ---------------------------------------------------------------------------
# Test 6: When passed=True, deficiencies is empty
# ---------------------------------------------------------------------------


def test_passed_result_has_empty_deficiencies():
    """When passed=True, deficiencies is an empty list."""
    url = "https://example.com/paper1"
    report = f"Report text. {url}"
    claims = [_claim(url)]

    agent = _make_agent(coherence=1.0, factuality=1.0)
    result = _run(agent.evaluate(report, claims, []))

    assert isinstance(result, EvaluatorOutput)
    assert result.passed is True
    assert result.deficiencies == []


# ---------------------------------------------------------------------------
# Test 7: citation_coverage — 1 claim with URL in report → 1.0
# ---------------------------------------------------------------------------


def test_citation_coverage_url_present_returns_1():
    """1 claim with URL in report → citation_coverage=1.0."""
    url = "https://example.com/paper1"
    report = f"See {url} for details."
    claims = [_claim(url)]

    coverage = _compute_citation_coverage(report, claims)

    assert coverage == 1.0


# ---------------------------------------------------------------------------
# Test 8: citation_coverage — 1 claim with URL NOT in report → 0.0
# ---------------------------------------------------------------------------


def test_citation_coverage_url_absent_returns_0():
    """1 claim with URL NOT in report → citation_coverage=0.0."""
    report = "Report with no URLs."
    claims = [_claim("https://not-in-report.com/paper")]

    coverage = _compute_citation_coverage(report, claims)

    assert coverage == 0.0


# ---------------------------------------------------------------------------
# Test 9: citation_coverage with 0 claims → 1.0 (vacuous)
# ---------------------------------------------------------------------------


def test_citation_coverage_no_claims_returns_1():
    """citation_coverage with 0 claims → 1.0 (vacuously covered)."""
    coverage = _compute_citation_coverage("Any report text.", [])

    assert coverage == 1.0


# ---------------------------------------------------------------------------
# Test 10: _parse_llm_scores valid JSON → correct floats
# ---------------------------------------------------------------------------


def test_parse_llm_scores_valid_json():
    """_parse_llm_scores('{"coherence": 0.8, "factuality": 0.7}') returns (0.8, 0.7)."""
    coherence, factuality = EvaluatorAgent._parse_llm_scores(
        '{"coherence": 0.8, "factuality": 0.7}'
    )

    assert coherence == 0.8
    assert factuality == 0.7


# ---------------------------------------------------------------------------
# Test 11: _parse_llm_scores invalid JSON → (0.5, 0.5) fallback
# ---------------------------------------------------------------------------


def test_parse_llm_scores_invalid_json_returns_fallback():
    """_parse_llm_scores('not json') returns (0.5, 0.5) fallback."""
    coherence, factuality = EvaluatorAgent._parse_llm_scores("not json")

    assert coherence == 0.5
    assert factuality == 0.5


# ---------------------------------------------------------------------------
# Test 12: _parse_llm_scores out-of-range values → clamped to [0.0, 1.0]
# ---------------------------------------------------------------------------


def test_parse_llm_scores_clamps_out_of_range():
    """_parse_llm_scores('{"coherence": 1.5, "factuality": -0.1}') clamps to (1.0, 0.0)."""
    coherence, factuality = EvaluatorAgent._parse_llm_scores(
        '{"coherence": 1.5, "factuality": -0.1}'
    )

    assert coherence == 1.0
    assert factuality == 0.0
