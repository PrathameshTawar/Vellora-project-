"""
Property-based tests for Evaluator_Agent (Properties 14, 15, 16, 17).

Feature: swarmiq-v2
Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
"""
from __future__ import annotations

import asyncio
import sys
import types
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

# AssistantAgent must be a real class so EvaluatorAgent.__init__ can call it
sys.modules["autogen"].AssistantAgent = MagicMock  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------

from swarmiq.agents.evaluator import EvaluatorAgent  # noqa: E402
from swarmiq.core.models import EvaluatorOutput, Reference, ScoredClaim  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DUMMY_LLM_CONFIG: dict = {}


def _make_agent(coherence: float, factuality: float) -> EvaluatorAgent:
    """Create an EvaluatorAgent whose _llm_score returns deterministic values."""
    agent = EvaluatorAgent(llm_config=_DUMMY_LLM_CONFIG)
    agent._llm_score = AsyncMock(return_value=(coherence, factuality))  # type: ignore[method-assign]
    return agent


def _make_scored_claim(source_url: str, subtask_id: str = "t1") -> ScoredClaim:
    return ScoredClaim(
        claim_id="c1",
        claim_text="Some claim text.",
        confidence=0.8,
        source_url=source_url,
        subtask_id=subtask_id,
        credibility_score=0.8,
    )


def _make_reference(url: str = "https://example.com/ref") -> Reference:
    return Reference(ref_id=1, url=url, title="A Reference")


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

score_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


# ---------------------------------------------------------------------------
# Property 14: Evaluator component scores in range and schema complete
# ---------------------------------------------------------------------------


class TestEvaluatorScoresInRangeAndSchemaComplete:
    """
    # Feature: swarmiq-v2, Property 14: Evaluator component scores in range and schema complete
    """

    @given(coherence=score_st, factuality=score_st, citation_coverage=score_st)
    @settings(max_examples=100)
    def test_scores_in_range_and_schema_complete(
        self, coherence: float, factuality: float, citation_coverage: float
    ):
        # Feature: swarmiq-v2, Property 14: Evaluator component scores in range and schema complete
        # — for any report processed by the EvaluatorAgent, the output JSON must contain
        # coherence, factuality, citation_coverage, and composite_score fields, each a number
        # in [0.0, 1.0], plus a boolean passed field.

        # Build a report that embeds the source URL so citation_coverage is driven
        # by the hypothesis-generated value via the claim list.
        # We control citation_coverage by including exactly `n_covered` out of `n_total` claims.
        # Simplest approach: use a single claim whose URL is in the report iff citation_coverage >= 0.5.
        # For full control, we embed the URL always and use 1 claim → coverage = 1.0,
        # or use 0 claims → coverage = 1.0 (vacuous). Instead, mock _llm_score and
        # accept whatever citation_coverage the deterministic function computes.

        agent = _make_agent(coherence, factuality)
        # Use a report that contains the claim URL so citation_coverage = 1.0 deterministically
        source_url = "https://example.com/source"
        report = f"Report text. See {source_url} for details."
        claims = [_make_scored_claim(source_url)]
        references = [_make_reference()]

        result = asyncio.run(agent.evaluate(report, claims, references))

        assert isinstance(result, EvaluatorOutput), (
            f"Expected EvaluatorOutput, got {type(result)}"
        )

        # All four score fields must be present and in [0.0, 1.0]
        for field_name, value in [
            ("coherence", result.coherence),
            ("factuality", result.factuality),
            ("citation_coverage", result.citation_coverage),
            ("composite_score", result.composite_score),
        ]:
            assert isinstance(value, float), (
                f"{field_name} must be a float, got {type(value)}"
            )
            assert 0.0 <= value <= 1.0, (
                f"{field_name}={value} is outside [0.0, 1.0]"
            )

        # passed must be a boolean
        assert isinstance(result.passed, bool), (
            f"passed must be bool, got {type(result.passed)}"
        )


# ---------------------------------------------------------------------------
# Property 15: Composite score formula correctness
# ---------------------------------------------------------------------------


class TestCompositeScoreFormulaCorrectness:
    """
    # Feature: swarmiq-v2, Property 15: Composite score formula correctness
    """

    @given(coherence=score_st, factuality=score_st, citation_coverage=score_st)
    @settings(max_examples=100)
    def test_composite_score_formula(
        self, coherence: float, factuality: float, citation_coverage: float
    ):
        # Feature: swarmiq-v2, Property 15: Composite score formula correctness
        # — for any triple (coherence, factuality, citation_coverage) each in [0.0, 1.0],
        # the composite_score must equal coherence * 0.4 + factuality * 0.3 +
        # citation_coverage * 0.3 (within floating-point tolerance of 1e-9).

        agent = _make_agent(coherence, factuality)

        # Control citation_coverage deterministically:
        # Use `n` claims all with URLs present in the report → coverage = n/n = 1.0,
        # or 0 claims → coverage = 1.0 (vacuous).
        # To get the exact citation_coverage value from hypothesis, we craft the claim list:
        # include `covered` claims whose URL appears in the report and `uncovered` whose URL does not.
        # Simplest: use a single claim; embed its URL in the report to get coverage=1.0,
        # or omit it to get coverage=0.0. For arbitrary coverage we need more claims.
        # We'll use 10 claims and embed exactly round(citation_coverage * 10) of their URLs.
        n_total = 10
        n_covered = round(citation_coverage * n_total)
        # Clamp to valid range
        n_covered = max(0, min(n_total, n_covered))

        covered_urls = [f"https://example.com/covered/{i}" for i in range(n_covered)]
        uncovered_urls = [
            f"https://example.com/uncovered/{i}" for i in range(n_total - n_covered)
        ]

        report = "Report text. " + " ".join(covered_urls)
        claims = [
            _make_scored_claim(url) for url in covered_urls + uncovered_urls
        ]
        references = [_make_reference()]

        result = asyncio.run(agent.evaluate(report, claims, references))

        assert isinstance(result, EvaluatorOutput)

        # Recompute expected composite using the actual citation_coverage from the result
        expected = (
            result.coherence * 0.4
            + result.factuality * 0.3
            + result.citation_coverage * 0.3
        )
        # Clamp expected to [0.0, 1.0] as the implementation does
        expected = max(0.0, min(1.0, expected))

        assert abs(result.composite_score - expected) <= 1e-9, (
            f"composite_score={result.composite_score} != expected={expected} "
            f"(coherence={result.coherence}, factuality={result.factuality}, "
            f"citation_coverage={result.citation_coverage})"
        )


# ---------------------------------------------------------------------------
# Property 16: Pass/fail threshold correctness
# ---------------------------------------------------------------------------


class TestPassFailThresholdCorrectness:
    """
    # Feature: swarmiq-v2, Property 16: Pass/fail threshold correctness
    """

    @given(coherence=score_st, factuality=score_st)
    @settings(max_examples=100)
    def test_pass_fail_threshold(self, coherence: float, factuality: float):
        # Feature: swarmiq-v2, Property 16: Pass/fail threshold correctness
        # — for any evaluator output, passed must be True if and only if
        # composite_score >= 0.90.

        agent = _make_agent(coherence, factuality)
        # Use all claims covered → citation_coverage = 1.0
        source_url = "https://example.com/source"
        report = f"Report. {source_url}"
        claims = [_make_scored_claim(source_url)]
        references = [_make_reference()]

        result = asyncio.run(agent.evaluate(report, claims, references))

        assert isinstance(result, EvaluatorOutput)

        if result.composite_score >= 0.90:
            assert result.passed is True, (
                f"passed should be True when composite_score={result.composite_score} >= 0.90"
            )
        else:
            assert result.passed is False, (
                f"passed should be False when composite_score={result.composite_score} < 0.90"
            )


# ---------------------------------------------------------------------------
# Property 17: Failing reports include non-empty deficiencies list
# ---------------------------------------------------------------------------


class TestFailingReportsHaveNonEmptyDeficiencies:
    """
    # Feature: swarmiq-v2, Property 17: Failing reports include non-empty deficiencies list
    """

    @given(coherence=score_st, factuality=score_st)
    @settings(max_examples=100)
    def test_failing_reports_have_deficiencies(
        self, coherence: float, factuality: float
    ):
        # Feature: swarmiq-v2, Property 17: Failing reports include non-empty deficiencies list
        # — for any evaluator output where passed is False, the deficiencies field must be
        # a non-empty list of strings.

        agent = _make_agent(coherence, factuality)
        source_url = "https://example.com/source"
        report = f"Report. {source_url}"
        claims = [_make_scored_claim(source_url)]
        references = [_make_reference()]

        result = asyncio.run(agent.evaluate(report, claims, references))

        assert isinstance(result, EvaluatorOutput)

        if not result.passed:
            assert isinstance(result.deficiencies, list), (
                "deficiencies must be a list when passed is False"
            )
            assert len(result.deficiencies) > 0, (
                f"deficiencies must be non-empty when passed=False "
                f"(composite_score={result.composite_score})"
            )
            for item in result.deficiencies:
                assert isinstance(item, str), (
                    f"Each deficiency must be a string, got {type(item)}"
                )
