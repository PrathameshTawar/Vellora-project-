"""
Property-based tests for the credibility scorer.

Feature: swarmiq-v2
Validates: Requirements 4.1, 4.2, 4.3
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from swarmiq.core.credibility import compute_credibility, compute_recency, get_domain_trust

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

score_component = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
fresh_days = st.integers(min_value=1, max_value=89)
stale_days = st.integers(min_value=731, max_value=3650)
intermediate_days = st.integers(min_value=91, max_value=729)
gov_edu_domains = st.sampled_from(
    ["example.gov", "agency.gov", "university.edu", "school.edu"]
)
news_domains = st.sampled_from(["bbc.com", "cnn.com", "reuters.com"])
unknown_domains = st.text(
    min_size=5, alphabet=st.characters(whitelist_categories=("Ll",))
).map(lambda s: s + ".xyz")


def _iso_timestamp(days_ago: int) -> str:
    """Return an ISO-8601 UTC timestamp for a given number of days in the past."""
    dt = datetime.now(tz=timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


# ---------------------------------------------------------------------------
# Property 8: Credibility formula correctness
# ---------------------------------------------------------------------------


class TestCredibilityFormulaCorrectness:
    """
    # Feature: swarmiq-v2, Property 8: Credibility formula correctness —
    # for any triple (domain_trust, recency, agreement) each in [0.0, 1.0],
    # the computed credibility_score must equal
    # domain_trust * 0.4 + recency * 0.3 + agreement * 0.3.
    """

    @given(score_component, score_component, score_component)
    @settings(max_examples=100)
    def test_compute_credibility_formula(
        self, domain_trust: float, recency: float, agreement: float
    ):
        # Feature: swarmiq-v2, Property 8: Credibility formula correctness
        expected = domain_trust * 0.4 + recency * 0.3 + agreement * 0.3
        result = compute_credibility(domain_trust, recency, agreement)
        assert math.isclose(result, expected, rel_tol=1e-9, abs_tol=1e-12)

    @given(fresh_days)
    @settings(max_examples=100)
    def test_compute_recency_fresh_returns_one(self, days: int):
        # Feature: swarmiq-v2, Property 8: Credibility formula correctness
        timestamp = _iso_timestamp(days)
        result = compute_recency(timestamp)
        assert result == 1.0

    @given(stale_days)
    @settings(max_examples=100)
    def test_compute_recency_stale_returns_zero(self, days: int):
        # Feature: swarmiq-v2, Property 8: Credibility formula correctness
        timestamp = _iso_timestamp(days)
        result = compute_recency(timestamp)
        assert result == 0.0

    @given(intermediate_days)
    @settings(max_examples=100)
    def test_compute_recency_intermediate_returns_between(self, days: int):
        # Feature: swarmiq-v2, Property 8: Credibility formula correctness
        timestamp = _iso_timestamp(days)
        result = compute_recency(timestamp)
        assert 0.0 < result < 1.0

    @given(gov_edu_domains)
    @settings(max_examples=100)
    def test_get_domain_trust_gov_edu_returns_one(self, domain: str):
        # Feature: swarmiq-v2, Property 8: Credibility formula correctness
        result = get_domain_trust(domain)
        assert result == 1.0

    @given(gov_edu_domains)
    @settings(max_examples=100)
    def test_get_domain_trust_edu_returns_one(self, domain: str):
        # Feature: swarmiq-v2, Property 8: Credibility formula correctness
        # Covers .edu specifically (sampled_from includes .edu entries)
        result = get_domain_trust(domain)
        assert result == 1.0

    @given(news_domains)
    @settings(max_examples=100)
    def test_get_domain_trust_news_returns_point_six(self, domain: str):
        # Feature: swarmiq-v2, Property 8: Credibility formula correctness
        result = get_domain_trust(domain)
        assert result == 0.6

    @given(unknown_domains)
    @settings(max_examples=100)
    def test_get_domain_trust_unknown_returns_point_three(self, domain: str):
        # Feature: swarmiq-v2, Property 8: Credibility formula correctness
        result = get_domain_trust(domain)
        assert result == 0.3

    @given(score_component, score_component, score_component)
    @settings(max_examples=100)
    def test_compute_credibility_output_in_range(
        self, domain_trust: float, recency: float, agreement: float
    ):
        # Feature: swarmiq-v2, Property 8: Credibility formula correctness
        result = compute_credibility(domain_trust, recency, agreement)
        assert 0.0 <= result <= 1.0
