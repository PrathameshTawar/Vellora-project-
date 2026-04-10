"""
Unit tests for swarmiq.core.credibility.

Validates: Requirements 4.1, 4.2, 4.3
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from swarmiq.core.credibility import (
    compute_credibility,
    compute_recency,
    get_domain_trust,
    score_claims,
)
from swarmiq.core.models import Claim, ScoredClaim


# ---------------------------------------------------------------------------
# compute_credibility – formula tests
# ---------------------------------------------------------------------------


def test_compute_credibility_all_ones():
    assert compute_credibility(1.0, 1.0, 1.0) == pytest.approx(1.0)


def test_compute_credibility_all_zeros():
    assert compute_credibility(0.0, 0.0, 0.0) == pytest.approx(0.0)


def test_compute_credibility_all_half():
    assert compute_credibility(0.5, 0.5, 0.5) == pytest.approx(0.5)


def test_compute_credibility_domain_only():
    # 1.0 * 0.4 + 0.0 * 0.3 + 0.0 * 0.3 = 0.4
    assert compute_credibility(1.0, 0.0, 0.0) == pytest.approx(0.4)


def test_compute_credibility_recency_only():
    # 0.0 * 0.4 + 1.0 * 0.3 + 0.0 * 0.3 = 0.3
    assert compute_credibility(0.0, 1.0, 0.0) == pytest.approx(0.3)


def test_compute_credibility_agreement_only():
    # 0.0 * 0.4 + 0.0 * 0.3 + 1.0 * 0.3 = 0.3
    assert compute_credibility(0.0, 0.0, 1.0) == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# get_domain_trust – tier lookups
# ---------------------------------------------------------------------------


def test_domain_trust_gov():
    assert get_domain_trust("example.gov") == pytest.approx(1.0)


def test_domain_trust_edu():
    assert get_domain_trust("university.edu") == pytest.approx(1.0)


def test_domain_trust_nature():
    assert get_domain_trust("nature.com") == pytest.approx(1.0)


def test_domain_trust_arxiv():
    assert get_domain_trust("arxiv.org") == pytest.approx(1.0)


def test_domain_trust_bbc():
    assert get_domain_trust("bbc.com") == pytest.approx(0.6)


def test_domain_trust_cnn():
    assert get_domain_trust("cnn.com") == pytest.approx(0.6)


def test_domain_trust_unknown():
    assert get_domain_trust("unknownblog.xyz") == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# compute_recency – boundary tests
# ---------------------------------------------------------------------------


def _ts(days_ago: float) -> str:
    """Return an ISO-8601 UTC timestamp for `days_ago` days in the past."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return dt.isoformat()


def test_recency_now():
    assert compute_recency(_ts(0)) == pytest.approx(1.0)


def test_recency_exactly_90_days():
    assert compute_recency(_ts(90)) == pytest.approx(1.0)


def test_recency_exactly_730_days():
    assert compute_recency(_ts(730)) == pytest.approx(0.0)


def test_recency_midpoint_410_days():
    # Linear interpolation: 1 - (410 - 90) / (730 - 90) = 1 - 320/640 = 0.5
    assert compute_recency(_ts(410)) == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# score_claims
# ---------------------------------------------------------------------------


def _make_claim(claim_id: str, source_url: str, confidence: float = 0.8) -> Claim:
    return Claim(
        claim_id=claim_id,
        claim_text="Some claim text.",
        confidence=confidence,
        source_url=source_url,
        subtask_id="st-1",
    )


def test_score_claims_empty_list():
    assert score_claims([]) == []


def test_score_claims_returns_scored_claim_objects():
    claims = [_make_claim("c1", "https://nature.com/article")]
    result = score_claims(claims)
    assert len(result) == 1
    assert isinstance(result[0], ScoredClaim)
    assert 0.0 <= result[0].credibility_score <= 1.0


def test_score_claims_credibility_in_range():
    claims = [
        _make_claim("c1", "https://arxiv.org/abs/1234"),
        _make_claim("c2", "https://unknownblog.xyz/post"),
        _make_claim("c3", "https://bbc.com/news/article"),
    ]
    result = score_claims(claims)
    assert len(result) == 3
    for sc in result:
        assert 0.0 <= sc.credibility_score <= 1.0


def test_score_claims_higher_credibility_for_more_cited_source():
    """Claims from a source cited multiple times get a higher agreement score."""
    # "popular.gov" appears 3 times; "lonely.xyz" appears once
    claims = [
        _make_claim("c1", "https://popular.gov/a"),
        _make_claim("c2", "https://popular.gov/a"),
        _make_claim("c3", "https://popular.gov/a"),
        _make_claim("c4", "https://lonely.xyz/post"),
    ]
    result = score_claims(claims)
    scored_by_id = {sc.claim_id: sc for sc in result}

    # popular.gov has agreement=1.0 (max citations) and domain_trust=1.0 (.gov)
    # lonely.xyz has agreement=1/3 and domain_trust=0.3
    assert scored_by_id["c1"].credibility_score > scored_by_id["c4"].credibility_score
