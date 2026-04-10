"""
Credibility Scorer for SwarmIQ v2.

Pure functions for computing credibility scores on claims.
"""
from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urlparse

from swarmiq.core.models import Claim, ScoredClaim

# ---------------------------------------------------------------------------
# Domain trust tiers
# ---------------------------------------------------------------------------

_HIGH_TRUST_TLDS = {".gov", ".edu"}

_HIGH_TRUST_DOMAINS = {
    # Peer-reviewed / academic publishers
    "nature.com",
    "science.org",
    "pubmed.ncbi.nlm.nih.gov",
    "ncbi.nlm.nih.gov",
    "arxiv.org",
    "sciencedirect.com",
    "springer.com",
    "wiley.com",
    "tandfonline.com",
    "journals.plos.org",
    "bmj.com",
    "thelancet.com",
    "nejm.org",
    "jamanetwork.com",
    "cell.com",
    "acs.org",
    "rsc.org",
    "ieeexplore.ieee.org",
    "acm.org",
    "jstor.org",
    "oxfordjournals.org",
    "cambridge.org",
}

_GENERAL_NEWS_DOMAINS = {
    "bbc.com",
    "bbc.co.uk",
    "cnn.com",
    "reuters.com",
    "nytimes.com",
    "theguardian.com",
    "washingtonpost.com",
    "apnews.com",
    "nbcnews.com",
    "abcnews.go.com",
    "cbsnews.com",
    "foxnews.com",
    "usatoday.com",
    "npr.org",
    "politico.com",
    "thehill.com",
    "bloomberg.com",
    "ft.com",
    "economist.com",
    "time.com",
    "newsweek.com",
    "forbes.com",
    "businessinsider.com",
}

_RECENCY_FRESH_DAYS = 90
_RECENCY_STALE_DAYS = 730  # 2 years


# ---------------------------------------------------------------------------
# Core pure functions
# ---------------------------------------------------------------------------


def get_domain_trust(domain: str) -> float:
    """Return a domain trust score in [0.0, 1.0] based on tiered lookup.

    Tiers:
      1.0 — .gov / .edu TLDs, or known peer-reviewed journal domains
      0.6 — known general news domains
      0.3 — everything else (unknown)
    """
    domain = domain.lower().strip()

    # Strip leading "www." for cleaner matching
    if domain.startswith("www."):
        domain = domain[4:]

    # Check TLD
    for tld in _HIGH_TRUST_TLDS:
        if domain.endswith(tld) or ("." + domain).endswith(tld):
            return 1.0

    # Check high-trust domain list (exact match or subdomain)
    for trusted in _HIGH_TRUST_DOMAINS:
        if domain == trusted or domain.endswith("." + trusted):
            return 1.0

    # Check general news domain list
    for news in _GENERAL_NEWS_DOMAINS:
        if domain == news or domain.endswith("." + news):
            return 0.6

    return 0.3


def _extract_domain(url: str) -> str:
    """Extract the hostname from a URL string."""
    try:
        parsed = urlparse(url)
        return parsed.hostname or url
    except Exception:
        return url


def compute_recency(retrieved_at: str) -> float:
    """Compute a recency score in [0.0, 1.0] from an ISO-8601 timestamp.

    - 1.0  if age ≤ 90 days
    - 0.0  if age ≥ 730 days (2 years)
    - Linear interpolation between 90 and 730 days
    """
    dt = datetime.fromisoformat(retrieved_at)
    # Make timezone-aware if naive (assume UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(tz=timezone.utc)
    age_days = (now - dt).total_seconds() / 86400.0

    if age_days <= _RECENCY_FRESH_DAYS:
        return 1.0
    if age_days >= _RECENCY_STALE_DAYS:
        return 0.0
    # Linear decay from 1.0 → 0.0 over [90, 730]
    return 1.0 - (age_days - _RECENCY_FRESH_DAYS) / (_RECENCY_STALE_DAYS - _RECENCY_FRESH_DAYS)


def compute_credibility(domain_trust: float, recency: float, agreement: float) -> float:
    """Compute the credibility score from its three components.

    Formula: domain_trust * 0.4 + recency * 0.3 + agreement * 0.3

    All inputs are expected to be in [0.0, 1.0].
    """
    return domain_trust * 0.4 + recency * 0.3 + agreement * 0.3


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------


def score_claims(claims: list[Claim]) -> list[ScoredClaim]:
    """Attach credibility scores to a list of claims.

    Agreement is computed as the normalized frequency of each source_url
    within the claims list (how many claims share the same source_url,
    normalized to [0, 1]).

    Returns a list of ScoredClaim objects in the same order as the input.
    """
    if not claims:
        return []

    # Count occurrences of each source_url
    url_counts: dict[str, int] = {}
    for claim in claims:
        url_counts[claim.source_url] = url_counts.get(claim.source_url, 0) + 1

    max_count = max(url_counts.values())

    scored: list[ScoredClaim] = []
    for claim in claims:
        domain = _extract_domain(claim.source_url)
        domain_trust = get_domain_trust(domain)

        # Use the claim's subtask_id as a proxy for retrieved_at if not available;
        # ScoredClaim inherits from Claim which has no retrieved_at field.
        # We need a retrieved_at — look it up from the Document if available,
        # but since Claim doesn't carry it, default recency to 1.0 here.
        # Callers that have Document objects should call compute_recency directly.
        recency = 1.0  # default when no timestamp available on the Claim itself

        # Normalized agreement: fraction of claims sharing this URL
        agreement = url_counts[claim.source_url] / max_count

        cred = compute_credibility(domain_trust, recency, agreement)

        scored.append(
            ScoredClaim(
                claim_id=claim.claim_id,
                claim_text=claim.claim_text,
                confidence=claim.confidence,
                source_url=claim.source_url,
                subtask_id=claim.subtask_id,
                credibility_score=cred,
            )
        )

    return scored
