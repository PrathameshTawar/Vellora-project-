"""
Evaluator_Agent for SwarmIQ v2.

Scores the final research report on three dimensions:
  - coherence    (LLM-scored, 0.0–1.0)
  - factuality   (LLM-scored, 0.0–1.0)
  - citation_coverage (deterministic, 0.0–1.0)

Composite formula:
  composite_score = coherence * 0.4 + factuality * 0.3 + citation_coverage * 0.3

Pass threshold: composite_score >= 0.90
If not passed, deficiencies lists specific gap descriptions.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Union

from autogen import AssistantAgent

from swarmiq.core.models import AgentError, EvaluatorOutput, Reference, ScoredClaim
from swarmiq.core.schemas import EVALUATOR_OUTPUT_SCHEMA
from swarmiq.core.validation import SchemaValidationError, validate_message
from swarmiq.utils.rate_limiter import groq_limiter

logger = logging.getLogger(__name__)

_PASS_THRESHOLD = 0.90

_EVALUATOR_SYSTEM_MESSAGE = """\
You are a research quality evaluator. You will be given a research report in Markdown \
and a list of claims. Your task is to score the report on two dimensions:

1. coherence (0.0–1.0): How logically structured, clear, and well-organised is the report?
   - 1.0 = perfectly coherent, flows naturally, no contradictions
   - 0.0 = incoherent, disorganised, contradictory

2. factuality (0.0–1.0): How well do the report's statements align with the provided claims?
   - 1.0 = every statement is directly supported by a claim
   - 0.0 = statements are unsupported or contradict the claims

Return ONLY a JSON object with exactly these two fields:
{"coherence": <float>, "factuality": <float>}

Do not include any other text, markdown fences, or explanation.
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _compute_citation_coverage(
    report_markdown: str,
    claims: list[ScoredClaim],
) -> float:
    """Deterministically compute citation coverage.

    Counts how many claims have their source_url appearing in the
    report_markdown, divided by total claims.

    Returns 1.0 if there are no claims (vacuously covered).
    """
    if not claims:
        return 1.0

    covered = sum(1 for c in claims if c.source_url in report_markdown)
    return covered / len(claims)


def _build_deficiencies(
    coherence: float,
    factuality: float,
    citation_coverage: float,
    composite_score: float,
) -> list[str]:
    """Produce specific gap descriptions when the report does not pass."""
    gaps: list[str] = []

    if coherence < 0.80:
        gaps.append(
            f"Coherence score {coherence:.2f} is below acceptable threshold (0.80): "
            "the report lacks logical structure or contains contradictions."
        )
    if factuality < 0.80:
        gaps.append(
            f"Factuality score {factuality:.2f} is below acceptable threshold (0.80): "
            "report statements are insufficiently supported by the provided claims."
        )
    if citation_coverage < 0.80:
        gaps.append(
            f"Citation coverage {citation_coverage:.2f} is below acceptable threshold (0.80): "
            "not all accepted claim sources are cited in the report."
        )
    if not gaps:
        # Composite failed but individual scores look OK — composite is just below threshold
        gaps.append(
            f"Composite score {composite_score:.4f} is below the pass threshold of "
            f"{_PASS_THRESHOLD}: marginal improvements across coherence, factuality, "
            "or citation coverage are required."
        )
    return gaps


class EvaluatorAgent:
    """Score the final research report and emit pass/fail.

    Args:
        llm_config: AutoGen LLM config dict.
    """

    def __init__(self, llm_config: dict) -> None:
        self._llm_config = llm_config
        self._agent = AssistantAgent(
            name="evaluator_agent",
            system_message=_EVALUATOR_SYSTEM_MESSAGE,
            llm_config=llm_config,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def evaluate(
        self,
        report_markdown: str,
        claims: list[ScoredClaim],
        references: list[Reference],
    ) -> Union[EvaluatorOutput, AgentError]:
        """Evaluate the report and return structured quality scores.

        Args:
            report_markdown: The synthesised Markdown report.
            claims:          ScoredClaim objects used for citation coverage.
            references:      Reference list (used for context in the prompt).

        Returns:
            EvaluatorOutput with all scores and pass/fail status.
            AgentError on unexpected failure.
        """
        try:
            return await self._run_evaluation(report_markdown, claims, references)
        except Exception as exc:  # noqa: BLE001
            logger.exception("EvaluatorAgent: unexpected error: %s", exc)
            return AgentError(
                agent_type="evaluator_agent",
                subtask_id=None,
                error_code="UNHANDLED_EXCEPTION",
                message=str(exc),
                timestamp=_now_iso(),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _run_evaluation(
        self,
        report_markdown: str,
        claims: list[ScoredClaim],
        references: list[Reference],
    ) -> EvaluatorOutput:
        """Core evaluation logic."""
        # 1. Ask the LLM to score coherence and factuality
        coherence, factuality = await self._llm_score(report_markdown, claims)

        # 2. Deterministically compute citation_coverage
        citation_coverage = _compute_citation_coverage(report_markdown, claims)

        # 3. Compute composite score
        composite_score = (
            coherence * 0.4 + factuality * 0.3 + citation_coverage * 0.3
        )
        # Clamp to [0.0, 1.0] to guard against floating-point drift
        composite_score = max(0.0, min(1.0, composite_score))

        # 4. Pass/fail
        passed = composite_score >= _PASS_THRESHOLD

        # 5. Deficiencies
        deficiencies: list[str] = []
        if not passed:
            deficiencies = _build_deficiencies(
                coherence, factuality, citation_coverage, composite_score
            )

        # 6. Build output dict and validate against schema
        output_dict = {
            "coherence": coherence,
            "factuality": factuality,
            "citation_coverage": citation_coverage,
            "composite_score": composite_score,
            "passed": passed,
            "deficiencies": deficiencies,
        }
        try:
            validate_message(output_dict, EVALUATOR_OUTPUT_SCHEMA)
        except SchemaValidationError as exc:
            logger.error("EvaluatorAgent: output schema validation failed: %s", exc)
            raise

        return EvaluatorOutput(
            coherence=coherence,
            factuality=factuality,
            citation_coverage=citation_coverage,
            composite_score=composite_score,
            passed=passed,
            deficiencies=deficiencies,
        )

    async def _llm_score(
        self,
        report_markdown: str,
        claims: list[ScoredClaim],
    ) -> tuple[float, float]:
        """Ask the LLM to return coherence and factuality scores.

        Falls back to conservative defaults (0.5) if the LLM call fails or
        returns unparseable output.
        """
        claims_summary = "\n".join(
            f"- {c.claim_text} (source: {c.source_url})"
            for c in claims[:50]  # cap to avoid token overflow
        )
        prompt = (
            f"## Report\n\n{report_markdown[:6000]}\n\n"
            f"## Claims\n\n{claims_summary}\n\n"
            "Score the report on coherence and factuality as instructed."
        )

        try:
            groq_limiter.wait_if_needed()
            reply = await self._agent.a_generate_reply(
                messages=[{"role": "user", "content": prompt}]
            )
            if isinstance(reply, dict):
                text = reply.get("content", "")
            else:
                text = str(reply) if reply is not None else ""

            return self._parse_llm_scores(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "EvaluatorAgent: LLM scoring failed: %s. Using fallback scores.", exc
            )
            return 0.5, 0.5

    @staticmethod
    def _parse_llm_scores(text: str) -> tuple[float, float]:
        """Parse coherence and factuality from the LLM JSON response.

        Returns (0.5, 0.5) as a safe fallback on any parse error.
        """
        text = text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        try:
            data = json.loads(text)
            coherence = float(data.get("coherence", 0.5))
            factuality = float(data.get("factuality", 0.5))
            # Clamp to [0.0, 1.0]
            coherence = max(0.0, min(1.0, coherence))
            factuality = max(0.0, min(1.0, factuality))
            return coherence, factuality
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning(
                "EvaluatorAgent: could not parse LLM scores from %r: %s", text, exc
            )
            return 0.5, 0.5
