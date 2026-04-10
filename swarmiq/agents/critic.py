"""
CriticAgent for SwarmIQ v2.

Reviews the synthesized report against the original research questions and
source list, then requests targeted revisions if specific issues are found.

Performs at most MAX_CRITIQUE_REVISIONS revision cycles (default: 2).
Returns a dict with keys: report (str), revisions (int), passed (bool).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from autogen import AssistantAgent

from swarmiq.utils.rate_limiter import groq_limiter

logger = logging.getLogger(__name__)

_FIND_ISSUES_SYSTEM_MESSAGE = """\
You are a research report critic. You will be given a research report, a list of \
research questions that should be answered, and a list of source URLs that should \
be cited.

Your task is to identify SPECIFIC issues in the report. Check for:
1. Research questions that are not answered in the report.
2. Claims or statements that are not backed by a citation (e.g. [1], [2]).
3. Missing or absent References section.

Return ONLY a JSON array of specific issue strings. Each issue must be a concrete, \
actionable description. If there are no issues, return an empty array: []

Example output:
["Question 'What are the economic impacts?' is not addressed in the report.",
 "The claim about market growth in section 2 has no citation.",
 "References section is missing."]

Do not include any other text, markdown fences, or explanation outside the JSON array.
"""

_REVISE_SYSTEM_MESSAGE = """\
You are a research report editor. You will be given a research report and a list of \
specific issues that need to be fixed.

Your task is to revise the report to address ONLY the listed issues. Do not rewrite \
sections that are already correct. Preserve all existing citations, structure, and \
content that is not related to the issues.

Return ONLY the complete revised report in Markdown format, with no additional \
commentary or explanation.
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CriticAgent:
    """Review and iteratively revise the synthesized report.

    Args:
        llm_config: AutoGen LLM config dict.
    """

    def __init__(self, llm_config: dict) -> None:
        self._llm_config = llm_config
        self._find_issues_agent = AssistantAgent(
            name="critic_find_issues",
            system_message=_FIND_ISSUES_SYSTEM_MESSAGE,
            llm_config=llm_config,
        )
        self._revise_agent = AssistantAgent(
            name="critic_revise",
            system_message=_REVISE_SYSTEM_MESSAGE,
            llm_config=llm_config,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def critique(
        self,
        report: str,
        research_questions: list[str],
        sources: list[str],
        max_revisions: int = 2,
    ) -> dict:
        """Review the report and revise it up to max_revisions times.

        Args:
            report:             The synthesized Markdown report.
            research_questions: List of research question strings to verify coverage.
            sources:            List of source URLs that should be cited.
            max_revisions:      Maximum number of revision cycles (default: 2).

        Returns:
            dict with keys:
              - report (str):    The final (possibly revised) report.
              - revisions (int): Number of revision cycles performed.
              - passed (bool):   True if no issues were found before max_revisions.
        """
        revisions = 0
        current_report = report

        for _ in range(max_revisions):
            issues = await self._find_issues(current_report, research_questions, sources)

            if not issues:
                return {"report": current_report, "revisions": revisions, "passed": True}

            current_report = await self._revise(current_report, issues)
            revisions += 1

        return {"report": current_report, "revisions": revisions, "passed": False}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _find_issues(
        self,
        report: str,
        questions: list[str],
        sources: list[str],
    ) -> list[str]:
        """Use an LLM to identify specific issues in the report.

        Checks: every question answered, every claim cited, references present.

        Returns a list of specific issue strings, or empty list if all good.
        """
        truncated_report = report[:5000]
        questions_text = "\n".join(f"- {q}" for q in questions) if questions else "(none)"
        sources_text = "\n".join(f"- {s}" for s in sources) if sources else "(none)"

        prompt = (
            f"## Report (truncated to 5000 chars)\n\n{truncated_report}\n\n"
            f"## Research Questions\n\n{questions_text}\n\n"
            f"## Expected Source URLs\n\n{sources_text}\n\n"
            "Identify all specific issues in the report as described in your instructions."
        )

        try:
            groq_limiter.wait_if_needed()
            reply = await self._find_issues_agent.a_generate_reply(
                messages=[{"role": "user", "content": prompt}]
            )
            if isinstance(reply, dict):
                text = reply.get("content", "")
            else:
                text = str(reply) if reply is not None else ""

            return self._parse_issues(text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CriticAgent._find_issues: LLM call failed: %s", exc)
            return []

    async def _revise(self, report: str, issues: list[str]) -> str:
        """Use an LLM to fix specific issues without rewriting the whole report.

        Returns the revised report string.
        """
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        prompt = (
            f"## Issues to Fix\n\n{issues_text}\n\n"
            f"## Current Report\n\n{report}\n\n"
            "Revise the report to address only the listed issues."
        )

        try:
            groq_limiter.wait_if_needed()
            reply = await self._revise_agent.a_generate_reply(
                messages=[{"role": "user", "content": prompt}]
            )
            if isinstance(reply, dict):
                text = reply.get("content", "")
            else:
                text = str(reply) if reply is not None else ""

            revised = text.strip()
            return revised if revised else report
        except Exception as exc:  # noqa: BLE001
            logger.warning("CriticAgent._revise: LLM call failed: %s. Returning original.", exc)
            return report

    @staticmethod
    def _parse_issues(text: str) -> list[str]:
        """Parse the JSON array of issue strings from the LLM response.

        Returns an empty list on any parse error (treat as no issues found).
        """
        import json

        text = text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(item) for item in data if item]
            return []
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("CriticAgent: could not parse issues from %r: %s", text[:200], exc)
            return []
