"""
GapDetector agent for SwarmIQ v2.

Given the original research plan questions and retrieved content so far,
identifies which questions remain inadequately answered.
"""
from __future__ import annotations

import asyncio
import logging

from autogen import AssistantAgent

logger = logging.getLogger(__name__)

_SYSTEM_MESSAGE = """\
You are a research gap analysis assistant. Your job is to identify which research questions \
have NOT been adequately answered by the retrieved content.

Rules:
- Return ONLY a newline-separated list of unanswered questions — no numbering, no explanation.
- If all questions are answered, return exactly: NONE
- Do not include questions that are clearly addressed in the content.
- Be concise; return only the question text itself.
"""

_CONTENT_TRUNCATION = 4000


def _build_prompt(research_plan: list[str], retrieved_content: str, max_gaps: int) -> str:
    truncated = retrieved_content[:_CONTENT_TRUNCATION]
    questions = "\n".join(f"- {q}" for q in research_plan)
    return (
        f"Research questions:\n{questions}\n\n"
        f"Retrieved content so far:\n{truncated}\n\n"
        f"Which of the above questions are NOT adequately answered by the content? "
        f"Return at most {max_gaps} unanswered questions, one per line. "
        f"If all are answered, return exactly: NONE"
    )


class GapDetector:
    """Uses an LLM to identify unanswered questions from a research plan."""

    def __init__(self, llm_config: dict) -> None:
        self._llm_config = llm_config
        self._agent = AssistantAgent(
            name="gap_detector",
            system_message=_SYSTEM_MESSAGE,
            llm_config=llm_config,
        )

    async def detect_gaps(
        self,
        research_plan: list[str],
        retrieved_content: str,
        max_gaps: int = 3,
    ) -> list[str]:
        """Identify unanswered questions from the research plan.

        Args:
            research_plan:      List of research question strings.
            retrieved_content:  Combined text of retrieved documents so far.
            max_gaps:           Maximum number of gaps to return.

        Returns:
            List of unanswered question strings (up to max_gaps), or empty list
            if all questions are answered or the LLM responds with "NONE".
        """
        if not research_plan:
            return []

        prompt = _build_prompt(research_plan, retrieved_content, max_gaps)
        messages = [{"role": "user", "content": prompt}]

        try:
            reply = await self._agent.a_generate_reply(messages=messages)
        except Exception as exc:
            logger.warning("GapDetector: LLM call failed: %s", exc)
            return []

        # Normalise reply to string
        if isinstance(reply, dict):
            raw: str = reply.get("content", "")
        else:
            raw = str(reply) if reply is not None else ""

        raw = raw.strip()
        if not raw or raw.upper() == "NONE":
            return []

        gaps = [line.strip() for line in raw.splitlines() if line.strip()]
        return gaps[:max_gaps]
