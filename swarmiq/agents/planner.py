"""
Planner_Agent for SwarmIQ v2.

Wraps an AutoGen AssistantAgent to decompose a user query into 3–5 structured subtasks.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Union

from autogen import AssistantAgent

from swarmiq.core.models import AgentError, PlannerOutput, SubTask
from swarmiq.core.schemas import SUBTASK_SCHEMA
from swarmiq.core.validation import SchemaValidationError, validate_message
from swarmiq.utils.rate_limiter import groq_limiter

logger = logging.getLogger(__name__)

_VALID_TYPES = {
    "literature_review",
    "summarization",
    "visualization",
    "conflict_resolution",
    "synthesis",
}

_SYSTEM_MESSAGE = """\
You are a research planning assistant. Your sole job is to decompose a user research query \
into 3 to 5 structured subtasks.

Rules:
- Return ONLY a valid JSON array — no markdown fences, no explanation, no extra text.
- Each element must be a JSON object with exactly these fields:
  - "subtask_id": a UUID v4 string
  - "type": one of "literature_review", "summarization", "visualization", \
"conflict_resolution", "synthesis"
  - "description": a non-empty string describing the subtask
  - "search_keywords": a non-empty array of relevant search keyword strings
- The array must contain between 3 and 5 objects.
- Do not include any other keys.
"""


def _build_user_prompt(query: str, domain_mode: str) -> str:
    return (
        f"Domain mode: {domain_mode}\n"
        f"Research query: {query}\n\n"
        "Decompose this query into 3–5 subtasks. Return ONLY the JSON array."
    )


class PlannerAgent:
    """Wraps an AutoGen AssistantAgent to decompose queries into subtasks."""

    def __init__(self, llm_config: dict) -> None:
        self._llm_config = llm_config
        self._agent = AssistantAgent(
            name="planner",
            system_message=_SYSTEM_MESSAGE,
            llm_config=llm_config,
        )

    async def decompose(
        self, query: str, domain_mode: str
    ) -> Union[PlannerOutput, AgentError, dict]:
        """Decompose *query* into 3–5 subtasks.

        Args:
            query:       The user's research query.
            domain_mode: One of "research", "business", or "policy".

        Returns:
            PlannerOutput on success.
            dict {"error": "INVALID_QUERY", "message": "..."} for short/empty queries.
            AgentError on timeout or unexpected failure.
        """
        # ── Input validation ────────────────────────────────────────────────
        if not query or not query.strip() or len(query.strip()) < 5:
            return {
                "error": "INVALID_QUERY",
                "message": (
                    "Query must be at least 5 non-whitespace characters. "
                    f"Received: {query!r}"
                ),
            }

        # ── LLM call with timeout ───────────────────────────────────────────
        try:
            result = await asyncio.wait_for(
                self._call_llm(query, domain_mode), timeout=10.0
            )
            return result
        except asyncio.TimeoutError:
            logger.error("PlannerAgent timed out after 10 seconds for query: %r", query)
            return AgentError(
                agent_type="planner_agent",
                error_code="TIMEOUT",
                message="Planner LLM call exceeded the 10-second timeout.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("PlannerAgent encountered an unexpected error: %s", exc)
            return AgentError(
                agent_type="planner_agent",
                error_code="UNHANDLED_EXCEPTION",
                message=str(exc),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

    async def _call_llm(self, query: str, domain_mode: str) -> Union[PlannerOutput, AgentError]:
        """Send the prompt to the LLM and parse/validate the response."""
        messages = [{"role": "user", "content": _build_user_prompt(query, domain_mode)}]

        groq_limiter.wait_if_needed()
        reply = await self._agent.a_generate_reply(messages=messages)

        # AutoGen may return a string or a dict; normalise to string
        if isinstance(reply, dict):
            raw_text: str = reply.get("content", "")
        else:
            raw_text = str(reply) if reply is not None else ""

        return self._parse_and_validate(raw_text)

    def _parse_and_validate(self, raw_text: str) -> Union[PlannerOutput, AgentError]:
        """Parse the LLM's raw text as JSON and validate each subtask."""
        # Strip accidental markdown fences
        text = raw_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # Drop first and last fence lines
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("PlannerAgent: failed to parse LLM response as JSON: %s", exc)
            return AgentError(
                agent_type="planner_agent",
                error_code="INVALID_JSON",
                message=f"LLM returned non-JSON output: {exc}",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        if not isinstance(data, list):
            return AgentError(
                agent_type="planner_agent",
                error_code="INVALID_FORMAT",
                message="Expected a JSON array of subtasks; got a non-list value.",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        subtasks: list[SubTask] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                return AgentError(
                    agent_type="planner_agent",
                    error_code="INVALID_FORMAT",
                    message=f"Subtask at index {i} is not a JSON object.",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

            # Ensure subtask_id is present; generate one if missing
            if not item.get("subtask_id"):
                item["subtask_id"] = str(uuid.uuid4())

            # Validate against SUBTASK_SCHEMA
            try:
                validate_message(item, SUBTASK_SCHEMA)
            except SchemaValidationError as exc:
                return AgentError(
                    agent_type="planner_agent",
                    error_code="SCHEMA_VALIDATION_ERROR",
                    message=f"Subtask {i} failed schema validation: {exc}",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )

            subtasks.append(
                SubTask(
                    subtask_id=item["subtask_id"],
                    type=item["type"],
                    description=item["description"],
                    search_keywords=item["search_keywords"],
                )
            )

        if not (3 <= len(subtasks) <= 5):
            return AgentError(
                agent_type="planner_agent",
                error_code="INVALID_COUNT",
                message=(
                    f"Expected 3–5 subtasks; LLM returned {len(subtasks)}."
                ),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        return PlannerOutput(subtasks=subtasks)
