"""
Summarizer_Agent for SwarmIQ v2.

Wraps an AutoGen AssistantAgent to extract discrete factual claims with
confidence scores from a list of retrieved documents.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Union

from autogen import AssistantAgent

from swarmiq.core.models import AgentError, Claim, Document, SummarizerOutput
from swarmiq.core.schemas import CLAIM_SCHEMA
from swarmiq.core.validation import SchemaValidationError, validate_message
from swarmiq.utils.rate_limiter import groq_limiter

logger = logging.getLogger(__name__)

_SYSTEM_MESSAGE = """\
You are a research claim extraction assistant. Your sole job is to read a set of \
documents and extract discrete, verifiable factual claims.

Rules:
- Return ONLY a valid JSON array — no markdown fences, no explanation, no extra text.
- Each element must be a JSON object with exactly these fields:
  - "claim_id": a UUID v4 string
  - "claim_text": a non-empty string stating a single factual claim
  - "confidence": a number between 0.0 and 1.0 reflecting how well-supported the claim is
  - "source_url": the URL of the document the claim was extracted from (must be a valid URI)
  - "subtask_id": the UUID of the subtask provided in the input
- Extract only claims that are directly supported by the document content.
- Assign higher confidence (closer to 1.0) to claims backed by strong evidence.
- Assign lower confidence (closer to 0.0) to speculative or weakly supported claims.
- Do not include any other keys.
"""


def _build_user_prompt(subtask_id: str, documents: list[Document]) -> str:
    doc_sections = []
    for i, doc in enumerate(documents, start=1):
        doc_sections.append(
            f"Document {i}:\n"
            f"  URL: {doc.url}\n"
            f"  Title: {doc.title}\n"
            f"  Content: {doc.content}\n"
        )
    docs_text = "\n".join(doc_sections)
    return (
        f"Subtask ID: {subtask_id}\n\n"
        f"Documents:\n{docs_text}\n\n"
        "Extract all discrete factual claims from the documents above. "
        "Return ONLY the JSON array of claim objects."
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SummarizerAgent:
    """Wraps an AutoGen AssistantAgent to extract claims from documents."""

    def __init__(self, llm_config: dict) -> None:
        self._llm_config = llm_config
        self._agent = AssistantAgent(
            name="summarizer",
            system_message=_SYSTEM_MESSAGE,
            llm_config=llm_config,
        )

    async def summarize(
        self,
        subtask_id: str,
        documents: list[Document],
    ) -> Union[SummarizerOutput, AgentError]:
        """Extract claims from *documents* for the given *subtask_id*.

        Args:
            subtask_id: UUID of the subtask these documents belong to.
            documents:  List of Document objects to extract claims from.

        Returns:
            SummarizerOutput with a list of validated Claim objects on success.
            SummarizerOutput(claims=[]) if documents is empty (logs a warning).
            AgentError on JSON parse failure or schema validation error.
        """
        if not documents:
            logger.warning(
                "SummarizerAgent: received empty document list for subtask %s. "
                "Returning empty claims.",
                subtask_id,
            )
            return SummarizerOutput(claims=[])

        try:
            return await self._call_llm(subtask_id, documents)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "SummarizerAgent encountered an unexpected error for subtask %s: %s",
                subtask_id,
                exc,
            )
            return AgentError(
                agent_type="summarizer_agent",
                subtask_id=subtask_id,
                error_code="UNHANDLED_EXCEPTION",
                message=str(exc),
                timestamp=_now_iso(),
            )

    async def _call_llm(
        self, subtask_id: str, documents: list[Document]
    ) -> Union[SummarizerOutput, AgentError]:
        """Send the prompt to the LLM and parse/validate the response."""
        messages = [
            {"role": "user", "content": _build_user_prompt(subtask_id, documents)}
        ]

        groq_limiter.wait_if_needed()
        reply = await self._agent.a_generate_reply(messages=messages)

        if isinstance(reply, dict):
            raw_text: str = reply.get("content", "")
        else:
            raw_text = str(reply) if reply is not None else ""

        return self._parse_and_validate(raw_text, subtask_id)

    def _parse_and_validate(
        self, raw_text: str, subtask_id: str
    ) -> Union[SummarizerOutput, AgentError]:
        """Parse the LLM's raw text as JSON and validate each claim."""
        text = raw_text.strip()
        # Strip accidental markdown fences
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error(
                "SummarizerAgent: failed to parse LLM response as JSON for subtask %s: %s",
                subtask_id,
                exc,
            )
            return AgentError(
                agent_type="summarizer_agent",
                subtask_id=subtask_id,
                error_code="INVALID_JSON",
                message=f"LLM returned non-JSON output: {exc}",
                timestamp=_now_iso(),
            )

        if not isinstance(data, list):
            return AgentError(
                agent_type="summarizer_agent",
                subtask_id=subtask_id,
                error_code="INVALID_FORMAT",
                message="Expected a JSON array of claims; got a non-list value.",
                timestamp=_now_iso(),
            )

        claims: list[Claim] = []
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                return AgentError(
                    agent_type="summarizer_agent",
                    subtask_id=subtask_id,
                    error_code="INVALID_FORMAT",
                    message=f"Claim at index {i} is not a JSON object.",
                    timestamp=_now_iso(),
                )

            # Ensure claim_id is present; generate one if missing
            if not item.get("claim_id"):
                item["claim_id"] = str(uuid.uuid4())

            # Ensure subtask_id is set correctly
            if not item.get("subtask_id"):
                item["subtask_id"] = subtask_id

            try:
                validate_message(item, CLAIM_SCHEMA)
            except SchemaValidationError as exc:
                logger.error(
                    "SummarizerAgent: claim %d failed schema validation for subtask %s: %s",
                    i,
                    subtask_id,
                    exc,
                )
                return AgentError(
                    agent_type="summarizer_agent",
                    subtask_id=subtask_id,
                    error_code="SCHEMA_VALIDATION_ERROR",
                    message=f"Claim {i} failed schema validation: {exc}",
                    timestamp=_now_iso(),
                )

            claims.append(
                Claim(
                    claim_id=item["claim_id"],
                    claim_text=item["claim_text"],
                    confidence=item["confidence"],
                    source_url=item["source_url"],
                    subtask_id=item["subtask_id"],
                )
            )

        return SummarizerOutput(claims=claims)
