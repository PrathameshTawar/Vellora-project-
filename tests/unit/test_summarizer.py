"""
Unit tests for SummarizerAgent.

Tests:
1.  Empty document list → returns SummarizerOutput(claims=[]) without calling LLM
2.  Valid documents + LLM returns 2 valid claims → SummarizerOutput with 2 Claim objects
3.  LLM returns non-JSON → AgentError with error_code="INVALID_JSON"
4.  LLM returns JSON array with claim missing `claim_text` → AgentError with error_code="SCHEMA_VALIDATION_ERROR"
5.  LLM returns claim with confidence=1.5 (out of range) → AgentError with error_code="SCHEMA_VALIDATION_ERROR"
6.  LLM returns claim without claim_id → auto-generates UUID (result is SummarizerOutput)
7.  LLM returns claim without subtask_id → fills in the provided subtask_id (result is SummarizerOutput)
8.  LLM returns empty array → SummarizerOutput(claims=[])
9.  Each claim in valid output has all required fields (claim_id, claim_text, confidence, source_url, subtask_id)
10. LLM returns non-list JSON (dict) → AgentError with error_code="INVALID_FORMAT"

Requirements: 3.1, 3.4, 3.5
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

# ---------------------------------------------------------------------------
# Stub out autogen before any swarmiq imports
# ---------------------------------------------------------------------------
_autogen_mod = types.ModuleType("autogen")
_autogen_mod.AssistantAgent = MagicMock  # type: ignore[attr-defined]
sys.modules.setdefault("autogen", _autogen_mod)

# ---------------------------------------------------------------------------
# Now safe to import swarmiq
# ---------------------------------------------------------------------------
from swarmiq.agents.summarizer import SummarizerAgent  # noqa: E402
from swarmiq.core.models import AgentError, Claim, Document, SummarizerOutput  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SUBTASK_ID = str(uuid.uuid4())
_CLAIM_ID = str(uuid.uuid4())
_SOURCE_URL = "https://example.com/paper"


def _make_doc(i: int = 0) -> Document:
    return Document(
        url=f"https://example.com/doc{i}",
        title=f"Doc {i}",
        content=f"Content of document {i}.",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        subtask_id=_SUBTASK_ID,
    )


def _make_docs(n: int) -> list[Document]:
    return [_make_doc(i) for i in range(n)]


def _make_agent() -> SummarizerAgent:
    return SummarizerAgent(llm_config={"model": "gpt-4o", "api_key": "fake"})


def _valid_claim_dict(
    *,
    claim_id: str | None = None,
    claim_text: str = "The sky is blue.",
    confidence: float = 0.9,
    source_url: str = _SOURCE_URL,
    subtask_id: str | None = None,
) -> dict:
    d: dict = {
        "claim_text": claim_text,
        "confidence": confidence,
        "source_url": source_url,
    }
    if claim_id is not None:
        d["claim_id"] = claim_id
    if subtask_id is not None:
        d["subtask_id"] = subtask_id
    return d


# ---------------------------------------------------------------------------
# Test 1: Empty document list → SummarizerOutput(claims=[]) without calling LLM
# ---------------------------------------------------------------------------


def test_empty_documents_returns_empty_output_without_llm():
    """Empty document list → SummarizerOutput(claims=[]) without calling LLM."""
    agent = _make_agent()

    with patch.object(agent, "_call_llm", new_callable=AsyncMock) as mock_llm:
        result = asyncio.run(agent.summarize(_SUBTASK_ID, []))

    assert isinstance(result, SummarizerOutput)
    assert result.claims == []
    mock_llm.assert_not_called()


# ---------------------------------------------------------------------------
# Test 2: Valid documents + LLM returns 2 valid claims → SummarizerOutput with 2 Claim objects
# ---------------------------------------------------------------------------


def test_valid_documents_returns_two_claims():
    """Valid documents + LLM returns 2 valid claims → SummarizerOutput with 2 Claim objects."""
    agent = _make_agent()
    docs = _make_docs(2)

    claims_payload = [
        _valid_claim_dict(
            claim_id=str(uuid.uuid4()),
            claim_text="Claim one.",
            confidence=0.8,
            source_url="https://example.com/doc0",
            subtask_id=_SUBTASK_ID,
        ),
        _valid_claim_dict(
            claim_id=str(uuid.uuid4()),
            claim_text="Claim two.",
            confidence=0.7,
            source_url="https://example.com/doc1",
            subtask_id=_SUBTASK_ID,
        ),
    ]

    async def _fake_call_llm(subtask_id, documents):
        return agent._parse_and_validate(json.dumps(claims_payload), subtask_id)

    with patch.object(agent, "_call_llm", side_effect=_fake_call_llm):
        result = asyncio.run(agent.summarize(_SUBTASK_ID, docs))

    assert isinstance(result, SummarizerOutput)
    assert len(result.claims) == 2
    assert all(isinstance(c, Claim) for c in result.claims)


# ---------------------------------------------------------------------------
# Test 3: LLM returns non-JSON → AgentError with error_code="INVALID_JSON"
# ---------------------------------------------------------------------------


def test_non_json_response_returns_invalid_json_error():
    """LLM returns non-JSON → AgentError with error_code='INVALID_JSON'."""
    agent = _make_agent()
    docs = _make_docs(1)

    async def _fake_call_llm(subtask_id, documents):
        return agent._parse_and_validate("This is not JSON at all!", subtask_id)

    with patch.object(agent, "_call_llm", side_effect=_fake_call_llm):
        result = asyncio.run(agent.summarize(_SUBTASK_ID, docs))

    assert isinstance(result, AgentError)
    assert result.error_code == "INVALID_JSON"


# ---------------------------------------------------------------------------
# Test 4: LLM returns JSON array with claim missing `claim_text` → SCHEMA_VALIDATION_ERROR
# ---------------------------------------------------------------------------


def test_claim_missing_claim_text_returns_schema_error():
    """LLM returns JSON array with claim missing claim_text → AgentError SCHEMA_VALIDATION_ERROR."""
    agent = _make_agent()
    docs = _make_docs(1)

    bad_claim = {
        "claim_id": str(uuid.uuid4()),
        # claim_text intentionally omitted
        "confidence": 0.8,
        "source_url": "https://example.com/doc0",
        "subtask_id": _SUBTASK_ID,
    }

    async def _fake_call_llm(subtask_id, documents):
        return agent._parse_and_validate(json.dumps([bad_claim]), subtask_id)

    with patch.object(agent, "_call_llm", side_effect=_fake_call_llm):
        result = asyncio.run(agent.summarize(_SUBTASK_ID, docs))

    assert isinstance(result, AgentError)
    assert result.error_code == "SCHEMA_VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# Test 5: LLM returns claim with confidence=1.5 (out of range) → SCHEMA_VALIDATION_ERROR
# ---------------------------------------------------------------------------


def test_confidence_out_of_range_returns_schema_error():
    """LLM returns claim with confidence=1.5 → AgentError with error_code='SCHEMA_VALIDATION_ERROR'."""
    agent = _make_agent()
    docs = _make_docs(1)

    bad_claim = {
        "claim_id": str(uuid.uuid4()),
        "claim_text": "Some claim.",
        "confidence": 1.5,  # out of [0.0, 1.0]
        "source_url": "https://example.com/doc0",
        "subtask_id": _SUBTASK_ID,
    }

    async def _fake_call_llm(subtask_id, documents):
        return agent._parse_and_validate(json.dumps([bad_claim]), subtask_id)

    with patch.object(agent, "_call_llm", side_effect=_fake_call_llm):
        result = asyncio.run(agent.summarize(_SUBTASK_ID, docs))

    assert isinstance(result, AgentError)
    assert result.error_code == "SCHEMA_VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# Test 6: LLM returns claim without claim_id → auto-generates UUID
# ---------------------------------------------------------------------------


def test_missing_claim_id_is_auto_generated():
    """LLM returns claim without claim_id → auto-generates UUID (result is SummarizerOutput)."""
    agent = _make_agent()
    docs = _make_docs(1)

    claim_no_id = {
        # claim_id omitted
        "claim_text": "Auto-ID claim.",
        "confidence": 0.75,
        "source_url": "https://example.com/doc0",
        "subtask_id": _SUBTASK_ID,
    }

    async def _fake_call_llm(subtask_id, documents):
        return agent._parse_and_validate(json.dumps([claim_no_id]), subtask_id)

    with patch.object(agent, "_call_llm", side_effect=_fake_call_llm):
        result = asyncio.run(agent.summarize(_SUBTASK_ID, docs))

    assert isinstance(result, SummarizerOutput)
    assert len(result.claims) == 1
    # claim_id should be a valid UUID string
    generated_id = result.claims[0].claim_id
    uuid.UUID(generated_id)  # raises ValueError if not a valid UUID


# ---------------------------------------------------------------------------
# Test 7: LLM returns claim without subtask_id → fills in the provided subtask_id
# ---------------------------------------------------------------------------


def test_missing_subtask_id_is_filled_in():
    """LLM returns claim without subtask_id → fills in the provided subtask_id."""
    agent = _make_agent()
    docs = _make_docs(1)

    claim_no_subtask = {
        "claim_id": str(uuid.uuid4()),
        "claim_text": "Claim without subtask_id.",
        "confidence": 0.6,
        "source_url": "https://example.com/doc0",
        # subtask_id omitted
    }

    async def _fake_call_llm(subtask_id, documents):
        return agent._parse_and_validate(json.dumps([claim_no_subtask]), subtask_id)

    with patch.object(agent, "_call_llm", side_effect=_fake_call_llm):
        result = asyncio.run(agent.summarize(_SUBTASK_ID, docs))

    assert isinstance(result, SummarizerOutput)
    assert len(result.claims) == 1
    assert result.claims[0].subtask_id == _SUBTASK_ID


# ---------------------------------------------------------------------------
# Test 8: LLM returns empty array → SummarizerOutput(claims=[])
# ---------------------------------------------------------------------------


def test_empty_array_response_returns_empty_output():
    """LLM returns empty array → SummarizerOutput(claims=[])."""
    agent = _make_agent()
    docs = _make_docs(1)

    async def _fake_call_llm(subtask_id, documents):
        return agent._parse_and_validate("[]", subtask_id)

    with patch.object(agent, "_call_llm", side_effect=_fake_call_llm):
        result = asyncio.run(agent.summarize(_SUBTASK_ID, docs))

    assert isinstance(result, SummarizerOutput)
    assert result.claims == []


# ---------------------------------------------------------------------------
# Test 9: Each claim in valid output has all required fields
# ---------------------------------------------------------------------------


def test_valid_claims_have_all_required_fields():
    """Each claim in valid output has all required fields: claim_id, claim_text, confidence, source_url, subtask_id."""
    agent = _make_agent()
    docs = _make_docs(1)

    claim = {
        "claim_id": str(uuid.uuid4()),
        "claim_text": "Water is H2O.",
        "confidence": 0.99,
        "source_url": "https://example.com/doc0",
        "subtask_id": _SUBTASK_ID,
    }

    async def _fake_call_llm(subtask_id, documents):
        return agent._parse_and_validate(json.dumps([claim]), subtask_id)

    with patch.object(agent, "_call_llm", side_effect=_fake_call_llm):
        result = asyncio.run(agent.summarize(_SUBTASK_ID, docs))

    assert isinstance(result, SummarizerOutput)
    assert len(result.claims) == 1
    c = result.claims[0]
    assert c.claim_id == claim["claim_id"]
    assert c.claim_text == claim["claim_text"]
    assert c.confidence == claim["confidence"]
    assert c.source_url == claim["source_url"]
    assert c.subtask_id == claim["subtask_id"]


# ---------------------------------------------------------------------------
# Test 10: LLM returns non-list JSON (dict) → AgentError with error_code="INVALID_FORMAT"
# ---------------------------------------------------------------------------


def test_non_list_json_returns_invalid_format_error():
    """LLM returns non-list JSON (dict) → AgentError with error_code='INVALID_FORMAT'."""
    agent = _make_agent()
    docs = _make_docs(1)

    non_list_payload = {"claim_text": "This is a dict, not a list."}

    async def _fake_call_llm(subtask_id, documents):
        return agent._parse_and_validate(json.dumps(non_list_payload), subtask_id)

    with patch.object(agent, "_call_llm", side_effect=_fake_call_llm):
        result = asyncio.run(agent.summarize(_SUBTASK_ID, docs))

    assert isinstance(result, AgentError)
    assert result.error_code == "INVALID_FORMAT"
