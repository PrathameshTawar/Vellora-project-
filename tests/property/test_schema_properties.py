"""
Property-based tests for schema validation and round-trip serialization.

Feature: swarmiq-v2
Validates: Requirements 15.1, 15.2, 15.4
"""
from __future__ import annotations

import dataclasses
import json

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from swarmiq.core.models import (
    SubTask,
    Claim,
    ScoredClaim,
    Resolution,
    EvaluatorOutput,
)
from swarmiq.core.schemas import (
    SUBTASK_SCHEMA,
    CLAIM_SCHEMA,
    EVALUATOR_OUTPUT_SCHEMA,
    RESOLUTION_SCHEMA,
)
from swarmiq.core.validation import validate_message, SchemaValidationError

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

uuid_str = st.uuids().map(str)
nonempty_text = st.text(min_size=1)
score = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
keyword_list = st.lists(st.text(min_size=1), min_size=1)
subtask_type = st.sampled_from(
    ["literature_review", "summarization", "visualization", "conflict_resolution", "synthesis"]
)
resolution_status = st.sampled_from(["accepted", "rejected", "uncertain"])

# Valid URI strategy (jsonschema "uri" format requires a scheme)
valid_uri = st.builds(
    lambda uid: f"https://example.com/{uid}",
    uid=st.uuids().map(str),
)


@st.composite
def valid_subtask_dict(draw) -> dict:
    return {
        "subtask_id": draw(uuid_str),
        "type": draw(subtask_type),
        "description": draw(nonempty_text),
        "search_keywords": draw(keyword_list),
    }


@st.composite
def valid_claim_dict(draw) -> dict:
    return {
        "claim_id": draw(uuid_str),
        "claim_text": draw(nonempty_text),
        "confidence": draw(score),
        "source_url": draw(valid_uri),
        "subtask_id": draw(uuid_str),
    }


@st.composite
def valid_resolution_dict(draw) -> dict:
    return {
        "claim_id": draw(uuid_str),
        "status": draw(resolution_status),
        "rationale": draw(nonempty_text),
        "credibility_score": draw(score),
    }


@st.composite
def valid_evaluator_output_dict(draw) -> dict:
    return {
        "coherence": draw(score),
        "factuality": draw(score),
        "citation_coverage": draw(score),
        "composite_score": draw(score),
        "passed": draw(st.booleans()),
    }


# ---------------------------------------------------------------------------
# Property 29: Schema validation catches invalid payloads
# ---------------------------------------------------------------------------

class TestSchemaValidationCatchesInvalidPayloads:
    """
    # Feature: swarmiq-v2, Property 29: Schema validation catches invalid payloads —
    # for any agent message object that violates the defined JSON schema (missing required
    # fields, wrong types, out-of-range values), validate_message must raise
    # SchemaValidationError and must NOT pass silently.
    """

    # --- SubTask schema ---

    @given(valid_subtask_dict(), st.sampled_from(["subtask_id", "type", "description", "search_keywords"]))
    @settings(max_examples=100)
    def test_subtask_missing_required_field_raises(self, payload: dict, field: str):
        # Feature: swarmiq-v2, Property 29: Schema validation catches invalid payloads
        del payload[field]
        with pytest.raises(SchemaValidationError):
            validate_message(payload, SUBTASK_SCHEMA)

    @given(valid_subtask_dict())
    @settings(max_examples=100)
    def test_subtask_invalid_type_enum_raises(self, payload: dict):
        # Feature: swarmiq-v2, Property 29: Schema validation catches invalid payloads
        payload["type"] = "invalid_type_value"
        with pytest.raises(SchemaValidationError):
            validate_message(payload, SUBTASK_SCHEMA)

    @given(valid_subtask_dict())
    @settings(max_examples=100)
    def test_subtask_empty_keywords_raises(self, payload: dict):
        # Feature: swarmiq-v2, Property 29: Schema validation catches invalid payloads
        payload["search_keywords"] = []
        with pytest.raises(SchemaValidationError):
            validate_message(payload, SUBTASK_SCHEMA)

    # --- Claim schema ---

    @given(valid_claim_dict(), st.sampled_from(["claim_id", "claim_text", "confidence", "source_url", "subtask_id"]))
    @settings(max_examples=100)
    def test_claim_missing_required_field_raises(self, payload: dict, field: str):
        # Feature: swarmiq-v2, Property 29: Schema validation catches invalid payloads
        del payload[field]
        with pytest.raises(SchemaValidationError):
            validate_message(payload, CLAIM_SCHEMA)

    @given(valid_claim_dict(), st.floats(min_value=1.001, max_value=100.0, allow_nan=False))
    @settings(max_examples=100)
    def test_claim_confidence_above_range_raises(self, payload: dict, bad_score: float):
        # Feature: swarmiq-v2, Property 29: Schema validation catches invalid payloads
        payload["confidence"] = bad_score
        with pytest.raises(SchemaValidationError):
            validate_message(payload, CLAIM_SCHEMA)

    @given(valid_claim_dict(), st.floats(min_value=-100.0, max_value=-0.001, allow_nan=False))
    @settings(max_examples=100)
    def test_claim_confidence_below_range_raises(self, payload: dict, bad_score: float):
        # Feature: swarmiq-v2, Property 29: Schema validation catches invalid payloads
        payload["confidence"] = bad_score
        with pytest.raises(SchemaValidationError):
            validate_message(payload, CLAIM_SCHEMA)

    # --- Resolution schema ---

    @given(valid_resolution_dict(), st.sampled_from(["claim_id", "status", "rationale", "credibility_score"]))
    @settings(max_examples=100)
    def test_resolution_missing_required_field_raises(self, payload: dict, field: str):
        # Feature: swarmiq-v2, Property 29: Schema validation catches invalid payloads
        del payload[field]
        with pytest.raises(SchemaValidationError):
            validate_message(payload, RESOLUTION_SCHEMA)

    @given(valid_resolution_dict())
    @settings(max_examples=100)
    def test_resolution_invalid_status_enum_raises(self, payload: dict):
        # Feature: swarmiq-v2, Property 29: Schema validation catches invalid payloads
        payload["status"] = "invalid_status"
        with pytest.raises(SchemaValidationError):
            validate_message(payload, RESOLUTION_SCHEMA)

    @given(valid_resolution_dict(), st.floats(min_value=1.001, max_value=100.0, allow_nan=False))
    @settings(max_examples=100)
    def test_resolution_credibility_above_range_raises(self, payload: dict, bad_score: float):
        # Feature: swarmiq-v2, Property 29: Schema validation catches invalid payloads
        payload["credibility_score"] = bad_score
        with pytest.raises(SchemaValidationError):
            validate_message(payload, RESOLUTION_SCHEMA)

    # --- EvaluatorOutput schema ---

    @given(
        valid_evaluator_output_dict(),
        st.sampled_from(["coherence", "factuality", "citation_coverage", "composite_score", "passed"]),
    )
    @settings(max_examples=100)
    def test_evaluator_output_missing_required_field_raises(self, payload: dict, field: str):
        # Feature: swarmiq-v2, Property 29: Schema validation catches invalid payloads
        del payload[field]
        with pytest.raises(SchemaValidationError):
            validate_message(payload, EVALUATOR_OUTPUT_SCHEMA)

    @given(valid_evaluator_output_dict(), st.floats(min_value=1.001, max_value=100.0, allow_nan=False))
    @settings(max_examples=100)
    def test_evaluator_output_score_above_range_raises(self, payload: dict, bad_score: float):
        # Feature: swarmiq-v2, Property 29: Schema validation catches invalid payloads
        payload["coherence"] = bad_score
        with pytest.raises(SchemaValidationError):
            validate_message(payload, EVALUATOR_OUTPUT_SCHEMA)

    @given(valid_evaluator_output_dict())
    @settings(max_examples=100)
    def test_evaluator_output_passed_wrong_type_raises(self, payload: dict):
        # Feature: swarmiq-v2, Property 29: Schema validation catches invalid payloads
        payload["passed"] = "yes"  # string instead of boolean
        with pytest.raises(SchemaValidationError):
            validate_message(payload, EVALUATOR_OUTPUT_SCHEMA)


# ---------------------------------------------------------------------------
# Property 30: Agent message round-trip serialization
# ---------------------------------------------------------------------------

class TestAgentMessageRoundTripSerialization:
    """
    # Feature: swarmiq-v2, Property 30: Agent message round-trip serialization —
    # for any valid agent message object (SubTask, Claim, ScoredClaim, Resolution,
    # EvaluatorOutput), serializing to JSON via json.dumps(dataclasses.asdict(obj)),
    # then parsing back with json.loads, must produce a dict deeply equal to
    # dataclasses.asdict(obj).
    """

    @given(uuid_str, subtask_type, nonempty_text, keyword_list)
    @settings(max_examples=100)
    def test_subtask_round_trip(self, subtask_id: str, stype: str, description: str, keywords: list):
        # Feature: swarmiq-v2, Property 30: Agent message round-trip serialization
        obj = SubTask(
            subtask_id=subtask_id,
            type=stype,
            description=description,
            search_keywords=keywords,
        )
        original = dataclasses.asdict(obj)
        serialized = json.dumps(original)
        restored = json.loads(serialized)
        assert restored == original

    @given(uuid_str, nonempty_text, score, valid_uri, uuid_str)
    @settings(max_examples=100)
    def test_claim_round_trip(
        self, claim_id: str, claim_text: str, confidence: float, source_url: str, subtask_id: str
    ):
        # Feature: swarmiq-v2, Property 30: Agent message round-trip serialization
        obj = Claim(
            claim_id=claim_id,
            claim_text=claim_text,
            confidence=confidence,
            source_url=source_url,
            subtask_id=subtask_id,
        )
        original = dataclasses.asdict(obj)
        serialized = json.dumps(original)
        restored = json.loads(serialized)
        assert restored == original

    @given(uuid_str, nonempty_text, score, valid_uri, uuid_str, score)
    @settings(max_examples=100)
    def test_scored_claim_round_trip(
        self,
        claim_id: str,
        claim_text: str,
        confidence: float,
        source_url: str,
        subtask_id: str,
        credibility_score: float,
    ):
        # Feature: swarmiq-v2, Property 30: Agent message round-trip serialization
        obj = ScoredClaim(
            claim_id=claim_id,
            claim_text=claim_text,
            confidence=confidence,
            source_url=source_url,
            subtask_id=subtask_id,
            credibility_score=credibility_score,
        )
        original = dataclasses.asdict(obj)
        serialized = json.dumps(original)
        restored = json.loads(serialized)
        assert restored == original

    @given(uuid_str, resolution_status, nonempty_text, score)
    @settings(max_examples=100)
    def test_resolution_round_trip(
        self, claim_id: str, status: str, rationale: str, credibility_score: float
    ):
        # Feature: swarmiq-v2, Property 30: Agent message round-trip serialization
        obj = Resolution(
            claim_id=claim_id,
            status=status,
            rationale=rationale,
            credibility_score=credibility_score,
        )
        original = dataclasses.asdict(obj)
        serialized = json.dumps(original)
        restored = json.loads(serialized)
        assert restored == original

    @given(score, score, score, score, st.booleans(), st.lists(nonempty_text))
    @settings(max_examples=100)
    def test_evaluator_output_round_trip(
        self,
        coherence: float,
        factuality: float,
        citation_coverage: float,
        composite_score: float,
        passed: bool,
        deficiencies: list,
    ):
        # Feature: swarmiq-v2, Property 30: Agent message round-trip serialization
        obj = EvaluatorOutput(
            coherence=coherence,
            factuality=factuality,
            citation_coverage=citation_coverage,
            composite_score=composite_score,
            passed=passed,
            deficiencies=deficiencies,
        )
        original = dataclasses.asdict(obj)
        serialized = json.dumps(original)
        restored = json.loads(serialized)
        assert restored == original
