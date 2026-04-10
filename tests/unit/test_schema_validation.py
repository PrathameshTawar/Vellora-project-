"""
Unit tests for schema validation — SwarmIQ v2.

Validates: Requirements 15.1, 15.2
"""
import json
import pytest

from swarmiq.core.schemas import (
    SUBTASK_SCHEMA,
    CLAIM_SCHEMA,
    EVALUATOR_OUTPUT_SCHEMA,
    RESOLUTION_SCHEMA,
)
from swarmiq.core.validation import validate_message, SchemaValidationError, pretty_print_json

# ---------------------------------------------------------------------------
# Fixtures — minimal valid payloads
# ---------------------------------------------------------------------------

VALID_SUBTASK = {
    "subtask_id": "123e4567-e89b-12d3-a456-426614174000",
    "type": "literature_review",
    "description": "Review recent papers on LLMs.",
    "search_keywords": ["LLM", "transformer"],
}

VALID_CLAIM = {
    "claim_id": "123e4567-e89b-12d3-a456-426614174001",
    "claim_text": "LLMs can reason.",
    "confidence": 0.85,
    "source_url": "https://example.com/paper",
    "subtask_id": "123e4567-e89b-12d3-a456-426614174000",
}

VALID_EVALUATOR_OUTPUT = {
    "coherence": 0.9,
    "factuality": 0.8,
    "citation_coverage": 0.75,
    "composite_score": 0.82,
    "passed": True,
}

VALID_RESOLUTION = {
    "claim_id": "123e4567-e89b-12d3-a456-426614174001",
    "status": "accepted",
    "rationale": "Supported by multiple sources.",
    "credibility_score": 0.9,
}


# ---------------------------------------------------------------------------
# Valid payload tests — should pass silently
# ---------------------------------------------------------------------------

def test_valid_subtask_passes():
    validate_message(VALID_SUBTASK, SUBTASK_SCHEMA)


def test_valid_claim_passes():
    validate_message(VALID_CLAIM, CLAIM_SCHEMA)


def test_valid_evaluator_output_passes():
    validate_message(VALID_EVALUATOR_OUTPUT, EVALUATOR_OUTPUT_SCHEMA)


def test_valid_resolution_passes():
    validate_message(VALID_RESOLUTION, RESOLUTION_SCHEMA)


# ---------------------------------------------------------------------------
# SubTask invalid payloads
# ---------------------------------------------------------------------------

def test_subtask_missing_subtask_id_raises():
    payload = {k: v for k, v in VALID_SUBTASK.items() if k != "subtask_id"}
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_message(payload, SUBTASK_SCHEMA)
    assert "subtask_id" in exc_info.value.field_path or "subtask_id" in str(exc_info.value)


def test_subtask_invalid_type_enum_raises():
    payload = {**VALID_SUBTASK, "type": "invalid_type"}
    with pytest.raises(SchemaValidationError):
        validate_message(payload, SUBTASK_SCHEMA)


def test_subtask_empty_search_keywords_raises():
    payload = {**VALID_SUBTASK, "search_keywords": []}
    with pytest.raises(SchemaValidationError):
        validate_message(payload, SUBTASK_SCHEMA)


# ---------------------------------------------------------------------------
# Claim invalid payloads
# ---------------------------------------------------------------------------

def test_claim_confidence_above_max_raises():
    payload = {**VALID_CLAIM, "confidence": 1.5}
    with pytest.raises(SchemaValidationError):
        validate_message(payload, CLAIM_SCHEMA)


def test_claim_confidence_below_min_raises():
    payload = {**VALID_CLAIM, "confidence": -0.1}
    with pytest.raises(SchemaValidationError):
        validate_message(payload, CLAIM_SCHEMA)


def test_claim_missing_claim_text_raises():
    payload = {k: v for k, v in VALID_CLAIM.items() if k != "claim_text"}
    with pytest.raises(SchemaValidationError):
        validate_message(payload, CLAIM_SCHEMA)


# ---------------------------------------------------------------------------
# EvaluatorOutput invalid payloads
# ---------------------------------------------------------------------------

def test_evaluator_output_passed_string_raises():
    payload = {**VALID_EVALUATOR_OUTPUT, "passed": "yes"}
    with pytest.raises(SchemaValidationError):
        validate_message(payload, EVALUATOR_OUTPUT_SCHEMA)


def test_evaluator_output_coherence_above_max_raises():
    payload = {**VALID_EVALUATOR_OUTPUT, "coherence": 2.0}
    with pytest.raises(SchemaValidationError):
        validate_message(payload, EVALUATOR_OUTPUT_SCHEMA)


# ---------------------------------------------------------------------------
# Resolution invalid payloads
# ---------------------------------------------------------------------------

def test_resolution_invalid_status_raises():
    payload = {**VALID_RESOLUTION, "status": "maybe"}
    with pytest.raises(SchemaValidationError):
        validate_message(payload, RESOLUTION_SCHEMA)


# ---------------------------------------------------------------------------
# SchemaValidationError attribute tests
# ---------------------------------------------------------------------------

def test_schema_validation_error_has_non_empty_field_path():
    payload = {**VALID_CLAIM, "confidence": 1.5}
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_message(payload, CLAIM_SCHEMA)
    assert exc_info.value.field_path != ""


def test_schema_validation_error_has_non_empty_schema_path():
    payload = {**VALID_CLAIM, "confidence": 1.5}
    with pytest.raises(SchemaValidationError) as exc_info:
        validate_message(payload, CLAIM_SCHEMA)
    assert exc_info.value.schema_path != ""


# ---------------------------------------------------------------------------
# pretty_print_json tests
# ---------------------------------------------------------------------------

def test_pretty_print_json_returns_string_with_2_space_indent():
    result = pretty_print_json({"key": "value"})
    assert isinstance(result, str)
    # 2-space indent means the value line starts with exactly 2 spaces
    assert '  "key": "value"' in result


def test_pretty_print_json_nested_dict_round_trips():
    obj = {"outer": {"inner": [1, 2, 3]}, "flag": True}
    result = pretty_print_json(obj)
    assert isinstance(result, str)
    assert json.loads(result) == obj
