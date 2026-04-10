"""
Schema validation and pretty-printer utilities for SwarmIQ v2.
"""
from __future__ import annotations

import json
from collections import deque
from typing import Any

import jsonschema
from jsonschema import ValidationError


class SchemaValidationError(Exception):
    """Raised when a payload fails JSON schema validation.

    Attributes:
        message:     Human-readable description of the failure.
        field_path:  Dot-separated path to the failing field (e.g. "claims[0].confidence").
        schema_path: Dot-separated path within the schema that triggered the error.
    """

    def __init__(self, error: ValidationError) -> None:
        self.field_path: str = _deque_to_path(error.absolute_path)
        self.schema_path: str = _deque_to_path(error.absolute_schema_path)
        detail = (
            f"{error.message}"
            + (f" (field: {self.field_path})" if self.field_path else "")
            + (f" (schema path: {self.schema_path})" if self.schema_path else "")
        )
        super().__init__(detail)


def _deque_to_path(d: deque) -> str:
    """Convert a jsonschema path deque to a human-readable dot/bracket string."""
    parts: list[str] = []
    for item in d:
        if isinstance(item, int):
            parts.append(f"[{item}]")
        else:
            parts.append(str(item))
    # Join with dots but collapse "field[0]" style segments correctly
    result = ""
    for part in parts:
        if part.startswith("["):
            result += part
        elif result:
            result += f".{part}"
        else:
            result = part
    return result


def validate_message(payload: dict, schema: dict) -> None:
    """Validate *payload* against *schema* using jsonschema.

    Args:
        payload: The dict to validate.
        schema:  A JSON Schema dict.

    Raises:
        SchemaValidationError: If validation fails, with field-level details.
    """
    try:
        jsonschema.validate(instance=payload, schema=schema)
    except ValidationError as exc:
        raise SchemaValidationError(exc) from exc


def pretty_print_json(obj: Any) -> str:
    """Return a pretty-printed JSON string for *obj* with 2-space indentation.

    Args:
        obj: Any JSON-serialisable object (dict, list, etc.).

    Returns:
        A formatted JSON string.
    """
    return json.dumps(obj, indent=2)
