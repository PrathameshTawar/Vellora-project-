"""
JSON Schema dicts for all agent message types in SwarmIQ v2.
Used by the validation layer to verify inter-agent payloads.
"""

SUBTASK_SCHEMA: dict = {
    "type": "object",
    "required": ["subtask_id", "type", "description", "search_keywords"],
    "properties": {
        "subtask_id": {"type": "string", "format": "uuid"},
        "type": {
            "enum": [
                "literature_review",
                "summarization",
                "visualization",
                "conflict_resolution",
                "synthesis",
            ]
        },
        "description": {"type": "string", "minLength": 1},
        "search_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "additionalProperties": False,
}

CLAIM_SCHEMA: dict = {
    "type": "object",
    "required": ["claim_id", "claim_text", "confidence", "source_url", "subtask_id"],
    "properties": {
        "claim_id": {"type": "string", "format": "uuid"},
        "claim_text": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "source_url": {"type": "string", "format": "uri"},
        "subtask_id": {"type": "string", "format": "uuid"},
    },
    "additionalProperties": False,
}

SCORED_CLAIM_SCHEMA: dict = {
    "type": "object",
    "required": [
        "claim_id",
        "claim_text",
        "confidence",
        "source_url",
        "subtask_id",
        "credibility_score",
    ],
    "properties": {
        "claim_id": {"type": "string", "format": "uuid"},
        "claim_text": {"type": "string", "minLength": 1},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "source_url": {"type": "string", "format": "uri"},
        "subtask_id": {"type": "string", "format": "uuid"},
        "credibility_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}

RESOLUTION_SCHEMA: dict = {
    "type": "object",
    "required": ["claim_id", "status", "rationale", "credibility_score"],
    "properties": {
        "claim_id": {"type": "string", "format": "uuid"},
        "status": {"enum": ["accepted", "rejected", "uncertain"]},
        "rationale": {"type": "string", "minLength": 1},
        "credibility_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "additionalProperties": False,
}

EVALUATOR_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "required": [
        "coherence",
        "factuality",
        "citation_coverage",
        "composite_score",
        "passed",
    ],
    "properties": {
        "coherence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "factuality": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "citation_coverage": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "composite_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "passed": {"type": "boolean"},
        "deficiencies": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

DOCUMENT_SCHEMA: dict = {
    "type": "object",
    "required": ["url", "title", "content", "retrieved_at", "subtask_id"],
    "properties": {
        "url": {"type": "string", "format": "uri"},
        "title": {"type": "string", "minLength": 1},
        "content": {"type": "string"},
        "retrieved_at": {"type": "string"},  # ISO-8601
        "subtask_id": {"type": "string", "format": "uuid"},
    },
    "additionalProperties": False,
}

REFERENCE_SCHEMA: dict = {
    "type": "object",
    "required": ["ref_id", "url", "title"],
    "properties": {
        "ref_id": {"type": "integer"},
        "url": {"type": "string", "format": "uri"},
        "title": {"type": "string", "minLength": 1},
        "authors": {"type": "array", "items": {"type": "string"}},
        "year": {"type": ["integer", "null"]},
    },
    "additionalProperties": False,
}

FIGURE_SCHEMA: dict = {
    "type": "object",
    "required": ["figure_id", "figure_type", "data"],
    "properties": {
        "figure_id": {"type": "string"},
        "figure_type": {"enum": ["plotly", "matplotlib"]},
        "data": {"type": "string"},  # JSON string for plotly; base64 for matplotlib
    },
    "additionalProperties": False,
}

ACTIVITY_EVENT_SCHEMA: dict = {
    "type": "object",
    "required": ["event_id", "agent_type", "status", "timestamp", "message"],
    "properties": {
        "event_id": {"type": "string"},
        "agent_type": {"type": "string", "minLength": 1},
        "subtask_id": {"type": ["string", "null"]},
        "status": {"type": "string", "minLength": 1},
        "timestamp": {"type": "string"},  # ISO-8601
        "message": {"type": "string"},
    },
    "additionalProperties": False,
}

AGENT_ERROR_SCHEMA: dict = {
    "type": "object",
    "required": ["agent_type", "error_code", "message", "timestamp"],
    "properties": {
        "agent_type": {"type": "string", "minLength": 1},
        "subtask_id": {"type": ["string", "null"]},
        "error_code": {"type": "string", "minLength": 1},
        "message": {"type": "string"},
        "timestamp": {"type": "string"},  # ISO-8601
    },
    "additionalProperties": False,
}

PLANNER_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "required": ["subtasks"],
    "properties": {
        "subtasks": {
            "type": "array",
            "items": SUBTASK_SCHEMA,
            "minItems": 3,
            "maxItems": 5,
        }
    },
    "additionalProperties": False,
}

SUMMARIZER_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "required": ["claims"],
    "properties": {
        "claims": {
            "type": "array",
            "items": CLAIM_SCHEMA,
        }
    },
    "additionalProperties": False,
}

CONFLICT_RESOLVER_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "required": ["resolutions"],
    "properties": {
        "resolutions": {
            "type": "array",
            "items": RESOLUTION_SCHEMA,
        }
    },
    "additionalProperties": False,
}

# Convenience mapping: type name → schema dict
SCHEMA_REGISTRY: dict[str, dict] = {
    "SubTask": SUBTASK_SCHEMA,
    "Claim": CLAIM_SCHEMA,
    "ScoredClaim": SCORED_CLAIM_SCHEMA,
    "Resolution": RESOLUTION_SCHEMA,
    "EvaluatorOutput": EVALUATOR_OUTPUT_SCHEMA,
    "Document": DOCUMENT_SCHEMA,
    "Reference": REFERENCE_SCHEMA,
    "Figure": FIGURE_SCHEMA,
    "ActivityEvent": ACTIVITY_EVENT_SCHEMA,
    "AgentError": AGENT_ERROR_SCHEMA,
    "PlannerOutput": PLANNER_OUTPUT_SCHEMA,
    "SummarizerOutput": SUMMARIZER_OUTPUT_SCHEMA,
    "ConflictResolverOutput": CONFLICT_RESOLVER_OUTPUT_SCHEMA,
}
