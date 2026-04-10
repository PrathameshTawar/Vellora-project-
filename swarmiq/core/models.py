"""
Dataclasses for all agent message types in SwarmIQ v2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SubTask:
    subtask_id: str
    type: str  # literature_review | summarization | visualization | conflict_resolution | synthesis
    description: str
    search_keywords: list[str] = field(default_factory=list)


@dataclass
class Claim:
    claim_id: str
    claim_text: str
    confidence: float
    source_url: str
    subtask_id: str


@dataclass
class ScoredClaim(Claim):
    credibility_score: float = 0.0


@dataclass
class Resolution:
    claim_id: str
    status: str  # accepted | rejected | uncertain
    rationale: str
    credibility_score: float


@dataclass
class EvaluatorOutput:
    coherence: float
    factuality: float
    citation_coverage: float
    composite_score: float
    passed: bool
    deficiencies: list[str] = field(default_factory=list)


@dataclass
class Document:
    url: str
    title: str
    content: str
    retrieved_at: str
    subtask_id: str


@dataclass
class Reference:
    ref_id: int
    url: str
    title: str
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None


@dataclass
class Figure:
    figure_id: str
    figure_type: str  # plotly | matplotlib
    data: str | bytes = ""


@dataclass
class ActivityEvent:
    event_id: str
    agent_type: str
    status: str
    timestamp: str
    message: str
    subtask_id: Optional[str] = None


@dataclass
class AgentError:
    agent_type: str
    error_code: str
    message: str
    timestamp: str
    subtask_id: Optional[str] = None


# ── Composite I/O types ──────────────────────────────────────────────────────

@dataclass
class PlannerOutput:
    subtasks: list[SubTask] = field(default_factory=list)


@dataclass
class SummarizerOutput:
    claims: list[Claim] = field(default_factory=list)


@dataclass
class ConflictResolverOutput:
    resolutions: list[Resolution] = field(default_factory=list)


@dataclass
class SynthesizerOutput:
    report_markdown: str = ""
    references: list[Reference] = field(default_factory=list)
