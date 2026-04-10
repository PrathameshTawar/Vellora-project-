"""
SwarmOrchestrator for SwarmIQ v2.
Standalone async orchestrator that manages the full research pipeline.
"""
from __future__ import annotations

import asyncio
import dataclasses
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from swarmiq.agents.critic import CriticAgent
from swarmiq.agents.evaluator import EvaluatorAgent
from swarmiq.agents.literature import LiteratureAgent, LiteratureOutput
from swarmiq.agents.planner import PlannerAgent
from swarmiq.agents.summarizer import SummarizerAgent
from swarmiq.agents.synthesizer import SynthesizerAgent
from swarmiq.agents.visualization import VisualizationAgent
from swarmiq.agents.conflict_resolver import ConflictResolverAgent
from swarmiq.core.credibility import score_claims
from swarmiq.core.models import (
    ActivityEvent,
    AgentError,
    Claim,
    EvaluatorOutput,
    Figure,
    PlannerOutput,
    Reference,
    Resolution,
    ScoredClaim,
    SubTask,
)
from swarmiq.core.schemas import (
    CONFLICT_RESOLVER_OUTPUT_SCHEMA,
    EVALUATOR_OUTPUT_SCHEMA,
    PLANNER_OUTPUT_SCHEMA,
    SCORED_CLAIM_SCHEMA,
    SUMMARIZER_OUTPUT_SCHEMA,
)
from swarmiq.core.validation import SchemaValidationError, validate_message

logger = logging.getLogger(__name__)

_MAX_LITERATURE_AGENTS = 5
_MIN_LITERATURE_AGENTS = 3


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_event_id() -> str:
    return str(uuid.uuid4())


# PipelineResult



@dataclass
class PipelineResult:
    """The complete output of a SwarmOrchestrator pipeline run."""

    session_id: str
    query: str
    domain_mode: str
    subtasks: list[SubTask] = field(default_factory=list)
    report_markdown: str = ""
    references: list[Reference] = field(default_factory=list)
    evaluator_output: Optional[EvaluatorOutput] = None
    figures: list[Figure] = field(default_factory=list)
    activity_events: list[ActivityEvent] = field(default_factory=list)
    errors: list[AgentError] = field(default_factory=list)
    status: str = "complete"  # "complete" | "failed" | "partial"


# SwarmOrchestrator



class SwarmOrchestrator:
    """Async orchestrator for the SwarmIQ v2 research pipeline.

    Args:
        planner:                  PlannerAgent instance.
        literature_agent_factory: Callable[[], LiteratureAgent] — creates a fresh
                                  LiteratureAgent for each subtask.
        summarizer:               SummarizerAgent instance.
        conflict_resolver:        ConflictResolverAgent instance.
        synthesizer:              SynthesizerAgent instance.
        evaluator:                EvaluatorAgent instance.
        visualization:            VisualizationAgent instance.
    """

    def __init__(
        self,
        planner: PlannerAgent,
        literature_agent_factory: Callable[[], LiteratureAgent],
        summarizer: SummarizerAgent,
        conflict_resolver: ConflictResolverAgent,
        synthesizer: SynthesizerAgent,
        evaluator: EvaluatorAgent,
        visualization: VisualizationAgent,
        critic: CriticAgent | None = None,
    ) -> None:
        self._planner = planner
        self._literature_agent_factory = literature_agent_factory
        self._summarizer = summarizer
        self._conflict_resolver = conflict_resolver
        self._synthesizer = synthesizer
        self._evaluator = evaluator
        self._visualization = visualization
        self._critic = critic
        self._queues: dict[str, asyncio.Queue[ActivityEvent]] = {}
        self._queues_lock = asyncio.Lock()

    async def create_queue(self, session_id: str) -> asyncio.Queue[ActivityEvent]:
        """Create and register a fresh activity queue for a specific session."""
        queue: asyncio.Queue[ActivityEvent] = asyncio.Queue()
        async with self._queues_lock:
            self._queues[session_id] = queue
        return queue

    async def get_queue(self, session_id: str) -> Optional[asyncio.Queue[ActivityEvent]]:
        """Get the activity queue for a specific session, if it exists."""
        async with self._queues_lock:
            return self._queues.get(session_id)

    async def remove_queue(self, session_id: str) -> None:
        """Remove the activity queue for a specific session once done."""
        async with self._queues_lock:
            self._queues.pop(session_id, None)

    # Public API


    async def run_pipeline(
        self,
        query: str,
        domain_mode: str,
        session_id: str,
    ) -> PipelineResult:
        """Execute the full research pipeline and return a PipelineResult.

        Args:
            query:       The user's research query.
            domain_mode: One of "research", "business", or "policy".
            session_id:  Unique identifier for this session.

        Returns:
            PipelineResult with all outputs, activity events, and errors.
        """
        result = PipelineResult(
            session_id=session_id,
            query=query,
            domain_mode=domain_mode,
        )

        # ── Step 1: Planner ─────────────────────────────────────────────
        subtasks = await self._run_planner(query, domain_mode, result)
        if subtasks is None:
            result.status = "failed"
            return self._finalise(result)

        result.subtasks = subtasks

        # Cap to _MAX_LITERATURE_AGENTS
        capped_subtasks = subtasks[:_MAX_LITERATURE_AGENTS]

        # ── Step 2: Literature Agents (parallel) ─────────────────────────
        literature_outputs = await self._run_literature_agents(
            capped_subtasks, session_id, query, result
        )

        # ── Step 3: Summarizer per subtask (parallel) ────────────────────
        async def _summarize_one(st, lo):
            if lo is None:
                return []
            return await self._run_summarizer(st, lo, result)

        summarizer_tasks = [
            _summarize_one(st, lo) 
            for st, lo in zip(capped_subtasks, literature_outputs)
        ]
        summarizer_outputs = await asyncio.gather(*summarizer_tasks)

        all_claims: list[Claim] = []
        for claims in summarizer_outputs:
            all_claims.extend(claims)

        # ── Step 4: Credibility scoring ──────────────────────────────────
        scored_claims = await self._run_credibility_scorer(all_claims, result)

        # ── Step 5: Conflict resolution ──────────────────────────────────
        resolutions = await self._run_conflict_resolver(scored_claims, result)

        # ── Step 6: Synthesizer ──────────────────────────────────────────
        synth_output = await self._run_synthesizer(
            resolutions, domain_mode, subtasks, scored_claims, result
        )
        if synth_output is not None:
            result.report_markdown = synth_output.report_markdown
            result.references = synth_output.references

        # ── Step 6b: Critic (optional) ───────────────────────────────────
        if self._critic is not None and result.report_markdown:
            research_questions = [st.description for st in result.subtasks]
            sources = [ref.url for ref in result.references]
            critic_result = await self._critic.critique(
                result.report_markdown,
                research_questions,
                sources,
            )
            result.report_markdown = critic_result["report"]
            await self._emit(
                agent_type="critic_agent",
                status="completed",
                message=(
                    f"Critic pass complete. revisions={critic_result['revisions']}, "
                    f"passed={critic_result['passed']}."
                ),
                subtask_id=None,
                result=result,
            )

        # ── Step 7: Evaluator ────────────────────────────────────────────
        evaluator_output = await self._run_evaluator(
            result.report_markdown, scored_claims, result.references, result
        )
        result.evaluator_output = evaluator_output

        # ── Step 8: Visualization ────────────────────────────────────────
        figures = await self._run_visualization(scored_claims, result)
        result.figures = figures

        # ── Determine final status ───────────────────────────────────────
        if result.errors:
            result.status = "partial" if result.report_markdown else "failed"
        else:
            result.status = "complete"

        return self._finalise(result)

    # Pipeline step helpers


    async def _run_planner(
        self,
        query: str,
        domain_mode: str,
        result: PipelineResult,
    ) -> Optional[list[SubTask]]:
        """Call PlannerAgent and validate output. Returns subtasks or None on failure."""
        await self._emit(
            agent_type="planner_agent",
            status="dispatched",
            message=f"Decomposing query: {query!r}",
            subtask_id=None,
            result=result,
        )

        planner_result = await self._planner.decompose(query, domain_mode)

        # Handle INVALID_QUERY dict
        if isinstance(planner_result, dict):
            error_code = planner_result.get("error", "PLANNER_ERROR")
            msg = planner_result.get("message", str(planner_result))
            err = AgentError(
                agent_type="planner_agent",
                error_code=error_code,
                message=msg,
                timestamp=_now_iso(),
            )
            result.errors.append(err)
            await self._emit(
                agent_type="planner_agent",
                status="failed",
                message=msg,
                subtask_id=None,
                result=result,
            )
            return None

        # Handle AgentError
        if isinstance(planner_result, AgentError):
            result.errors.append(planner_result)
            await self._emit(
                agent_type="planner_agent",
                status="failed",
                message=planner_result.message,
                subtask_id=None,
                result=result,
            )
            return None

        # planner_result is PlannerOutput — validate schema
        planner_output: PlannerOutput = planner_result
        payload = {
            "subtasks": [dataclasses.asdict(st) for st in planner_output.subtasks]
        }
        try:
            validate_message(payload, PLANNER_OUTPUT_SCHEMA)
        except SchemaValidationError as exc:
            err = AgentError(
                agent_type="planner_agent",
                error_code="SCHEMA_VALIDATION_ERROR",
                message=str(exc),
                timestamp=_now_iso(),
            )
            result.errors.append(err)
            await self._emit(
                agent_type="planner_agent",
                status="failed",
                message=f"Schema validation error: {exc}",
                subtask_id=None,
                result=result,
            )
            return None

        await self._emit(
            agent_type="planner_agent",
            status="completed",
            message=f"Produced {len(planner_output.subtasks)} subtasks.",
            subtask_id=None,
            result=result,
        )
        return planner_output.subtasks

    async def _run_literature_agents(
        self,
        subtasks: list[SubTask],
        session_id: str,
        query: str,
        result: PipelineResult,
    ) -> list[Optional[LiteratureOutput]]:
        """Spawn one LiteratureAgent per subtask via asyncio.gather."""
        import hashlib

        query_fingerprint = hashlib.sha256(query.encode()).hexdigest()

        async def _run_one(subtask: SubTask) -> Optional[LiteratureOutput]:
            await self._emit(
                agent_type="literature_agent",
                status="dispatched",
                message=f"Retrieving documents for subtask: {subtask.description!r}",
                subtask_id=subtask.subtask_id,
                result=result,
            )
            agent = self._literature_agent_factory()
            lit_result = await agent.run(subtask, session_id, query_fingerprint)

            if isinstance(lit_result, AgentError):
                result.errors.append(lit_result)
                await self._emit(
                    agent_type="literature_agent",
                    status="failed",
                    message=lit_result.message,
                    subtask_id=subtask.subtask_id,
                    result=result,
                )
                return None

            await self._emit(
                agent_type="literature_agent",
                status="completed",
                message=(
                    f"Retrieved {len(lit_result.documents)} documents "
                    f"for subtask {subtask.subtask_id}."
                ),
                subtask_id=subtask.subtask_id,
                result=result,
            )
            return lit_result

        outputs = await asyncio.gather(*[_run_one(st) for st in subtasks])
        return list(outputs)

    async def _run_summarizer(
        self,
        subtask: SubTask,
        lit_output: LiteratureOutput,
        result: PipelineResult,
    ) -> list[Claim]:
        """Call SummarizerAgent for one subtask. Returns claims (may be empty)."""
        await self._emit(
            agent_type="summarizer_agent",
            status="dispatched",
            message=f"Summarizing {len(lit_output.documents)} documents for subtask {subtask.subtask_id}.",
            subtask_id=subtask.subtask_id,
            result=result,
        )

        summ_result = await self._summarizer.summarize(
            subtask.subtask_id, lit_output.documents
        )

        if isinstance(summ_result, AgentError):
            result.errors.append(summ_result)
            await self._emit(
                agent_type="summarizer_agent",
                status="failed",
                message=summ_result.message,
                subtask_id=subtask.subtask_id,
                result=result,
            )
            return []

        # Validate summarizer output schema
        payload = {"claims": [dataclasses.asdict(c) for c in summ_result.claims]}
        try:
            validate_message(payload, SUMMARIZER_OUTPUT_SCHEMA)
        except SchemaValidationError as exc:
            err = AgentError(
                agent_type="summarizer_agent",
                error_code="SCHEMA_VALIDATION_ERROR",
                message=str(exc),
                timestamp=_now_iso(),
                subtask_id=subtask.subtask_id,
            )
            result.errors.append(err)
            await self._emit(
                agent_type="summarizer_agent",
                status="failed",
                message=f"Schema validation error: {exc}",
                subtask_id=subtask.subtask_id,
                result=result,
            )
            # Halt this subtask's pipeline — return no claims
            return []

        await self._emit(
            agent_type="summarizer_agent",
            status="completed",
            message=f"Extracted {len(summ_result.claims)} claims for subtask {subtask.subtask_id}.",
            subtask_id=subtask.subtask_id,
            result=result,
        )
        return summ_result.claims

    async def _run_credibility_scorer(
        self,
        claims: list[Claim],
        result: PipelineResult,
    ) -> list[ScoredClaim]:
        """Score claims synchronously (pure function). Returns scored claims."""
        await self._emit(
            agent_type="credibility_scorer",
            status="dispatched",
            message=f"Scoring {len(claims)} claims.",
            subtask_id=None,
            result=result,
        )

        try:
            scored = score_claims(claims)
        except Exception as exc:  # noqa: BLE001
            logger.exception("CredibilityScorer failed: %s", exc)
            err = AgentError(
                agent_type="credibility_scorer",
                error_code="UNHANDLED_EXCEPTION",
                message=str(exc),
                timestamp=_now_iso(),
            )
            result.errors.append(err)
            await self._emit(
                agent_type="credibility_scorer",
                status="failed",
                message=str(exc),
                subtask_id=None,
                result=result,
            )
            return []

        # Validate each scored claim against schema
        valid_scored: list[ScoredClaim] = []
        for sc in scored:
            payload = dataclasses.asdict(sc)
            try:
                validate_message(payload, SCORED_CLAIM_SCHEMA)
                valid_scored.append(sc)
            except SchemaValidationError as exc:
                err = AgentError(
                    agent_type="credibility_scorer",
                    error_code="SCHEMA_VALIDATION_ERROR",
                    message=str(exc),
                    timestamp=_now_iso(),
                    subtask_id=sc.subtask_id,
                )
                result.errors.append(err)
                await self._emit(
                    agent_type="credibility_scorer",
                    status="failed",
                    message=f"Schema validation error on claim {sc.claim_id}: {exc}",
                    subtask_id=sc.subtask_id,
                    result=result,
                )
                # Halt this claim — skip it

        await self._emit(
            agent_type="credibility_scorer",
            status="completed",
            message=f"Scored {len(valid_scored)} claims.",
            subtask_id=None,
            result=result,
        )
        return valid_scored

    async def _run_conflict_resolver(
        self,
        scored_claims: list[ScoredClaim],
        result: PipelineResult,
    ) -> list[Resolution]:
        """Call ConflictResolverAgent and validate output. Returns resolutions."""
        await self._emit(
            agent_type="conflict_resolver_agent",
            status="dispatched",
            message=f"Resolving conflicts among {len(scored_claims)} scored claims.",
            subtask_id=None,
            result=result,
        )

        try:
            cr_output = self._conflict_resolver.resolve(scored_claims)
        except Exception as exc:  # noqa: BLE001
            logger.exception("ConflictResolverAgent failed: %s", exc)
            err = AgentError(
                agent_type="conflict_resolver_agent",
                error_code="UNHANDLED_EXCEPTION",
                message=str(exc),
                timestamp=_now_iso(),
            )
            result.errors.append(err)
            await self._emit(
                agent_type="conflict_resolver_agent",
                status="failed",
                message=str(exc),
                subtask_id=None,
                result=result,
            )
            return []

        # Validate output schema
        import dataclasses as dc
        payload = {"resolutions": [dc.asdict(r) for r in cr_output.resolutions]}
        try:
            validate_message(payload, CONFLICT_RESOLVER_OUTPUT_SCHEMA)
        except SchemaValidationError as exc:
            err = AgentError(
                agent_type="conflict_resolver_agent",
                error_code="SCHEMA_VALIDATION_ERROR",
                message=str(exc),
                timestamp=_now_iso(),
            )
            result.errors.append(err)
            await self._emit(
                agent_type="conflict_resolver_agent",
                status="failed",
                message=f"Schema validation error: {exc}",
                subtask_id=None,
                result=result,
            )
            return []

        await self._emit(
            agent_type="conflict_resolver_agent",
            status="completed",
            message=f"Resolved {len(cr_output.resolutions)} claims.",
            subtask_id=None,
            result=result,
        )
        return cr_output.resolutions

    async def _run_synthesizer(
        self,
        resolutions: list[Resolution],
        domain_mode: str,
        subtasks: list[SubTask],
        scored_claims: list[ScoredClaim],
        result: PipelineResult,
    ):
        """Call SynthesizerAgent. Returns SynthesizerOutput or None on failure."""
        from swarmiq.core.models import SynthesizerOutput

        await self._emit(
            agent_type="synthesizer_agent",
            status="dispatched",
            message="Synthesizing final report.",
            subtask_id=None,
            result=result,
        )

        synth_result = await self._synthesizer.synthesize(
            resolutions, domain_mode, subtasks, scored_claims
        )

        if isinstance(synth_result, AgentError):
            result.errors.append(synth_result)
            await self._emit(
                agent_type="synthesizer_agent",
                status="failed",
                message=synth_result.message,
                subtask_id=None,
                result=result,
            )
            return None

        await self._emit(
            agent_type="synthesizer_agent",
            status="completed",
            message=f"Report synthesized ({len(synth_result.report_markdown)} chars).",
            subtask_id=None,
            result=result,
        )
        return synth_result

    async def _run_evaluator(
        self,
        report_markdown: str,
        scored_claims: list[ScoredClaim],
        references: list[Reference],
        result: PipelineResult,
    ) -> Optional[EvaluatorOutput]:
        """Call EvaluatorAgent and validate output. Returns EvaluatorOutput or None."""
        await self._emit(
            agent_type="evaluator_agent",
            status="dispatched",
            message="Evaluating report quality.",
            subtask_id=None,
            result=result,
        )

        eval_result = await self._evaluator.evaluate(
            report_markdown, scored_claims, references
        )

        if isinstance(eval_result, AgentError):
            result.errors.append(eval_result)
            await self._emit(
                agent_type="evaluator_agent",
                status="failed",
                message=eval_result.message,
                subtask_id=None,
                result=result,
            )
            return None

        # Validate output schema
        import dataclasses as dc
        payload = dc.asdict(eval_result)
        try:
            validate_message(payload, EVALUATOR_OUTPUT_SCHEMA)
        except SchemaValidationError as exc:
            err = AgentError(
                agent_type="evaluator_agent",
                error_code="SCHEMA_VALIDATION_ERROR",
                message=str(exc),
                timestamp=_now_iso(),
            )
            result.errors.append(err)
            await self._emit(
                agent_type="evaluator_agent",
                status="failed",
                message=f"Schema validation error: {exc}",
                subtask_id=None,
                result=result,
            )
            return None

        await self._emit(
            agent_type="evaluator_agent",
            status="completed",
            message=(
                f"Evaluation complete. composite_score={eval_result.composite_score:.3f}, "
                f"passed={eval_result.passed}."
            ),
            subtask_id=None,
            result=result,
        )
        return eval_result

    async def _run_visualization(
        self,
        scored_claims: list[ScoredClaim],
        result: PipelineResult,
    ) -> list[Figure]:
        """Call VisualizationAgent. Returns figures list."""
        await self._emit(
            agent_type="visualization_agent",
            status="dispatched",
            message=f"Generating visualizations from {len(scored_claims)} scored claims.",
            subtask_id=None,
            result=result,
        )

        try:
            viz_output = self._visualization.generate(scored_claims)
        except Exception as exc:  # noqa: BLE001
            logger.exception("VisualizationAgent failed: %s", exc)
            err = AgentError(
                agent_type="visualization_agent",
                error_code="UNHANDLED_EXCEPTION",
                message=str(exc),
                timestamp=_now_iso(),
            )
            result.errors.append(err)
            await self._emit(
                agent_type="visualization_agent",
                status="failed",
                message=str(exc),
                subtask_id=None,
                result=result,
            )
            return []

        await self._emit(
            agent_type="visualization_agent",
            status="completed",
            message=f"Generated {len(viz_output.figures)} figure(s).",
            subtask_id=None,
            result=result,
        )
        return viz_output.figures

    # Activity event helpers


    async def _emit(
        self,
        agent_type: str,
        status: str,
        message: str,
        subtask_id: Optional[str],
        result: PipelineResult,
    ) -> None:
        """Create an ActivityEvent, put it on the queue, and append to result."""
        event = ActivityEvent(
            event_id=_new_event_id(),
            agent_type=agent_type,
            status=status,
            timestamp=_now_iso(),
            message=message,
            subtask_id=subtask_id,
        )
        result.activity_events.append(event)
        queue = await self.get_queue(result.session_id)
        if queue is not None:
            await queue.put(event)

    # Finalise


    def _finalise(self, result: PipelineResult) -> PipelineResult:
        """Drain any remaining queue items into result.activity_events (no-op here)."""
        return result
