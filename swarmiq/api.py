import asyncio
import dataclasses
import hashlib
import json
import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from swarmiq.core.orchestrator import SwarmOrchestrator
from swarmiq.store.session_store import Session, SessionStore
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RunQueryRequest(BaseModel):
    query: str
    domain: str = "Research"


def create_router(orchestrator: SwarmOrchestrator, session_store: SessionStore) -> APIRouter:
    router = APIRouter()

    @router.post("/api/run")
    async def run_query(req: RunQueryRequest, request: Request):
        session_id = str(uuid.uuid4())
        query = req.query
        domain = req.domain

        async def event_generator():
            queue = await orchestrator.create_queue(session_id)
            pipeline_task = asyncio.create_task(
                orchestrator.run_pipeline(query, domain.lower(), session_id)
            )

            try:
                while True:
                    if await request.is_disconnected():
                        pipeline_task.cancel()
                        break

                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=0.25)
                        data = json.dumps(dataclasses.asdict(event))
                        yield f"data: {data}\n\n"
                        continue
                    except asyncio.TimeoutError:
                        pass

                    if pipeline_task.done() and queue.empty():
                        break

                if await request.is_disconnected():
                    return

                result = await pipeline_task

                evaluator_dict = None
                evaluator_json = ""
                if result.evaluator_output:
                    evaluator_dict = dataclasses.asdict(result.evaluator_output)
                    evaluator_json = json.dumps(evaluator_dict)

                figures_out = []
                for fig in result.figures:
                    fig_dict = dataclasses.asdict(fig)
                    if isinstance(fig_dict.get("data"), bytes):
                        fig_dict["data"] = fig_dict["data"].decode("utf-8")
                    figures_out.append(fig_dict)

                fingerprint = hashlib.sha256(query.encode()).hexdigest()
                session = Session(
                    session_id=session_id,
                    query_text=query,
                    query_fingerprint=fingerprint,
                    domain_mode=domain.lower(),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    report_markdown=result.report_markdown,
                    evaluator_json=evaluator_json,
                    status=result.status,
                )
                try:
                    session_store.save_session(session)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to persist session %s: %s", session_id, exc)

                final_data = {
                    "type": "final_result",
                    "report_markdown": result.report_markdown,
                    "references": [dataclasses.asdict(r) for r in result.references],
                    "evaluator_output": evaluator_dict,
                    "figures": figures_out,
                }
                yield f"data: {json.dumps(final_data)}\n\n"
            except asyncio.CancelledError:
                pipeline_task.cancel()
                raise
            finally:
                if not pipeline_task.done():
                    pipeline_task.cancel()
                try:
                    await pipeline_task
                except asyncio.CancelledError:
                    logger.info("Cancelled pipeline task for disconnected session %s", session_id)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Pipeline task for session %s failed: %s", session_id, exc)
                await orchestrator.remove_queue(session_id)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return router
