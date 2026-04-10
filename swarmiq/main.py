"""
SwarmIQ v2 — application entry point.
Wires together all components and launches FastAPI app + frontend.
"""
from __future__ import annotations

import logging

from swarmiq import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Config


from swarmiq import config  # noqa: E402

# Check required OpenAI key early
if not config.openai_api_key:
    logger.error("OPENAI_API_KEY is required. Exiting.")
    raise ValueError("Missing OPENAI_API_KEY")

_MISSING: list[str] = []
for _key in ("pinecone_api_key", "tavily_api_key"):
    if not getattr(config, _key, None):
        _MISSING.append(_key.replace("_", "_").upper())
        logger.warning(
            "Environment variable %s not set. Some functionality will be limited.",
            _key.upper().replace("_", "_"),
        )

# Build shared llm_config


llm_config: dict = {
    "config_list": [
        {
            "model": config.model_name,
            "api_key": config.openai_api_key,
        }
    ],
    "temperature": config.temperature,
}

# Instantiate stores


from swarmiq.core.knowledge_store import KnowledgeStore  # noqa: E402
from swarmiq.store.session_store import SessionStore  # noqa: E402

knowledge_store: KnowledgeStore | None = None
if config.pinecone_api_key:
    try:
        knowledge_store = KnowledgeStore(
            api_key=config.pinecone_api_key,
            index_name=config.pinecone_index_name,
        )
        logger.info("KnowledgeStore initialised (index: %s).", config.pinecone_index_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("KnowledgeStore could not be initialised: %s", exc)
else:
    logger.warning(
        "PINECONE_API_KEY not set — KnowledgeStore is unavailable. "
        "Literature retrieval and session deletion will not work."
    )

session_store = SessionStore()
logger.info("SessionStore initialised.")

# Instantiate agents


from swarmiq.agents.planner import PlannerAgent  # noqa: E402
from swarmiq.agents.literature import LiteratureAgent  # noqa: E402
from swarmiq.agents.summarizer import SummarizerAgent  # noqa: E402
from swarmiq.agents.conflict_resolver import ConflictResolverAgent  # noqa: E402
from swarmiq.agents.synthesizer import SynthesizerAgent  # noqa: E402
from swarmiq.agents.evaluator import EvaluatorAgent  # noqa: E402
from swarmiq.agents.visualization import VisualizationAgent  # noqa: E402

planner_agent = PlannerAgent(llm_config=llm_config)
summarizer_agent = SummarizerAgent(llm_config=llm_config)
conflict_resolver_agent = ConflictResolverAgent(llm_config=llm_config)
synthesizer_agent = SynthesizerAgent(llm_config=llm_config)
evaluator_agent = EvaluatorAgent(llm_config=llm_config)
visualization_agent = VisualizationAgent()

logger.info("All agents instantiated.")


def literature_agent_factory() -> LiteratureAgent:
    """Create a fresh LiteratureAgent with the shared KnowledgeStore."""
    return LiteratureAgent(
        knowledge_store=knowledge_store,  # type: ignore[arg-type]
        tavily_api_key=config.tavily_api_key,
        serpapi_key=config.serpapi_key,
    )


# Instantiate orchestrator


from swarmiq.core.orchestrator import SwarmOrchestrator  # noqa: E402

orchestrator = SwarmOrchestrator(
    planner=planner_agent,
    literature_agent_factory=literature_agent_factory,
    summarizer=summarizer_agent,
    conflict_resolver=conflict_resolver_agent,
    synthesizer=synthesizer_agent,
    evaluator=evaluator_agent,
    visualization=visualization_agent,
)
logger.info("SwarmOrchestrator instantiated.")

# Instantiate export module


from swarmiq.export.exporter import ExportModule  # noqa: E402

export_module = ExportModule()
logger.info("ExportModule instantiated.")

# Launch Application


from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from swarmiq.api import create_router
from pathlib import Path

app = FastAPI()

# Mount API
api_router = create_router(orchestrator, session_store)
app.include_router(api_router)

# Mount Premium Frontend
public_dir = Path(__file__).resolve().parent / "public"

@app.get("/")
async def serve_index():
    return FileResponse(public_dir / "index.html")

app.mount("/", StaticFiles(directory=str(public_dir)), name="public")

if __name__ == "__main__":
    logger.info("Starting SwarmIQ via Uvicorn (Frontend: http://127.0.0.1:8000/)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
