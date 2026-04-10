# SwarmIQ v2 🧠📚

[![Tests](https://github.com/yourusername/swarmiq/actions/workflows/test.yml/badge.svg)](https://github.com/yourusername/swarmiq/actions)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/yourusername/swarmiq/actions)

**Multi-Agent AI Research Assistant.** Automates literature review, claim extraction, conflict resolution, synthesis, evaluation, and visualization.

![Demo Screenshot](docs/screenshots/dashboard.png)
*Gradio UI: Activity streaming, explainability panels, exports (MD/PDF/PPT/JSON).*

## 🚀 Quick Start

1. **Clone & Setup**
   ```bash
   git clone https://github.com/yourusername/swarmiq.git
   cd swarmiq
   cp .env.example .env
   pip install pip-tools
   pip-compile requirements.in requirements-dev.in
   pip-sync requirements.txt requirements-dev.txt
   pre-commit install
   ```

2. **API Keys** (edit `.env`)
   ```
   OPENAI_API_KEY=sk-...
   PINECONE_API_KEY=...
   TAVILY_API_KEY=...
   SERPAPI_KEY=...
   MODEL_NAME=gpt-4o-mini  # or gpt-4o
   TEMPERATURE=0.2
   ```

3. **Run**
   ```bash
   python swarmiq/main.py
   ```
   Open [http://127.0.0.1:7860](http://127.0.0.1:7860)

## 🏗️ Architecture

```
Query → [Planner] → Subtasks
          ↓ parallel
[Literature x3-5 (Tavily/Pinecone)] → [Summarizer] → Claims → [Credibility] → [Resolver] → [Synthesizer] → Report
                                                                   ↓
                                                              [Evaluator] + [Visualizer]
```

- **Agents**: AutoGen + GPT-4o (Planner, Literature, Summarizer, Resolver, Synthesizer, Evaluator, Viz).
- **Core**: Async Orchestrator, JSONSchema validation, ActivityQueue streaming.
- **Storage**: Pinecone (vector DB), SessionStore (query history).
- **UI**: Gradio Blocks (panels, sessions, exports).
- **Tests**: 100% coverage (unit/property/e2e + Hypothesis).

## 📋 Features
- Real-time agent activity feed.
- Explainability: Claim resolutions, credibility/confidence scores.
- Metrics: Coherence, factuality, citation coverage.
- Sessions: History + delete.
- Exports: Markdown/PDF/PPT/JSON (APA/MLA).
- Robust: Schema validation, error recovery.

## 🔧 Development

```bash
ruff check --fix .
mypy .
pytest --cov
pre-commit run --all-files
```

Domain modes: `research`/`business`/`policy`.

## 🛠️ Roadmap
- Multi-LLM (Anthropic).
- Docker deployment.
- Redis caching.
- Auth.

## 📄 License
MIT

