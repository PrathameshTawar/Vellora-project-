# SwarmIQ v2 Improvements TODO

## High Priority (Core Functionality) ✅ In Progress
- [x] Create root files: README.md, pyproject.toml, .gitignore, .env.example, LICENSE
- [x] Update swarmiq/config.py: Use Pydantic BaseSettings for .env loading
- [x] Update swarmiq/main.py: Use dynamic config (MODEL_NAME, TEMPERATURE); stricter API checks
- [ ] Fix swarmiq/ui/app.py: Implement explainability/conflict panels, citation clicks

## Medium Priority (Developer Experience)
- [ ] Create requirements.in + pip-compile for pinned deps
- [ ] Create requirements-dev.txt: ruff/black/mypy/pre-commit
- [ ] Create .pre-commit-config.yaml + install instructions
- [ ] Add Sphinx docs/ folder

## Low Priority (Production/Extras)
- [ ] Dockerfile + docker-compose.yml
- [ ] GitHub Actions CI (.github/workflows)
- [ ] Multi-LLM support in config
- [ ] Coverage badge + tox

## Follow-up Commands
```bash
pip install pip-tools pre-commit
pip-compile requirements.in requirements-dev.in
pip install -r requirements.txt -r requirements-dev.txt
pre-commit install
ruff check .
mypy .
pytest
python swarmiq/main.py  # Test with .env
```

Progress tracked here; updates after each step.

