# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Coding Principles (from AGENTS.md)

1. **Think before coding** — State assumptions explicitly, surface ambiguities, ask rather than guess.
2. **Simplicity first** — Minimum code to solve the problem; no speculative features or abstractions.
3. **Surgical changes** — Only touch what was requested; do not improve unrelated code; clean up only orphans your own changes create.
4. **Goal-driven execution** — Transform tasks into verifiable goals, state a brief plan with checkpoints before implementing.

## Commands

```bash
# Run the application (interactive)
uv run sparq

# Run with the predefined test query
uv run sparq -t

# Run tests
uv run python -m unittest          # all tests
uv run python -m unittest tests.tools.python_repl.test_executor   # single test module

# Run via scripts
./scripts/run.sh       # local run
./scripts/run_tests.sh # test suite
./scripts/run_docker.sh # Docker build+run (prod or dev mode)

# Download datasets from HuggingFace Hub (zayanhugsAI)
uv run python -m sparq.utils.download_data
```

**Package manager**: `uv` (Python 3.13.3). Use `uv add <package>` to add dependencies; `uv.lock` is committed for reproducibility.

**Test framework**: `unittest` (not pytest). Tests are in `tests/` with pattern `test_*.py`.

## Architecture

SPARQ is a multi-agent LangGraph pipeline for Salmonella epidemiological research. The graph processes a user's natural-language query through these nodes:

```
START → router → [True] → planner → executor → aggregator → saver → END
                [False] → END
```

### Node Responsibilities

| Node | File | Role |
|------|------|------|
| **Router** | `architectures/v1/nodes/router.py` | Classifies query: needs data analysis (True) or direct answer (False) |
| **Planner** | `architectures/v1/nodes/planner.py` | ReAct agent; reads `data/data_manifest.json` and produces a structured `Plan` (list of `Step` objects) |
| **Executor** | `architectures/v1/nodes/executor.py` | ReAct agent; iterates over plan steps, runs Python code via the custom REPL |
| **Aggregator** | `architectures/v1/nodes/aggregator.py` | Plain LLM call that synthesizes executor results into a final narrative |
| **Saver** | `architectures/v1/nodes/saver.py` | Writes `trace.json` and `final_answer.json` to a timestamped `output/` directory |

The graph is constructed in `src/sparq/architectures/v1/system.py` (`Agentic_system` class) and executed asynchronously.

### Python REPL Subsystem (`tools/python_repl/`)

The most sophisticated component. The executor agent runs Python code through a subprocess-based REPL with:
- **Namespace persistence** across steps via pickling (modules tracked separately since they can't be pickled)
- **Subprocess isolation** using `multiprocessing.spawn` per execution
- **Timeout enforcement** and stdout/stderr capture
- **Whitelisted package auto-installation** on `ModuleNotFoundError` (allowed packages defined in `package_config.toml`)
- **AST-based last-expression evaluation** (`ast_utils.py`) to return the value of the final expression

Entry point for tool use: `tools/python_repl/python_repl_tool.py` (`@tool` decorator).

### Key Schema Types (`schemas/`)

- `State` (TypedDict) — LangGraph state flowing through the graph; defined in `schemas/state.py`
- `Plan`, `Step` — structured planner output in `schemas/planning.py`; `Step` records dataset choices, rationale, task types, and dependencies
- `Router`, `StepResult` — router and executor results in `schemas/planning.py`
- Run contracts live in `schemas/runs.py`; batch manifests live in `schemas/evaluation.py`

### Settings and Configuration

`Settings` class in `src/sparq/settings.py` is the single source of truth for paths, config, and env. It loads:
1. `src/sparq/default_config.toml` (baked into the package)
2. `config/config.toml` (user overrides: LLM model + provider per node, test query)
3. API keys from `~/.secrets/.llm_apis` or `.env`

Supported LLM providers: Google GenAI (default), OpenAI, AWS Bedrock, Ollama, OpenRouter. Provider/model is configured per-node in `config/config.toml`.

LLM instantiation is centralized in `utils/helpers.py` (`get_llm` function).

### LangSmith Tracing

Set `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2=true`, and `LANGCHAIN_PROJECT` in `.env` to enable LangSmith tracing.

### Data

Datasets are downloaded from HuggingFace Hub (`zayanhugsAI`) and cached locally. The data manifest at `data/data_manifest.json` describes available datasets and is the primary reference the Planner uses to decide which datasets to load.
