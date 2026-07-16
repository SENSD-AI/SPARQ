# Docs Index

- **[improvements.md](improvements.md)** — master v2 roadmap; priority-ranked list of architectural improvements, links out to the design docs below.

## Design docs — for specific improvements

- [artifact_organizer.md](artifact_organizer.md) — post-executor node that organizes output files (improvement #12)
- [multi_turn_support.md](multi_turn_support.md) — multi-turn chat support for `State` and the v1 graph
- [parallel_execution.md](parallel_execution.md) — step dependency tracking and concurrent execution of plan steps (improvement #1)
- [subtask_parallelism.md](subtask_parallelism.md) — sub-task parallelism within a worker via `async_subagent_tool` (improvement #16)
- [weather_tool.md](weather_tool.md) — adding a weather-lookup tool; geocoding and date-granularity complexities across datasets
- [why_not_deepagents.md](why_not_deepagents.md) — rationale for not adopting LangGraph's `deepagents` module

## Architecture — how existing subsystems work

- [repl_architecture.md](repl_architecture.md) — how the executor invokes Python and how the REPL subsystem works end-to-end
- [repl.md](repl.md) — REPL subsystem quirks and non-obvious constraints
- [concurrency.md](concurrency.md) — notes on running multiple `sparq` invocations concurrently

## Migration guides — adopting a new dependency or surface

- [migration_langchain_langgraph.md](migration_langchain_langgraph.md) — upgrading the LangChain/LangGraph pin set to 1.x
- [tui_migration.md](tui_migration.md) — turning SPARQ into a terminal chat app
- [web_ui_migration.md](web_ui_migration.md) — turning SPARQ into a web chat app (LangGraph Studio / Agent Chat UI)

## Setup

- [aws/README.md](aws/README.md) — AWS Bedrock provider setup

## Other

- [progress_summaries/](progress_summaries/) — periodic dev progress summaries
- [chapter_sparq_architecture.md](chapter_sparq_architecture.md) — manuscript draft, not an engineering doc
