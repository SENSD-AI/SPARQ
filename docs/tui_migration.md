# Turning SPARQ into a Terminal Chat App (TUI)

Related: [docs/web_ui_migration.md](./web_ui_migration.md)

## Context

Same starting point as the web-UI migration: `__main__.py` is a single blocking run (parse args →
build `Agentic_system` → `asyncio.run(agentic_system.run(user_query))`), with no conversation loop,
message history, or checkpointer. Two ways to get to a chat-style TUI, depending on whether it
reuses `langgraph dev` as a backend or runs fully in-process.

## Option A — TUI as a thin client of `langgraph dev` (recommended)

If the [web UI migration](./web_ui_migration.md) steps 1–6 are done anyway (module-level graph
factory, `langgraph.json`, checkpointer via the dev server, `messages`-shaped `State`), a TUI is
just another frontend hitting the same local API server — no custom conversation-loop or
checkpointer code needed, since `langgraph dev` already provides threads/persistence/streaming.

1. `uv add langgraph-sdk`.
2. `uv add textual` (or `prompt_toolkit` for something lighter than a full widget/layout framework).
3. Run `langgraph dev` as the backend process (or spawn it from the TUI app on startup).
4. In the TUI: `client = get_client(url="http://127.0.0.1:2024")`, create a thread once per chat
   session, then for each user message call
   `client.runs.stream(thread_id, graph_id, input={"messages": [...]}, stream_mode="messages")`
   and render events into a scrolling chat panel as they arrive.
5. Wire Textual's `Input` widget to a background `asyncio` task per turn (Textual's own event loop
   handles this) so the UI stays responsive while a turn streams.
6. Handle executor REPL output (stdout, dataframes, plots) — a TUI can't render images inline
   without a terminal image protocol (sixel/iTerm2 via `textual-image`); simplest first pass is to
   show stdout/dataframe text and print saved plot file paths rather than rendering images.
7. Add a `sparq-chat` entry point in `pyproject.toml`, separate from the existing `-t` batch-mode
   `sparq` command.

Tradeoff: requires running `langgraph dev` as a second process alongside the TUI, but avoids
duplicating conversation/checkpoint logic that the web UI already needs.

## Option B — TUI fully in-process (no `langgraph dev` dependency)

Import the compiled graph directly and drive `graph.astream(...)` from inside Textual, owning the
checkpointer/thread/message-history plumbing directly:

1. Add a `messages` field to `State` (`schemas/state.py`), same as web UI step 5, so multi-turn
   follow-ups have context.
2. Add a LangGraph checkpointer (e.g. `MemorySaver`, or `SqliteSaver` for persistence across
   process restarts) keyed by a `thread_id` per chat session; invoke `graph.astream(...)` once per
   user message instead of once per process run.
3. `uv add textual`; replace `Agentic_system.run()`'s `print(chunk)` loop with Textual widget
   updates per `astream` chunk (`stream_mode="updates"` or `"messages"` for token-level streaming
   from the aggregator's LLM call).
4. Run the graph invocation as a background task (`run_worker`/`asyncio.create_task`) so the input
   box stays responsive and a long executor step can be cancelled mid-run.
5. Same REPL-output rendering caveat as Option A, step 6.
6. Decide per-turn output handling: `saver_node` currently writes one `trace.json`/
   `final_answer.json` per full run into a timestamped dir — decide whether to append turns into one
   session dir or write one per turn.
7. New `sparq-chat` entry point, as in Option A.

Tradeoff: single standalone binary, no separate server process — but reimplements what
`langgraph dev` provides for free (threads, persistence, streaming server), so more code to
maintain.

## Recommendation

Option A, since `messages`-shaped state and a chat-capable graph are needed for the web UI
regardless — building the TUI as a second client on top of that avoids duplicating
conversation/checkpoint logic in two places. Option B only makes sense if the TUI must be a fully
standalone binary with no external server process.
