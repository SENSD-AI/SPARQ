# Turning SPARQ into a Web Chat App (LangGraph Studio / Agent Chat UI)

Reference: [LangGraph UI / local development](https://docs.langchain.com/oss/python/langgraph/ui#local-development), [LangGraph Studio](https://docs.langchain.com/oss/python/langgraph/studio)

## Context

`__main__.py` currently does a single blocking run: parse CLI args → `ENVSettings(verbose=True)` →
build `Agentic_system` → `asyncio.run(agentic_system.run(user_query))`, which builds the graph and
drives `graph.astream(...)` itself, printing chunks via `rich.print`. There is no conversation loop,
no message history, and no checkpointer.

Rather than hand-rolling a custom chat frontend (TUI or otherwise), `langgraph dev` provides a local
API server with threads, an in-memory checkpointer, and streaming built in, and either **LangGraph
Studio** (browser UI at smith.langchain.com pointed at the local server) or **Agent Chat UI** (a
self-hosted Next.js app) can serve as the chat frontend. This offloads conversation-loop/thread/
streaming plumbing to LangGraph instead of building it by hand.

## Tasks

1. **Make the graph importable without the CLI**
   - `ENVSettings(verbose=True)` and graph construction currently only happen inside `main()`.
     `langgraph dev` imports the module directly and calls a factory/graph object — no `argparse`.
   - Add `src/sparq/graph.py` exposing either a compiled `graph` object or a
     `def make_graph(config=None): ...` factory that runs `ENVSettings()` +
     `Agentic_system()._get_node_definitions()` + `_build_graph()` and returns `self.graph`,
     without depending on CLI args.
   - Split `Agentic_system.run()` (currently builds the graph *and* drives `astream` in one method)
     so the dev server can own the streaming/serving loop instead of the current
     `for chunk in astream: print(...)`.

2. **Add `langgraph.json`**
   - At repo root, pointing `graphs.<name>` at the module:attribute from step 1, plus
     `dependencies` (`.`) and `env` (`.env`) fields.

3. **Add the CLI dependency**
   - `uv add "langgraph-cli[inmem]"` (dev-only — not needed at runtime for the packaged app).

4. **Checkpointer**
   - Don't pass a custom checkpointer to `.compile()` — `langgraph dev` injects persistence
     (threads/state) itself. Leave `compile()` as-is.

5. **Chat-shaped state** (determines UX quality)
   - `State` (`schemas/state.py`) currently has one-shot `query`/`answer` fields — fine for
     Studio's generic form/state-inspector view, but **Agent Chat UI expects a chat-shaped graph**
     with `messages: Annotated[list[AnyMessage], add_messages]`.
   - For real chat bubbles + multi-turn follow-ups (not just Studio's raw state JSON), add a
     `messages` field and have router/aggregator read the latest human message / append the AI
     answer to it, rather than only `query`/`answer`. This is the conversational-memory work that's
     needed regardless of frontend choice — it's just now consumed by LangGraph's UI instead of a
     hand-rolled client.

6. **Verify the executor's subprocess REPL under the dev server process**
   - It spawns via `multiprocessing.spawn`; should be independent of how the parent graph is
     invoked, but worth a smoke test since it's the most stateful/fragile part of the system.

7. **Pick the frontend**
   - **LangGraph Studio**: zero frontend code, works immediately once `langgraph dev` is running,
     requires a (free) LangSmith account/API key, generic state-inspector/chat hybrid view.
   - **Agent Chat UI** (`npx create-agent-chat-app`): self-hosted Next.js app, no LangSmith account
     needed, proper chat-bubble UX, but only looks right once step 5 (`messages` state) is done.

8. **Run & test**
   - `langgraph dev` from repo root → connect via the Studio URL, or point Agent Chat UI's
     deployment URL at `http://127.0.0.1:2024`.

## Open decision

Whether to prioritize step 5 (`messages`-shaped state, needed for real chat/follow-up UX) before or
after step 1 (module-level graph factory + `langgraph.json`) — the latter unblocks running
`langgraph dev` at all, the former determines whether the resulting UI actually feels like a chat app.
