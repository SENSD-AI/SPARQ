# SPARQ Run Monitor (FastAPI + React)

Distinct from [web_ui_migration.md](web_ui_migration.md), which turns SPARQ into a conversational
chat app via LangGraph Studio / Agent Chat UI. This doc is for a run **observability** dashboard: a
side panel listing run IDs, a main panel showing the input query and the live streaming node updates
for the selected run. Runs are still launched the existing way (CLI, `experiments/00.py` batch
script) — the web app only observes them.

## Context

`Agentic_system.run()` (`src/sparq/architectures/v1/system.py`) currently does:

```python
async for chunk in self.graph.astream(input=input_data, config=..., stream_mode="updates"):
    logger.debug(chunk)
```

Chunks only ever reach loguru's per-run `log.txt` (`logging_config.py`). Nothing persists the query
or per-chunk stream data until the very end — `saver_node` (`nodes/saver.py`) only writes
`trace.json` / `final_answer.json` after the graph completes. A file-based UI needs two small writes
added at the source before there's anything to watch.

## Checkpoint A — instrument the run (surgical, `system.py`)

- On run start (after `run_dir.mkdir`): write `run_meta.json` = `{run_id, query, started_at}` into
  `run_dir`.
- Per chunk in the `astream` loop: append it as one JSON line to `stream.jsonl` in `run_dir`
  (alongside the existing `logger.debug(chunk)` — don't remove that).
- In the `finally` block: update `run_meta.json` with `finished_at` / `status`.
- Factor the existing `pydantic_encoder` out of `nodes/saver.py` into a shared
  `utils/json_utils.py` so both the saver and the new JSONL writer use the same
  (Pydantic-model-aware, `str()`-fallback) encoder — avoids duplication and guarantees a stream
  write never crashes a run on an unserializable object.
- Verify: `uv run sparq -t` produces a well-formed `run_meta.json` + `stream.jsonl` next to
  `trace.json`.

## Checkpoint B — FastAPI backend (new `src/sparq/web/`)

- Background poller scans configured root dir(s) every ~2s for subdirectories containing
  `run_meta.json`, keeps an in-memory index (`run_id` → dir, query, status, timestamps). Polling,
  not inotify — simpler and robust across the separate CLI/batch processes actually producing the
  files.
- `GET /api/runs` — list of run summaries, newest first.
- `GET /api/runs/{run_id}` — meta + full `stream.jsonl` replay (for finished/historical runs).
- `GET /api/runs/{run_id}/stream` — SSE endpoint that tails `stream.jsonl` from EOF and pushes new
  lines live; closes once `run_meta.json` shows `finished_at`.

**Open item:** default watch root is `output/` (matches single-run `Settings.paths.output_dir`),
but `experiments/00.py` writes to a separate `00_results/` at cwd — outside that root. Plan is to
make watch roots a configurable list (default `[output/]`, extra roots via flag/env) rather than
touching `00.py`, unless it's preferable to just point `00.py` at `output/` too.

## Checkpoint C — React frontend (Vite, `web/`)

- Sidebar: polls `/api/runs`, renders run cards (short id, truncated query, status badge,
  timestamp); click selects a run.
- Main panel: header shows full query from `run_meta`; body replays `stream.jsonl` via the REST
  call, then opens `EventSource` on `/stream` only if the run is still `running`, appending chunks
  live as collapsible per-node cards, auto-scrolling.
- Dev: Vite proxies `/api` → FastAPI on `:8000`; prod: FastAPI serves the built static bundle.

## Checkpoint D — end-to-end verification

- Start a CLI run and a 2-3 question `experiments/00.py` batch concurrently, confirm the sidebar
  shows all of them independently updating live, and that closing/reopening the browser still shows
  finished runs from disk.
