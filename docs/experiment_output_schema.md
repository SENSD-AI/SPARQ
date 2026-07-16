# Experiment Output Schema

Plan for a `pydantic` type capturing one row of experiment output from `experiments/00.py` (and
future experiment scripts), so results can be diffed across ablations/models and fed to a judge
later. Target fields, in order:

| Field | Type | Source |
|---|---|---|
| `run_id` | `str` | generated in `Agentic_system.run()` (`system.py:77`) — currently local-only, not returned; needs to be returned or otherwise captured by the caller |
| `query` | `str` | `data/Q_dataset.json` question text |
| `difficulty` | `int` | `data/Q_dataset.json`'s `grade` field per question |
| `ablation_config` | `dict` | set by the experiment script itself (which nodes/settings vary per run) |
| `response` | `str` | `State.answer`, written to `final_answer.json` by `saver_node` |
| `token_out` | `dict` | LangSmith, see below |
| `models` | `dict` | `V1Settings.llm_config` — per-node provider/model from `config/config.toml`, already available with no extra plumbing |
| `cost` | `dict` | LangSmith, see below |
| `time_started` / `time_ended` / `duration` | `datetime` / `datetime` / `float` | wrap the `agentic_system.run(...)` call in the experiment script |
| `sparq_judge_score` | `dict` | not yet built — no judge exists in this codebase yet |
| `sparq_judge_review` | `str` | not yet built — same as above |

## Token/cost via LangSmith, not manual plumbing

Considered capturing `usage_metadata` off each node's LLM response directly (executor.py:208,
aggregator.py:76 both discard the raw `AIMessage` returned by `agent.invoke()`, which carries
`usage_metadata` from the provider). Rejected in favor of LangSmith, since tracing
(`LANGCHAIN_TRACING_V2`) is already a supported opt-in (`langsmith` is a `pyproject.toml`
dependency) and requires no changes to the graph nodes.

Verified against the LangSmith SDK reference docs (2026-07-16):

- `Client.read_run(run_id)` / `Client.list_runs(...)` return `Run` objects with `total_tokens`,
  `prompt_tokens`, `completion_tokens`, and — computed server-side from LangSmith's pricing map —
  `total_cost`, `prompt_cost`, `completion_cost` directly. No manual summing of prompt+completion
  cost needed.
- Caveat: server-side cost computation only works for models LangSmith has priced. Unclear whether
  this project's non-OpenAI/Anthropic providers (Bedrock, Ollama, OpenRouter, per `CLAUDE.md`) are
  covered — needs a spot check once wired up.
- No SDK helper sums cost across a run tree. A single `agentic_system.run()` call fans out into a
  root run plus per-node child runs (router → planner → executor → aggregator); getting one
  `total_cost`/`total_tokens` per top-level question means calling
  `Client.list_runs(trace_id=root_run_id)` and summing `total_cost` across the children yourself.
- LangSmith's API has a short indexing delay after a run completes before it's queryable — the
  experiment script needs to account for that (poll/retry) rather than querying immediately after
  `run()` returns.

## Prerequisite: `run_id` needs to leave `Agentic_system.run()`

`Agentic_system.run()` generates `run_id = str(uuid.uuid4())` internally and never returns it
(`system.py:72-86`), so today a caller has no way to look up the corresponding LangSmith trace.
Smallest fix: have `run()` return `run_id` (or store it as `self.run_id` after the call), rather
than only passing it into `config={"configurable": {"run_id": run_id}}` and `cleanup_run()`.

## Not building yet

- `sparq_judge_score` / `sparq_judge_review`: no judge/scoring pipeline exists in this codebase.
  Out of scope until that's designed separately.
- Manual token/cost capture inside the graph nodes: superseded by the LangSmith approach above.
