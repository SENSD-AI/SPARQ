# Parallel Execution (feat/parallel-repl-execution)
- [x] Fix `aggregator.py` — still reads `state.executor_results` (old format); update to consume `state.results: List[StepResult]`
- [x] Clean up per-step namespaces — `cleanup_ns` in `system.py` only removes the main run namespace; add cleanup for `{run_id}_step_{n}` temp files created by workers
- [x] End-to-end test — run `uv run sparq -t` and verify the parallel execution path works correctly
- [x] Remove unused scaffolding — `WorkerState` in `state.py` and `Send`/`END` imports in `executor.py` (leftover from abandoned LangGraph fan-out design)
- [x] Fix `test_executor`/`__main__` in `executor.py` — still calls old `executor_node` signature and references removed `executor_results` field

# Incorporating v2
- [x] Change `AgenticSystemSettings` to `BaseAgenticSettings`
    - [x] Fix imports:
        - [x] `tests/test_settings.py`
    - [x] Change usage pattern of `LLMSettings`. Introduced `BaseLLMSettings` + `BaseAgenticSettings[LLMConfigT: BaseLLMSettings]` generic pattern.

- [x] keep a default config for each architecture.
    - [x] the config file for each architecture should be copied into `USER_CONFIG_DIR/<architecture>/config.toml` on first run (done via `setup.py`).
- [x] `__main__.py` should receive CLI arg that specifies the architecture.
- [x] Setup runs on first import via `__init__.py` sentinel guard.

# Q_dataset experiment (experiments/00.py)
- [ ] Fix `run_dir` collision — `PathSettings.set_run_dir` computes `run_dir` once at `V1Settings`/`Agentic_system` construction, so looping `run()` over multiple questions on one instance overwrites the same `trace.json`/`final_answer.json`. Instantiate a fresh `Agentic_system()` per question. (Also resolved as a side effect of the multi-turn trace/saver restructuring below.)
- [x] Decide how to handle `follow_ups` in `data/Q_dataset.json` — resolved: add multi-turn support first. See `docs/multi_turn_support.md` and the checklist below.
- [ ] Build batch runner over `data/Q_dataset.json`: save per-question full detail via existing `trace.json`/`final_answer.json`, plus a consolidated `experiments/00_results/results.jsonl` (append-per-question, one line per question: `question, grade, answer, run_id, trace_path, timestamp`) for easy grading/analysis.

# Multi-Turn Conversation Support (feat/multi-turn)
Design complete, see `docs/multi_turn_support.md`. Ordered by build dependency.
- [ ] Provision a Postgres instance (e.g. AWS RDS via lab credits) and get a connection string
- [ ] `settings.py`: add Postgres connection string to `ENVSettings` (secret, not `PathSettings`); add namespace directory path
- [ ] `system.py`: wire pooled `AsyncPostgresSaver` into `Agentic_system` (compile-once-reuse, not per-`run()`); accept `thread_id` from caller; call `checkpointer.setup()` once; remove unconditional `cleanup_run(...)`; add `end_session(thread_id)`
- [ ] `router.py`: append `HumanMessage` to `state.messages`; rewrite heuristic from "in-domain" to "needs new computation"
- [ ] `saver.py`: append `AIMessage` to `state.messages`; restructure `trace.json`/`final_answer.json` paths per `thread_id`/turn
- [ ] `planner.py` / `aggregator.py`: pass `state.messages` history into prompts
- [ ] `namespace.py`: deterministic per-`thread_id` paths (not `tempfile.mkstemp`); turn-scoped step keys; seed-from-thread/merge-back-to-thread helpers
- [ ] `executor.py`: adopt turn-scoped step keys; call seed/merge-back helpers at turn boundaries
- [ ] `experiments/00.py`: one `thread_id` per top-level `Q_dataset.json` question, iterate `follow_ups` within it
- [ ] Tests: namespace path durability across restart, `messages` accumulation across two `run()` calls, router classification (recall vs. new-computation), end-to-end follow-up against `Q_dataset.json`