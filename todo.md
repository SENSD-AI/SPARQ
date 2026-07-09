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
- [ ] Fix `run_dir` collision — `PathSettings.set_run_dir` computes `run_dir` once at `V1Settings`/`Agentic_system` construction, so looping `run()` over multiple questions on one instance overwrites the same `trace.json`/`final_answer.json`. Instantiate a fresh `Agentic_system()` per question.
- [ ] Decide how to handle `follow_ups` in `data/Q_dataset.json` — they're phrased as continuations ("If high risk -> ...") implying multi-turn context, but `Agentic_system.run()`/`State` has no conversation memory today. Either run them as standalone questions (simplest, doesn't test real multi-turn behavior) or add multi-turn support first.
- [ ] Build batch runner over `data/Q_dataset.json`: save per-question full detail via existing `trace.json`/`final_answer.json`, plus a consolidated `experiments/00_results/results.jsonl` (append-per-question, one line per question: `question, grade, answer, run_id, trace_path, timestamp`) for easy grading/analysis.