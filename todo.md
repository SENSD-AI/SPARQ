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