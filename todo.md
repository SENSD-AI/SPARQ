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

# Ablation Study Support (system.py node toggles)
- [ ] `_build_graph()`: wire router on/off and aggregator on/off independently (simple bypass — router off forces route=True; aggregator off is already safe since `aggregator_node` returns a placeholder on empty `results`)
- [ ] `_build_graph()`: planner/executor are coupled, not independent — only 3 valid states (both on / planner off / executor off); raise a clear error on "both off" (nothing would run)
- [ ] New node `executor_self_planning_node` (+ new prompt) — used when planner is off; executor plans its own steps instead of consuming `state.plan`
- [ ] New node `planner_and_executor_node` (+ new prompt) — used when executor is off; planner executes as it plans instead of just emitting a `Plan`
- [ ] Make `planner_node()`/`executor_node()` dispatchers: always wire both under the same node names regardless of ablation config, each checking config internally to pick default/self-planning/self-executing/no-op behavior (keeps graph shape identical across arms for LangSmith trace comparability); no-op path returns `{}` (passthrough) when a node's job was absorbed by its neighbor
- [ ] `_load_prompts()`: support a per-node prompt override (the `roles` ablation dimension) via a dict lookup instead of always loading the fixed `*_message.txt` files

# Concurrency Safety (experiments/00.py batch runs)
- [x] `package_manager.py`: class-level `threading.Lock` around `install_package`'s check-then-install section — fixes a race when two concurrent questions hit a missing package at the same time (tool calls run in `run_in_executor`'s thread pool under `agent.ainvoke()`, confirmed via `langchain_core/tools/base.py`)
- [x] `experiments/00.py`: cap concurrent questions with `asyncio.Semaphore(MAX_CONCURRENT_RUNS)` via a `run_bounded()` wrapper around each `agentic_system.run(question)` call before `asyncio.gather`

# Logging (replace `print()` for concurrent batch runs)
Superseded the original stdlib-`logging`/`contextvars.ContextVar`/`FileHandler` plan below with loguru instead — same goals, different mechanism (`logger.contextualize()` instead of a manual `ContextVar`; `logger.add(sink, filter=...)` instead of a `FileHandler`).
- [x] Replace runtime-path `print()`/`rich.print()` calls in `executor.py`, `aggregator.py`, `planner.py`, `system.py`, `package_manager.py` with loguru `logger` calls (dev-only `test_*`/`__main__` blocks intentionally left as `print()`; CLI/setup scripts — `settings.py`, `setup.py`, `download_data.py`, `helpers.py`'s table display — intentionally untouched, not part of the concurrent-run interleaving problem)
- [x] Tag each run with `run_id` via `logger.contextualize(run_id=...)` in `run_log_context()` (`logging_config.py`), entered at the top of `Agentic_system.run()`
- [x] Tag each node's logs with `node=<name>` via `logger.contextualize(node=...)` inside each node function body (`router_node`, `planner_node`, `executor_node`, `aggregator_node`, `saver_node`)
- [x] Attach a per-run loguru file sink (`run_dir/log.txt`, filtered by `run_id`) instead of stdout, added/removed by `run_log_context()`
- [x] `system.py`: astream loop logs each chunk via `logger.debug(chunk)` (kept off console — INFO — since it's a raw per-node state dump, redundant with the human-readable progress lines nodes already log at INFO)
- [ ] Known gap, left out of scope: `tools/python_repl/executor.py`'s `_target` (the `multiprocessing.spawn`'d code-execution subprocess) still has one bare `print()` on a rare failure path (writing the result file) — `contextualize()` can't cross a real process boundary, so it doesn't get run/node tagging or file routing

# Eval Pipeline (gold-standard human review)
- [ ] Curate/refine gold query set (start from the 11 in `experiments/00.py`; get domain-expert input on representativeness)
- [ ] Build a blinded, position-randomized pairwise comparison review tool (Streamlit/Gradio) — reviewer picks a winner between two `final_answer.json`s for the same query; no auth/infra needed; works for both domain experts and lab members
- [ ] Define `GoldQuery` (id, query, difficulty) and `RunRecord` (run_id, query_id, ablation_config, response, token_out, models, cost, timing, sparq_judge_score, sparq_judge_review) Pydantic types in `schemas/eval_schemas.py`
- [ ] `AblationConfig` type: `nodes: Dict[str,str]` + `roles: Dict[str,str]` (free-form values — 'yes'/'no' toggle or a variant/prompt-reference label)
- [ ] Store pairwise comparisons as `(query_id, run_id_a, run_id_b, winner, reviewer_id, reason, timestamp)`; aggregate via win-rate or Bradley-Terry per config pair (not raw counts)
- [ ] Double-review a subset of pairs (domain expert + lab member, or two domain experts) to report inter-rater agreement
- [ ] Automated `sparq_judge_score`/`sparq_judge_review` (LLM-judge) as a separate, cheaper-to-scale signal — validate it against human pairwise win-rates before trusting it alone