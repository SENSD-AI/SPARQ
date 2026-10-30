# SPARQ Chapter: Context and Remaining Work

## Completed Sections

✓ §1 Motivation (218 words)
✓ §2 System Overview (revised to emphasize planner–executor core; router/aggregator are scaffolding)
✓ §3 Planner: Domain-Grounded Decomposition
✓ §4 Executor: Parallel Execution

**Remaining:**
- §5 Results (needs empirical runs and findings)
- §6 Limitations (drafted; three mechanisms-anchored failure modes)

## Key Design Decisions Locked In

### Core Framing
- SPARQ's **core is planner–executor**; router, aggregator, saver are engineering scaffolding
- The section emphasizes the *why*, not the *how* — implementation detail defers to the accompanying CS paper
- Writing style: claim-first, no meta-labels ("the design claim is…"), no bullet points, formal academic prose

### Register Discipline
- No feature enumeration (no "seven intervention modes" style inflation)
- No manufactured vices to set up rescues
- Contrasts are named once, at the point of maximum legibility
- Components are named only when the name carries a design argument

### The Three Main Claims
1. **Planner (§3):** Externalizing domain knowledge (datasets + epidemiological meaning) into a versioned, auditable artifact makes decomposition legible and correctable
2. **Executor (§4):** Scheduling from declared dependencies enables parallel execution while namespace reconciliation preserves state across branches
3. **Overall (§2):** Separating planning from execution prevents the single-agent failure mode of conflating dataset selection with code-writing

### Citations
- **Wang et al., 2023** (Plan-and-Solve, ACL) — cited in §2 for the lineage of plan→execute separation
- **Kim et al., 2024** (LLMCompiler, ICML) — cited in §2 for planner-as-DAG, executor-as-scheduler with parallel task dispatch; extends from stateless function calls to stateful data analysis

## Content Not Yet Written

### §5 Results
**What it needs:**
- Representative query examples SPARQ was run on
- The findings SPARQ surfaced
- Contrast with what a manual workflow would have missed or produced
- Empirical demonstration that SPARQ surfaces non-obvious epidemiological insights
- Format: narrative with tables/figures, not tutorial walkthrough; the point is what the *system produced*, not how to use it

**Do NOT include:** System metrics, latency numbers, cost comparisons. This is a case study, not a systems paper.

### §6 Limitations
**Already drafted with three mechanism-anchored failure modes:**
1. Manifest incompleteness: datasets absent from data-summaries are invisible to the planner
2. Code correctness: executor-generated code can be analytically plausible but statistically wrong; passes execution without error; requires human review
3. Declared dependencies: undeclared dependencies allow steps to race; parallelization only as sound as the plan's declared edges

**Note:** These are honest to the architecture; they're not generic limitations stated after the fact. They flow directly from the design choices made in §3 and §4.

## Audience and Tone Reminders
- **Audience:** Researchers and public-health professionals familiar with data science, *not* systems ML researchers
- **Tone:** Formal academic prose; no jargon inflation; concepts should land for domain experts unfamiliar with agentic systems
- **Register:** No "namespaces," "workers," "DAGs," or "topological waves" — use "sub-agents," "data they produced," etc.

## File References
- **Planner:** `src/sparq/architectures/v1/nodes/planner.py` — loads data context, emits typed Plan with Step objects
- **Executor:** `src/sparq/architectures/v1/nodes/executor.py` — topological scheduling, parallel sub-agents via asyncio.gather, namespace merging per step
- **REPL/Namespace:** `src/sparq/tools/python_repl/executor.py`, `namespace.py` — subprocess-per-execution, pickle-based namespace persistence
- **Data Context:** `data/data_summaries_short.json` (not `data_manifest.json`) — injected into planner prompt; schema + domain meaning

## Notes for Another Device/Instance
- The chapter is part of a Springer case-studies volume: "Autonomous Intelligence in Epidemiological Surveillance: From Language Models to Agentic Systems"
- The chapter has already covered transformers, the agentic paradigm (messages, tool use, structured output), and agentic systems in healthcare before this SPARQ section
- An accompanying peer-reviewed CS paper will carry detailed results and mechanism descriptions (parallel scheduling, namespace reconciliation, REPL subprocess design, etc.)
- This chapter draft is **not yet submitted**; it's a working version for review and refinement
