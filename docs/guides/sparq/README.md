# Understanding SPARQ

SPARQ turns a research question into an analysis and a saved report. This guide
builds up the system stage by stage: first the overall architecture, then the
work performed at each stage, then what changes when several runs overlap.
Read the chapters in order on a first visit. Each chapter continues the same
illustrative question: **How do Salmonella rates compare across Missouri counties?**
The example explains the flow; it is not a validated analysis or a fixed plan
that SPARQ always produces.

1. [Architecture and the information that flows through it](01-architecture.md):
   understand the graph, agents, tools, and shared state before following a run.
2. [Starting, routing, and planning](02-starting-and-planning.md): see how a
   question becomes either a direct answer or a dependency-aware analytical plan.
3. [Execution, Python namespaces, and parallel steps](03-execution.md): follow a
   worker into its tools, learn how variables survive Python calls, and see how
   dependent workers receive those variables.
4. [Synthesis, reports, and saved results](04-reports-and-results.md): follow
   evidence into the final answer, then understand the files and report formats.
5. [Batch evaluation and concurrent runs](05-evaluation-and-concurrency.md): wrap
   the familiar single run in an evaluation and distinguish the different levels
   of concurrency and their limits.

For installation, credentials, configuration, and downloading data, use the
[repository setup instructions](../../../README.md#setup). [Bedrock setup](bedrock.md)
adds provider-specific instructions. External repository utilities have their own
[documentation](../utils/README.md).

These chapters describe the current v1 implementation. Adopted decisions remain
in [ADRs](../../adr/); future designs and historical records remain in
[proposals](../../proposals/) and [progress](../../progress/).
