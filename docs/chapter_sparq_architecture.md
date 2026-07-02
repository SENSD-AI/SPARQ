<!-- Draft for the "SPARQ's architecture and workflow" subsection. Copy into the manuscript under the existing "SPARQ: A Multi-agent System for Pathogen Analysis, Reasoning and Querying" heading, replacing the "At its core SPARQ" stub. -->

## SPARQ's architecture and workflow

SPARQ is realized as a graph of five specialized agents, each responsible for one stage of the
analytical workflow: a *router*, a *planner*, an *executor*, an *aggregator*, and a *saver*. Every
query enters at the router and, depending on its content, either passes through the full
analytical pipeline or is answered directly, as illustrated in Figure []. This division of labor is
the structural answer to the limitations of both the manual analyst workflow and the undifferentiated
general-purpose agent described above: each node owns a single responsibility, and the boundary
between nodes is itself a checkpoint at which the system's intermediate reasoning becomes inspectable.

The **router** is the system's gatekeeper. It issues a single structured-output call that classifies
the incoming query as either requiring data analysis or answerable directly from the model's own
knowledge. Its governing heuristic is deliberately conservative: any query touching Salmonella,
food safety, or the socioeconomic indicators in SPARQ's datasets is always deferred to the planner,
and only queries clearly outside this domain are answered directly. This asymmetry is a considered
tradeoff — a false positive costs an unnecessary pass through the full pipeline, while a false
negative would mean a genuine analytical question gets a shallow, unverified answer. The router's
decision is binary and is the only branch point in the graph; the conditional edge it drives sends
the query either to the planner or directly to the saver, bypassing the analytical pipeline entirely
for non-domain queries.

The **planner** addresses the first failure mode of the manual workflow described earlier: the
undocumented, ad hoc selection of which datasets bear on a given question. Rather than relying on
the model's parametric knowledge of what data might exist, the planner is given an explicit,
machine-readable index — a manifest enumerating every dataset available to the system together with
a summary of its structure (the columns each dataset contains and their types) — injected directly
into its context. Against this index, the planner produces a typed plan: an ordered list of steps,
where each step names the datasets it draws on, states its rationale, and is tagged with one or more
task types drawn from a fixed vocabulary (data exploration, aggregation, statistical analysis,
correlation, and so on). Because this plan is a structured object rather than free text, the rationale
for selecting one dataset over another — the very thing the manual workflow leaves undocumented — is
captured as a first-class, auditable field rather than existing only in an analyst's head.

The **executor** is where the plan becomes analysis. It is implemented as a tool-using ReAct agent:
for each step in the plan, it reasons about how to accomplish the step, acts by invoking tools —
dataset loading, file discovery, and a Python code-execution tool — observes the result, and repeats
until the step is satisfied, at which point it produces a typed summary of what it found before
moving to the next step. Crucially, each step is given not only its own description but a
reconstructed context of everything completed so far, so that a later step can build on the output
of an earlier one without re-deriving it. This is the second pillar promised above — *isolated
execution* — realized concretely: every step's code runs in a sandboxed Python environment rather
than in the surrounding agent process, so an error or a runaway computation in one step cannot
corrupt the orchestrating system itself.

That sandbox is itself worth describing on its own terms, since it is the component that lets the
executor behave like a careful human analyst rather than a one-shot code generator. Each invocation
of the code tool runs in a freshly spawned interpreter, so no stray state from a previous, unrelated
execution can leak in; yet across the steps of a single plan, the variables a step creates — a loaded
dataframe, a fitted model, an intermediate aggregate — persist and remain available to later steps,
giving the executor the same continuity of state a human analyst would have working in an interactive
notebook. The tool also mimics an interactive session in how it reports results: it returns the value
of the final expression in a block of code the way a REPL would, rather than requiring the model to
remember to print everything it wants reported. Because the executor occasionally needs a package
that is not already available, the sandbox can install missing packages on demand — but only from a
fixed, curated allow-list, so the analytical environment can grow to meet a task without becoming an
open door to arbitrary system access. A timeout bounds every execution, ensuring a stalled or
runaway computation cannot stall the pipeline indefinitely.

The **aggregator** is, by design, the simplest node in the system: a single plain-language call, with
no tools and no structured output, that reads the original query alongside every result the executor
produced and writes a narrative synthesis. This simplicity is intentional. Separating synthesis from
analysis means the agent writing the final report is not the same agent — and not working from the
same step-by-step vantage point — as the one that produced the underlying results, giving the
synthesis a degree of independence from the process that generated what it is summarizing.

Finally, the **saver** closes the loop on reproducibility. It writes two artifacts for every run: a
complete trace recording the query, the router's decision, the full plan, the dataset index it was
built from, every step's results, and the final answer, suitable as a full audit trail of the run;
and a concise file containing only the original query and the final answer, suitable for direct
consumption by an end user uninterested in the analytical machinery behind it. Where the manual
workflow described earlier leaves the rationale for a dataset choice or a synthesis step recoverable
only from an analyst's memory, every such decision in SPARQ exists, by construction, as a field in
this trace.

Taken together, these five nodes supply exactly the three structural elements promised above as
the alternative to both manual analysis and undifferentiated general-purpose agents: an explicit,
machine-readable index of what data exists and why a given dataset was chosen (the planner); typed,
inspectable analytical plans rather than free-text reasoning (the planner's output and the executor's
per-step results); and isolated, sandboxed execution of the code those plans require (the executor's
REPL). The result is a system in which the model retains the flexibility of an open-ended language
model while being constrained, at every stage, to produce an output that can be checked, repeated,
and traced back to the data and reasoning that produced it.
