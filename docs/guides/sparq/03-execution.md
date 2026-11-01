# 3. Execution, namespaces, and parallel steps

[Chapter 2](02-starting-and-planning.md) left us with four illustrative steps:
load isolate counts, load population, calculate rates, then plot them. The
executor must turn that plan into evidence while preserving the variables each
calculation needs. Start with one worker before considering several at once.

## Stage 4a: give a step its own worker

For a ready step, the executor creates a worker agent with the executor model,
a step-specific task, dataset context, and tool access. The worker can discover
cached datasets, inspect spreadsheet sheets, load data, execute Python, work
with files, inspect indexed artifacts, and analyze images. `TodoListMiddleware`
provides task-tracking support inside the worker.

The worker alternates model decisions and tool calls until it returns a
structured `StepResult` or fails. That result contains its ID, assigned task,
success flag, findings, generated-file names, and notes. The worker's model
configuration supplies its agent recursion limit.

A result summary tells later stages what the worker found. It does not by itself
carry a live DataFrame. For that we need a namespace.

## Stage 4b: preserve variables across Python calls

A namespace is a dictionary mapping variable names to Python values. After
loading county data, a worker might have `counts` mapped to a DataFrame and `pd`
mapped to the pandas module. Later Python code needs those names to exist.

SPARQ starts a fresh Python process for each code execution. Consequently, it
cannot rely on the previous process remaining alive with its variables in
memory. Instead, a temporary pickle file stores the worker's persistent data.
`get_ns_path()` maps a namespace ID such as `<run-id>_step_1` to that file.

The dataset-loading tool and Python tool receive the same worker namespace
path. `load_dataset` runs its data-reading code through `execute_code()` and
saves the loaded variable there. A later Python tool call must set
`persist_namespace=True` to read and update that persistent namespace. Its
default is `False`, which uses a fresh temporary namespace for that call and
discards it afterward. Having both tools wired to the same path does not
make persistence automatic for every Python call.

This distinction explains a common failure: a successful dataset load can be
followed by `NameError` if the next Python call uses an ephemeral namespace.

## Stage 4c: follow one code execution into the child process

Here is the sequence within `execute_code()`:

```text
worker tool call
  → parent creates a temporary JSON result file
  → spawn a fresh Python interpreter
      → change to the executor workspace
      → load the namespace pickle
      → restore module aliases
      → parse, transform, and execute the submitted code
      → save picklable variables on success
      → write output/error information as JSON
  → parent reads that JSON and returns the tool result
```

Python's [spawn start method](https://docs.python.org/3.13/library/multiprocessing.html#contexts-and-start-methods)
starts a fresh interpreter. The code explicitly requests `spawn`; this behavior
is not left to the operating system's default. The child runs with the executor
output directory as its working directory and requests the noninteractive
`Agg` matplotlib backend with `setdefault`, preserving an existing override.

The parent waits up to the execution timeout, normally 120 seconds per attempt.
If the child is still alive, it terminates that process and returns a structured
timeout error. Missing or unreadable result JSON is also reported as a failed
execution. Process separation supports this lifecycle; the installation
allowlist described below does not establish a general sandbox for Python code.

### How the value of the last expression reaches the model

Consider code ending with `counts.head()`. Normal `exec()` runs statements but
does not return that expression's value. SPARQ parses the code into an abstract
syntax tree and rewrites a final bare expression into an assignment to
`__repl_result__`. After execution it removes that temporary value and formats it
for the tool response. [Python's AST reference](https://docs.python.org/3.13/library/ast.html#ast.Expr)
defines the expression node used here.

The result formatter distinguishes three cases:

| Submitted code ends with | Returned output |
| --- | --- |
| An expression producing a value | The string form of that value |
| An expression producing `None` | Captured stdout, or the string `None` |
| A statement rather than a bare expression | Captured stdout, or an empty string |

A fresh sentinel object distinguishes an absent result from a legitimate `None`.
If a final expression produces a non-`None` value, that value takes precedence
over earlier stdout. Stderr is appended with a `[stderr]:` prefix on success;
on failure, captured stderr is included in the error context.

`redirect_stdout` and `redirect_stderr` send output into in-memory `StringIO`
buffers during execution. See the Python documentation for
[output redirection](https://docs.python.org/3.13/library/contextlib.html#contextlib.redirect_stdout).
`compile_for_repl()` also registers the original source in `linecache` under
`<repl>` so tracebacks can show the submitted line, while compiling the transformed
tree with its original line positions.

### What survives in the namespace

On success, the child removes internal names, tests variables for picklability,
and merges the resulting objects into the existing pickle. Module objects are
stored separately as alias-to-module-name entries in `__modules__` and reimported
on the next execution. Objects that cannot be pickled are recorded by name and
type in `__unpicklable__`; their values do not survive.

Interactive function definitions and lambdas should not be assumed to persist.
Python's [pickling rules](https://docs.python.org/3.13/library/pickle.html#what-can-be-pickled-and-unpickled)
require importable definitions for functions/classes serialized by name. The
namespace update also merges into existing data, so deleting a name in one
execution is not a reliable way to remove its previously saved value.

The JSON response to the parent contains output, errors, success, and a
JSON-safe namespace summary. Complex objects are represented by type strings in
that summary. Their actual values remain in the pickle. This keeps the tool
response serializable, although the dependency-merging code discussed next does
load the saved Python objects in the parent process.

### Import failures and package recovery

If execution returns an import error with a recognized missing package, the
parent checks the package configuration, attempts installation, and reruns the
code in a fresh child. It allows up to five install/re-execution retries.
Installation uses `uv pip` when available and otherwise the environment's
`python -m pip`. The bundled allowlist includes NumPy, pandas, statsmodels,
SciPy, matplotlib, seaborn, and Plotly. Package installation is serialized with
a parent-process lock; this is separate from namespace synchronization.

There are two earlier failure points this loop cannot generally repair. First,
`spawn` may reimport the entry module before submitted code starts. An unavailable
optional dependency in that import chain can prevent result generation. Second,
unpickling a saved object may require its defining package before the new code
can run. The loader therefore creates DataFrames through `execute_code()` rather
than depositing new parent-created DataFrames in the namespace. Keep optional
data-library imports near their use when extending the CLI/tool import chain.

## Stage 4d: connect workers through dependency namespaces

Now return to the whole plan. Steps 1 and 2 have different namespace files:

```text
step 1 namespace: counts
                        ↘
                          step 3 namespace: counts + population → county_rates
                        ↗                                         ↓
step 2 namespace: population                                step 4 namespace
```

Before starting step 3, `merge_namespaces_of_previous_deps()` loads the namespaces
for dependencies 1 and 2 and saves the merged dictionary into step 3's namespace.
Module alias dictionaries are merged separately. Step 4 subsequently inherits
step 3's saved namespace. Dependencies determine which variables are inherited;
there is no single mutable namespace shared by all plan workers.

Merging uses dictionary updates in the order listed in `Step.dependencies`.
If two dependencies use the same variable name, the later dependency's value
wins. Unique, descriptive variable names avoid ambiguous inheritance. The worker
prompt includes dependency result summaries plus inherited variable names and
types; it does not include full values. More than 30 inherited variables triggers
a warning about context size. This threshold counts variables, not their memory
size or analytical importance.

## Stage 4e: schedule independent steps together

The executor repeatedly selects every unfinished step whose dependency IDs are
terminal, invokes those workers with `asyncio.gather()`, and waits for the whole
round before choosing the next one. For our example:

| Round | Ready workers | Why |
| --- | --- | --- |
| 1 | Steps 1 and 2 | Neither has dependencies |
| 2 | Step 3 | Both inputs are now terminal |
| 3 | Step 4 | Step 3 is now terminal |

The barrier is per round: a dependent step does not start immediately when its
own inputs finish if another worker in the same round is still running. This is
concurrent worker scheduling; synchronous tools or model calls can still block
parts of the execution. The source does not promise that every operation runs
in parallel or impose a separate worker-count cap inside this executor loop.

A worker exception becomes a failed `StepResult`. The executor allows up to
three worker attempts before retaining the failure as terminal. A terminal
failed dependency still permits downstream steps to become ready; its failure
flag and notes are available in their context. If unfinished steps exist but
none can become ready, the executor raises an unresolved-dependency error.

Per-step namespaces avoid different workers writing the same namespace file in
the normal scheduler. The same namespace file has no general read/write lock,
however; concurrent calls within one worker that modify it can race. Workers
also share the executor output directory. Distinct variable storage does not
prevent two workers from choosing the same output filename.

## Stage 4f: collect evidence for synthesis

After each round, `organize()` moves generated files into `figures/`, `tables/`,
`docs/`, or `other/` under `executor/`, according to extension. `index.json` maps
artifact IDs to relative paths. An existing destination is not overwritten; a
filename collision raises an error. Workers should find organized files through
the index rather than assuming an earlier unorganized path is still current.

The executor updates completed IDs, structured results, and artifact paths in
state, then repeats until all steps are terminal. Python namespaces remain
available during the run and are removed by `cleanup_run(run_id)` in the outer
graph execution's `finally` block. Saved evidence files remain on disk.

## Where this is implemented

- [Worker scheduling and dependency merges](../../../src/sparq/architectures/v1/nodes/executor.py)
- [Python tool and persistence switch](../../../src/sparq/tools/python_repl/python_repl_tool.py)
- [Child execution and package-retry loop](../../../src/sparq/tools/python_repl/executor.py)
- [Namespace paths and cleanup](../../../src/sparq/tools/python_repl/namespace.py)
- [Expression rewriting and compilation](../../../src/sparq/tools/python_repl/ast_utils.py)
- [Package installation policy](../../../src/sparq/tools/python_repl/package_manager.py)
- [Dataset loading](../../../src/sparq/tools/data_discovery_tools.py)
- [Artifact organization](../../../src/sparq/architectures/v1/artifacts.py)

The linked Python sections were read for the process, serialization, expression,
and output-redirection explanations. SPARQ-specific ordering, limits, and
behavior above come from inspection of the linked implementation; the county
example is illustrative rather than an executed validation.

Next: [turn the collected evidence into reports and saved results](04-reports-and-results.md).
