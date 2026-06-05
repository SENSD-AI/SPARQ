# Python REPL — Architecture and Execution Flow

This document explains how sparq's executor node invokes Python and how the REPL subsystem works end-to-end. For quirks and past bugs, see [repl.md](repl.md).

---

## End-to-end call path

```
                      Figure 1.  SPARQ pipeline: query → answer
    ═══════════════════════════════════════════════════════════════════

    user query
         │
         ▼
   ┌───────────┐   False
   │  Router   │──────────────────────────────────────────────────────┐
   └─────┬─────┘                                                      │
    True │                                                            │
         ▼                                                            │
   ┌───────────┐                                                      │
   │  Planner  │  reads data_manifest.json                            │
   └─────┬─────┘  emits Plan { steps: [Step, …] }                    │
         │                                                            │
         ▼                                                            │
   ┌─────────────────────────────────────────────────────────────┐    │
   │  Executor                                    executor.py:53 │    │
   │                                                             │    │
   │  ns_path ← /tmp/<uuid>_ns.pkl                               │    │
   │                                                             │    │
   │  tools ─┬─ load_dataset_tool(ns_path)                      │    │
   │          └─ python_repl_tool(ns_path)  ← same ns_path      │    │
   │                                                             │    │
   │  ┌────────────────────────────────────────────────────┐    │    │
   │  │  for each Step in Plan                             │    │    │
   │  │                                                    │    │    │
   │  │  context = prior results + live namespace vars     │    │    │
   │  │                    │                               │    │    │
   │  │                    ▼                               │    │    │
   │  │         ┌────────────────────┐                    │    │    │
   │  │         │    ReAct Agent     │                    │    │    │
   │  │         │  ┌──────┐  call   │                    │    │    │
   │  │         │  │ LLM  │───────► │                    │    │    │
   │  │         │  │      │◄─result─│                    │    │    │
   │  │         │  └──────┘         │                    │    │    │
   │  │         │   tools ──────────┼──► subprocess      │    │    │
   │  │         │                   │        (Fig. 2)    │    │    │
   │  │         └────────────────────┘                    │    │    │
   │  └────────────────────────────────────────────────────┘    │    │
   └────────────────────────────┬────────────────────────────────┘    │
                                │ executor_results                    │
                                ▼                                     │
                          ┌───────────┐                               │
                          │Aggregator │  synthesises narrative         │
                          └─────┬─────┘                               │
                                │                                     │
                                ▼                                     │
                          ┌───────────┐ ◄────────────────────────────┘
                          │   Saver   │  trace.json  final_answer.json
                          └─────┬─────┘
                                │
                                ▼
                               END
```

Cleanup happens in `system.py:85` — `cleanup_ns(run_id)` deletes the pickle file in a `finally` block.

---

## REPL subsystem layers

| Step | File | Lines | What happens |
|------|------|-------|--------------|
| Tool entry | `python_repl_tool.py` | 7–44 | Closure captures `ns_path`; if `persist_namespace=False`, passes `None` to `execute_code` (ephemeral run) |
| Orchestrate | `executor.py` | 60–120 | Creates temp result file, spawns subprocess, drives auto-install retry loop (max 5), cleans up temp files |
| Subprocess | `executor.py` | 123–181 | `mp.get_context("spawn")` → fresh Python process per execution; `process.join(timeout=120s)`; terminate on timeout |
| Target fn | `executor.py` | 184–295 | Loads namespace, re-imports modules, AST rewrite → exec, capture result, pickle back |
| NS load | `namespace.py` | 51–57 | `pickle.load` from file; returns `{}` on `EOFError` |
| Module handling | `namespace.py` | 66–67 | Modules stored as `{var_name: module.__name__}`; re-imported via `importlib.import_module` in subprocess |
| AST rewrite | `ast_utils.py` | 11–47 | Last bare `ast.Expr` → `__repl_result__ = <expr>` so `exec()` captures its value |
| Linecache | `ast_utils.py` | 75–83 | Source registered under `"<repl>"` so tracebacks show real line numbers |
| Stdout/stderr | `executor.py` | 220–251 | `contextlib.redirect_stdout/stderr` → `io.StringIO`; stderr appended as `[stderr]: ...` |
| NS persist | `executor.py` | 253–264 | `pickle_vars()` skips unpicklable (stored in `__unpicklable__`); merges into existing pickle; writes back |
| Auto-install | `executor.py` | 89–107 | `ModuleNotFoundError` → extract package name → check whitelist → `pip install` → re-execute |
| Whitelist | `package_config.toml` | — | `numpy, pandas, statsmodels, scipy, matplotlib, seaborn, plotly` |
| Schemas | `schemas.py` | 4–18 | Input: `PythonREPLInput(code, persist_namespace)` → Output: `OutputSchema(output, error, namespace, success)` |

```
                   Figure 2.  REPL subprocess execution model
    ═══════════════════════════════════════════════════════════════════

    execute_code(code, ns_path)                          executor.py:60
         │
         │  ┌─────────────────────────────────────────────────────┐
         │  │ Parent process                                      │
         │  │                                                     │
         │  │  result_path ← tempfile.mkstemp()                  │
         │  │  process ← mp.spawn.Process(target=_target)        │
         │  │  process.start()                                    │
         │  │  process.join(timeout=120 s)  ◄── blocks here       │
         │  │                                                     │
         │  │  process.is_alive()?                                │
         │  │      yes → terminate()  →  return TimeoutError      │
         │  │      no  → json.load(result_path) → OutputSchema    │
         │  └─────────────────────────────────────────────────────┘
         │                         │
         │           spawn: fresh Python interpreter
         │                         │
         │                         ▼
         │  ┌─────────────────────────────────────────────────────┐
         │  │ Child process  _target(code, ns_path, result_path)  │
         │  │                                                     │
         │  │  1. load_ns(ns_path)       → namespace {}           │
         │  │  2. re-import __modules__  → restore pd, plt, …    │
         │  │  3. redirect stdout/stderr → io.StringIO            │
         │  │  4. ast.parse(code)                                 │
         │  │     rewrite_last_expr(tree)                         │
         │  │     compile_for_repl(code, tree)                    │
         │  │  5. exec(code_obj, namespace)                       │
         │  │  6. pop __repl_result__ ; read stdout/stderr        │
         │  │  7. pickle_vars → merge → write back to ns_path     │
         │  │  8. json.dump(OutputSchema) → result_path           │
         │  └─────────────────────────────────────────────────────┘
         │
         └─ on ModuleNotFoundError:
              extract pkg → check whitelist → pip install → retry (≤5×)
```

---

## Namespace persistence model

The namespace is a pickle file on disk (`/tmp/tmpXXXX_<run_id>_ns.pkl`). Each subprocess execution:

1. **Loads** the file with `pickle.load` → recovers variables from previous steps.
2. **Re-imports modules** by name from the `__modules__` key (e.g. `{"pd": "pandas"}`) because module objects are not picklable.
3. **Executes** user code inside the loaded namespace dict.
4. **Merges** new/modified variables back: `pickle_vars()` attempts to pickle each variable; unpicklable ones (functions, lambdas) go into `__unpicklable__` and are not persisted.
5. **Writes** the merged dict back to the same file.

The parent process never deserializes user data objects — `OutputSchema.namespace` contains only JSON-safe summaries (complex objects are represented as `"<module.ClassName>"`).

```
                Figure 3.  Namespace state across plan steps
    ═══════════════════════════════════════════════════════════════════

    Shared file: /tmp/<uuid>_ns.pkl

         ┌──── Step 1 spawn ─────┐
         │                       │
    {} ──┤ read → exec() → write ├──► {df: <pkl>,
         │                       │     __modules__: {pd: "pandas"}}
         └───────────────────────┘
                                 │
         ┌──── Step 2 spawn ─────┤
         │                       │
         ┤ read → exec() → write ├──► {df: <pkl>,
         │                       │     result: <pkl>,
         └───────────────────────┘     __modules__: {pd: "pandas"}}
                                 │
         ┌──── Step 3 spawn ─────┤
         │                       │
         ┤ read → exec() → write ├──► {df: <pkl>,
         │                       │     result: <pkl>,
         └───────────────────────┘     plot: <pkl>,
                                       __modules__: {pd: "pandas",
                                                     plt: "matplotlib.pyplot"}}
                                 │
                          cleanup_ns()  ←  pkl file deleted (system.py:85)
```

---

## AST last-expression capture

`exec()` discards the value of bare expressions. The REPL works around this by rewriting the AST before compilation (`ast_utils.py:11–47`):

```python
# User writes:
df.describe()

# AST is rewritten to:
__repl_result__ = df.describe()
```

After `exec`, the result is popped from the namespace with a sentinel to distinguish three cases:
- No bare expression → use stdout only
- Expression returns `None` → output `"None"`
- Expression returns a value → output `str(value)`

`compile_for_repl` also registers the source in `linecache` under filename `"<repl>"` so tracebacks display readable line numbers and source text.

---

## Auto-install pipeline

When the subprocess exits with `ModuleNotFoundError` or `ImportError`, `execute_code` loops up to 5 times:

1. Extract the missing package name by scanning `error.message` for `"No module named 'X'"` against the whitelist.
2. Check `PackageUtils.is_whitelisted(package)` — if not whitelisted, abort.
3. Run `uv pip install <package>` (falls back to `python -m pip install`).
4. Re-spawn the subprocess with the same code and `ns_path`.

The whitelist lives in `package_config.toml`. Blocked stdlib modules (`subprocess`, `os`, `socket`, etc.) are also listed there.

---

## Cross-tool namespace sharing

Both `make_load_dataset_tool(ns_path)` and `make_python_repl_tool(ns_path)` are created with the **same** `ns_path` (`executor.py:76–78`). This means:

- A DataFrame loaded by `load_dataset("file.csv", var_name="df")` is written to the pickle file via `execute_code(code, ns_path=ns_path)`.
- The very next `python_repl_tool` call in the same step reads the same pickle and finds `df` already in the namespace.

No explicit handoff is needed — the shared file is the coordination point.

---

## Step context injection

Before each step, `_build_context(results, ns_path)` (`executor.py:19–50`) constructs a string injected into the ReAct agent's user message:

- **Previous step summaries** — each step's `execution_results` and `misc` fields.
- **Live namespace variables** — reads the pickle and reports `{var_name: type_name}` for every non-dunder key.

This gives the agent full awareness of what data is already loaded and what analysis has already been done, without re-reading any files.

---

## Appendix: the execution core, line by line

`_target()` in `executor.py:220–251` is the heart of every code execution. Here is what each part does and why.

### Capturing stdout and stderr with `io.StringIO`

```python
stdout_buffer = io.StringIO()
stderr_buffer = io.StringIO()
```

`io.StringIO` is an in-memory file-like object — it behaves exactly like an open text file but writes into a string instead of disk. Any code that calls `print()` or writes to `sys.stdout` will go here.

Why not just read `sys.stdout` after execution? Because `exec()` runs code that may call `print` at any point during execution. There is no hook to intercept that after the fact — you have to redirect the stream *before* execution starts.

### `redirect_stdout` and `redirect_stderr`

```python
with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
    ...
```

`contextlib.redirect_stdout` temporarily replaces `sys.stdout` with the object you give it, then restores the original when the `with` block exits. `redirect_stderr` does the same for `sys.stderr`. Both are from Python's standard library `contextlib` module.

The `with` block covers the entire execution — parse, compile, and exec — so any output produced at any of those stages is captured, including errors printed by libraries during import.

### Parsing, rewriting, and compiling

```python
tree = ast.parse(code)
tree, has_result = rewrite_last_expr(tree)
code_obj = compile_for_repl(code, tree)
exec(code_obj, namespace)
```

`ast.parse` turns the source string into an Abstract Syntax Tree — a data structure representing the code's structure, not the text itself. If the code has a syntax error, `parse` raises `SyntaxError` here. That exception falls through to the outer `except` block and is written to the result file like any other error.

`rewrite_last_expr` inspects the final node in the tree. If it is a bare expression (e.g. `df.head()`) rather than a statement (e.g. `x = 5`), the node is transformed into an assignment: `__repl_result__ = df.head()`. This is necessary because `exec()` runs statements and discards expression values — the rewrite turns the value into something we can retrieve. `has_result` is `True` when this transformation happened.

`compile_for_repl` calls Python's built-in `compile()` on the rewritten tree. It also registers the original source string in `linecache` under the filename `"<repl>"`. `linecache` is the module that `traceback.format_exc()` uses to look up source lines when building a traceback — without this registration, tracebacks would say `File "<repl>", line 3` but show no source text. With it, you get the actual line of code.

`exec(code_obj, namespace)` runs the compiled code inside the `namespace` dict. Variables created or modified during execution land directly in that dict, which is why variable persistence across calls works.

### Inside `rewrite_last_expr`

```python
def rewrite_last_expr(tree: ast.AST) -> Tuple[ast.AST, bool]:
```

The AST is a tree of node objects. The top-level node is a `Module`, and its `.body` is a list of statements — one per line (or per logical block). The function looks at the *last* element of that list.

```python
if not tree.body or not isinstance(tree.body[-1], ast.Expr):
    return tree, False
```

`ast.Expr` is the node type for a statement that is purely a value — a function call used as a statement (`df.head()`), an arithmetic expression (`x + y`), a string literal written alone. It is distinct from nodes like `ast.Assign` (`x = 5`), `ast.For`, or `ast.FunctionDef`. If the last statement is not an `ast.Expr`, there is nothing to capture, so the tree is returned unchanged.

When the last statement *is* a bare expression:

```python
last = tree.body[-1]
tree.body[-1] = ast.Assign(
    targets=[ast.Name(id="__repl_result__", ctx=ast.Store())],
    value=last.value,
    lineno=last.lineno,
    col_offset=last.col_offset,
    end_lineno=last.end_lineno,
    end_col_offset=last.end_col_offset,
)
ast.fix_missing_locations(tree)
return tree, True
```

```
             Figure 4.  AST rewrite: bare expression → assignment
    ═══════════════════════════════════════════════════════════════════

    Code:  df = pd.read_csv("data.csv")
           df.head()                       ← bare Expr node; exec() discards result

    BEFORE rewrite_last_expr()         AFTER rewrite_last_expr()
    ─────────────────────────          ─────────────────────────
    Module                             Module
    └── body                           └── body
         ├── Assign                         ├── Assign
         │    ├── targets: [Name "df"]      │    ├── targets: [Name "df"]
         │    └── value: Call(read_csv)     │    └── value: Call(read_csv)
         │                                  │
         └── Expr            ──────────►    └── Assign
              └── value:                         ├── targets:
                    Call                         │    [Name "__repl_result__"
                    ├── func: Attribute           │     ctx=Store()]
                    │    ├── Name "df"            └── value:
                    │    └── attr "head"                Call
                    └── args: []                        ├── func: Attribute
                                                        │    ├── Name "df"
                                                        │    └── attr "head"
                                                        └── args: []

    exec() evaluates Expr.value but     exec() stores return value in
    throws the result away.             namespace["__repl_result__"].
```

`last.value` is the expression node itself (e.g. the `Call` node representing `df.head()`). The rewrite wraps it in an `ast.Assign` with a single target: `ast.Name(id="__repl_result__", ctx=ast.Store())`. The `ctx=ast.Store()` signals that this name is being *written to* — `ast.Load()` would mean reading it.

The location fields (`lineno`, `col_offset`, `end_lineno`, `end_col_offset`) are copied from the original node rather than left to `ast.fix_missing_locations`. This matters because `fix_missing_locations` fills absent fields by inheriting from a parent node — which would give `end_lineno=1` on a node that lives on line 5, producing an invalid range (`lineno=5, end_lineno=1`) that causes a `ValueError` at compile time.

`ast.fix_missing_locations` is called after the replacement to fill in any fields that are still absent on child nodes we created (the `ast.Name` target), using the nearest parent's values as defaults.

### Inside `compile_for_repl`

```python
def compile_for_repl(code: str, tree: ast.AST) -> types.CodeType:
```

Python's built-in `compile(source, filename, mode)` turns an AST (or a string) into a code object that `exec()` can run. The `filename` argument is embedded in the code object and appears in any traceback produced during execution:

```
File "<repl>", line 3, in <module>
```

Without the next step, that would be all you see — no source line, no `^^^` underline. The reason is how Python generates tracebacks.

When `traceback.format_exc()` formats an exception, it calls `linecache.getlines(filename)` to look up the source text for that file. `linecache` is the module Python uses for this across the board — for `.py` files it reads from disk, but for synthetic filenames like `"<repl>"` it looks in an in-memory cache dict.

```python
linecache.cache["<repl>"] = (
    len(code),
    None,
    code.splitlines(keepends=True),
    "<repl>",
)
```

The cache entry is a 4-tuple that linecache expects: `(size, mtime, lines, fullname)`. `size` is the length of the source string. `mtime=None` tells linecache never to re-read or expire this entry (normally it uses mtime to detect file changes on disk). `lines` is the source split into individual lines, with newlines kept (`keepends=True`) because the formatter expects them. `fullname` is a second copy of the key.

After this registration, any exception raised inside `exec(code_obj, namespace)` produces a full traceback with the actual source line and underline:

```
File "<repl>", line 3, in <module>
  df.groupby("serotype").size().sort_values()
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
NameError: name 'df' is not defined
```

Without linecache registration the same error would show:

```
File "<repl>", line 3, in <module>
NameError: name 'df' is not defined
```

Note that `compile` receives the *rewritten* `tree` (from `rewrite_last_expr`) but the *original* `code` string. This is intentional — the source registered in linecache must match the original line numbers so the traceback points to the right line. If the rewritten AST were serialised back to source and registered, line numbers could shift.

### The sentinel pattern

```python
_sentinel = object()
repl_result = namespace.pop("__repl_result__", _sentinel) if has_result else _sentinel
```

After `exec`, if the AST rewrite ran, `__repl_result__` is now a key in `namespace`. `namespace.pop(key, default)` removes and returns it, using `_sentinel` as the default if the key is absent.

Why a sentinel instead of `None`? Because `None` is a legitimate expression value. Without the sentinel you cannot tell whether:

- The expression was absent (no rewrite happened, so `__repl_result__` was never set), or
- The expression was present and returned `None` (e.g. `print("hello")` returns `None`).

`object()` creates a fresh object that is not equal to anything else in existence. Checking `repl_result is not _sentinel` is an identity check — it asks "is this the exact object I created a moment ago?" which can only be true if `pop` returned the default, meaning the key was never set.

### Output assembly

```python
stdout_text = stdout_buffer.getvalue().strip()
if has_result and repl_result is not _sentinel:
    output = (stdout_text if stdout_text else "None") if repl_result is None else str(repl_result)
else:
    output = stdout_text if stdout_text else ""
```

`stdout_buffer.getvalue()` returns everything written to stdout during execution as a single string. `.strip()` removes leading/trailing whitespace.

The output logic handles three cases:

| Situation | Output shown |
|-----------|-------------|
| Bare expression that returned a value | `str(repl_result)` — e.g. a DataFrame repr |
| Bare expression that returned `None` | `stdout_text` if anything was printed, otherwise the string `"None"` |
| No bare expression (only statements or `print` calls) | `stdout_text` if anything was printed, otherwise empty string |

The `None` case is subtle: `print("hello")` is a bare expression (it's a function call used as a statement), and it returns `None`. The useful output is what was printed, not the `None` return value, so we fall back to stdout.

### Stderr handling

```python
stderr_text = stderr_buffer.getvalue().strip()
if stderr_text:
    output = f"{output}\n[stderr]: {stderr_text}" if output else f"[stderr]: {stderr_text}"
```

Stderr is collected separately and appended to the output string with a `[stderr]:` prefix if anything was written there. This matters because many libraries (e.g. `tqdm`, some matplotlib backends) write warnings or progress to stderr rather than stdout. Appending it keeps the LLM informed without discarding it.
