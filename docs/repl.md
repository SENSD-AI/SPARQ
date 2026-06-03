# Python REPL Subsystem — Quirks and Constraints

This file documents non-obvious constraints of the REPL subsystem
(`src/sparq/tools/python_repl/`) that are not apparent from reading the code alone.

---

## 1. Top-level imports in the `sparq` package must not import third-party libraries

**Symptom:** Agent receives `JSONDecodeError: Expecting value: line 1 column 1 (char 0)`
from `python_repl_tool`. In the console, the subprocess prints:

```
ModuleNotFoundError: No module named 'pandas'
```

**Root cause:**

The executor spawns child processes via `multiprocessing.spawn`. Unlike `fork`, `spawn`
starts a fresh Python interpreter and re-runs `__main__` to reconstruct the process
state before executing `_target`. When the application is launched via `uv run sparq`,
`__main__` is the CLI entry point (`.venv/bin/sparq`), which triggers the full sparq
import chain:

```
sparq CLI → sparq.__main__ → sparq.utils.helpers → from pandas import DataFrame  ← crash
```

The child process crashes before executing any user code and before writing its result
file. The parent reads the empty result file and raises `JSONDecodeError`.

This does not affect `uv run python -m unittest` because the test runner is `__main__`,
not the sparq CLI, so the child process does not reimport the sparq package.

**Fix:** Any module reachable from `sparq.__main__` at import time must use only stdlib
at the top level. Third-party imports (`pandas`, `rich`, etc.) must be deferred inside
the functions that use them:

```python
# Bad — triggers on subprocess reimport
from pandas import DataFrame

# Good — only triggered when the function is actually called
def get_df_summary(df):
    from pandas import DataFrame
    ...
```

**Affected file:** `src/sparq/utils/helpers.py` — `pandas` and `rich` imports moved
inside `get_df_summary`, `get_df_summaries_from_manifest`, and `render_records_table`.
