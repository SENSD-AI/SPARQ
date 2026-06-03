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

---

## 2. Tools that inject data into the REPL namespace must use `execute_code`

**Symptom:** Agent receives `NameError: name 'mmg_2009' is not defined` after successfully
calling `load_dataset`. The DataFrame was loaded in the parent process but is invisible
inside the REPL subprocess.

**Root cause:**

`load_dataset` previously stored a pandas DataFrame directly into the namespace pickle
file from the parent process. When the REPL subprocess called `load_ns(ns_path)`,
`pickle.load` needed to import pandas to reconstruct the DataFrame — but at that point
user code had not run yet and the auto-installer had not fired. The deserialization
failed, the namespace came back empty, and the agent's code failed with `NameError`.

**Rule:** Any tool that needs to make a variable available in the REPL namespace must
create it **inside** a subprocess call via `execute_code(code, ns_path=ns_path)`, not
by writing Python objects directly to the pickle file from the parent process.

```python
# Bad — writes DataFrame from parent process; subprocess can't deserialize it
ns = load_ns(ns_path)
ns['df'] = pd.read_csv(file_path)
pickle.dump(ns, open(ns_path, 'wb'))

# Good — DataFrame is created inside the subprocess; auto-installer handles pandas
result = execute_code(f"import pandas as pd\ndf = pd.read_csv({repr(file_path)})", ns_path=ns_path)
```

Once the first `execute_code` call installs pandas and writes the DataFrame back to
`ns_path`, all subsequent subprocesses can deserialize it — pandas is now in the venv.

**Affected file:** `src/sparq/tools/data_discovery_tools.py` — `make_load_dataset_tool`
now delegates to `execute_code` instead of pickling directly.

**Future direction:** The root cause of all these subprocess constraints is the custom
`multiprocessing.spawn` REPL. A community-maintained data science skill (in Anthropic's
skill paradigm) would handle package availability, namespace persistence, and
cross-process state transparently. This should be considered when the system is next
significantly rearchitected.
