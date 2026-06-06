# LangChain / LangGraph Migration Guide

Upgrade from the current 0.3.x / 0.4.x pin set to the latest 1.x releases.

---

## Version Targets

| Package | Current | Target |
|---|---|---|
| `langchain` | 0.3.25 | 1.3.4 |
| `langchain-core` | 0.3.66 | 1.4.1 |
| `langchain-community` | 0.3.24 | 0.4.2 |
| `langchain-experimental` | 0.3.4 | **REMOVE** |
| `langchain-google-genai` | 2.1.5 | 4.2.4 |
| `langchain-openai` | 0.3.17 | 1.2.2 |
| `langchain-aws` | unpinned | 1.5.0 |
| `langchain-ollama` | 0.3.3 | 1.1.0 |
| `langgraph` | 0.4.5 | 1.2.4 |
| `langgraph-prebuilt` | 0.1.8 | 1.1.0 |
| `langsmith` | 0.3.45 | 0.8.9 |

`langchain-experimental` is being sunset by the LangChain team. Its only import in this codebase (`PythonREPL` in `data_discovery_tools.py`) is dead code — the project's own subprocess REPL is used instead. Remove the package and the import.

---

## Required Code Changes

### 1. Remove dead import

**File:** `src/sparq/tools/data_discovery_tools.py`, line 3

```python
# DELETE this line
from langchain_experimental.utilities import PythonREPL
```

`PythonREPL` is never referenced after import. The file uses `execute_code` from `sparq.tools.python_repl.executor`.

### 2. Fix deprecated tool import

**File:** `src/sparq/tools/python_repl/python_repl_tool.py`, line 1

```python
# Before
from langchain.tools import tool

# After
from langchain_core.tools import tool
```

Every other file in the project already uses the `langchain_core` path. The `langchain` re-export was deprecated and removed in 1.x.

---

## Risky Areas — Post-Upgrade Status

All three risky areas were verified to be fine after the upgrade.

### `init_chat_model` import path ✅

`from langchain.chat_models import init_chat_model` still works in LangChain 1.x. No change needed.

### `create_react_agent` response_format tuple ✅

The `response_format=(prompt, ExecutorOutput)` tuple form is still supported in LangGraph 1.2.4. No change needed.

### `langchain-google-genai` 4.x ✅

Compatible. `init_chat_model('google_genai', model=...)` works without config changes.

### `langchain-community` deprecation warning ⚠️

`langchain-community` is also being sunset (like `langchain-experimental`). It emits a deprecation warning on import. Current usage is `FileManagementToolkit` in `src/sparq/tools/filesystemtools.py`. Not breaking yet — track upstream migration guidance at https://github.com/langchain-ai/langchain-community/issues/674.

---

## Upgrade Procedure

1. **Create a branch:**
   ```bash
   git checkout -b chore/upgrade-langchain-langgraph
   ```

2. **Update `pyproject.toml`** — bump all pinned versions per the table; delete the `langchain-experimental` line entirely.

3. **Apply the two code fixes** above.

4. **Install:**
   ```bash
   uv sync
   ```

5. **Run tests:**
   ```bash
   uv run python -m unittest
   ```

6. **Run the app end-to-end:**
   ```bash
   uv run sparq -t
   ```

7. **Address any runtime issues** in the risky areas listed above.

8. **Update `CHANGES.md`** — record what was upgraded, any breaking changes encountered, and the fixes applied.

9. **Commit and open a PR** once everything is green.

---

## Verification Checklist

- [x] `uv sync` completes without dependency conflicts
- [x] `uv run python -m unittest` — all 59 tests pass
- [ ] `uv run sparq -t` — graph runs to completion (blocked by expired AWS SSO token, unrelated to upgrade)
- [ ] `output/` contains a valid `final_answer.json`
- [x] No deprecation warnings about removed APIs (one warning about `langchain-community` sunset — not a removed API)
