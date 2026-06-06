# Changelog

## Unreleased

### Dependencies — LangChain / LangGraph major version upgrade

Upgraded the full LangChain ecosystem from 0.3.x to 1.x and LangGraph from 0.4.x to 1.x.

**Package changes:**

| Package | Before | After |
|---|---|---|
| `langchain` | 0.3.25 | 1.3.4 |
| `langchain-core` | 0.3.66 | 1.4.1 |
| `langchain-community` | 0.3.24 | 0.4.2 |
| `langchain-experimental` | 0.3.4 | removed |
| `langchain-google-genai` | 2.1.5 | 4.2.4 |
| `langchain-openai` | 0.3.17 | 1.2.2 |
| `langchain-aws` | unpinned | 1.5.0 |
| `langchain-ollama` | 0.3.3 | 1.1.0 |
| `langgraph` | 0.4.5 | 1.2.4 |
| `langgraph-prebuilt` | 0.1.8 | 1.1.0 |
| `langsmith` | 0.3.45 | 0.8.9 |
| `openai` | 1.81.0 | 2.41.0 (required by langchain-openai 1.x) |
| `ollama` | 0.4.8 | 0.6.2 (required by langchain-ollama 1.x) |
| `boto3` / `botocore` | 1.40.55 | 1.43.24 (required by langchain-aws 1.5.0) |

**Code changes:**

- `src/sparq/tools/data_discovery_tools.py`: removed dead import `from langchain_experimental.utilities import PythonREPL` (never used; package is now sunset).
- `src/sparq/tools/python_repl/python_repl_tool.py`: migrated `from langchain.tools import tool` → `from langchain_core.tools import tool` (canonical path; langchain re-export removed in 1.x).

**Known deprecation warnings (not yet breaking):**

- `langchain-community` is being sunset. Current usage: `FileManagementToolkit` in `src/sparq/tools/filesystemtools.py`. Migration path TBD per upstream guidance at https://github.com/langchain-ai/langchain-community/issues/674.

**Verification:** All 59 unit tests pass. App initialises and graph executes correctly (end-to-end test blocked by expired AWS SSO token, unrelated to this change).
