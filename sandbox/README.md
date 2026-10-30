# Sandbox

Development exploration, not maintained examples or reproducible research runs:

- `notebooks/`: current exploratory notebook.
- `scripts/bedrock/`: direct SDK and LangChain API trials using `ENVSettings`.
- `archive/notebooks/`: historical REPL/package-manager trials whose imports target
  removed APIs. Contents and saved outputs are retained as research history.
- `archive/prompts/`: historical prompt variants; runtime prompts stay beside their architecture or tool.

Archived notebooks are not runnable examples. Current APIs are `ENVSettings`,
`sparq.utils.get_llm.get_llm`, `make_python_repl_tool`, and the run-scoped namespace
helpers. Empty scratch placeholders have been removed.

Some sandbox files make network calls or install/uninstall packages. They are not
executed by the project test launcher.
