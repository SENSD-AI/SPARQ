# LangChain / LangGraph / LangSmith latest-version compatibility audit

**Audit date:** 2026-10-19

This is a source-level compatibility report, not an applied dependency upgrade. The
versions called “latest” below are the latest stable releases returned by the PyPI JSON
API on the audit date. Pre-releases were not considered.

## Executive result

SPARQ's active LangChain, LangGraph, and LangSmith usage is **runtime-compatible with
the latest stable trio**:

| Distribution | Locked in `uv.lock` | Latest audited | Result |
|---|---:|---:|---|
| `langchain` | 1.3.11 | 1.4.6 | Compatible |
| `langchain-core` (required by LangChain) | 1.4.8 | 1.6.12 | Compatible, with stricter typing findings noted below |
| `langgraph` | 1.2.8 | 1.2.16 | Compatible |
| `langsmith` | 0.9.8 | 0.14.11 | Compatible |

The clean-room resolver installed those four versions together on Python 3.13.13.
The current direct requirements already use lower bounds for these distributions, so
`uv lock --upgrade` can select them. The repository has **not actually migrated** until
`uv.lock` is regenerated and committed.

There are no removed imports or runtime API breaks in the active source. There are,
however, four migration-adjacent cautions:

1. The integration distributions remain constrained separately. The compatibility
   environment retained `langchain-openai==1.2.2`, `langchain-aws==1.5.0`,
   `langchain-ollama==1.1.0`, and `langgraph-prebuilt==1.1.0`, while resolving
   `langchain-google-genai==4.3.7`. This proves the requested trio against the project's
   declared integration constraints; it is not an audit of upgrading every integration
   to its newest release.
2. `langchain-community==0.4.2` still emits its upstream sunset warning when
   `FileManagementToolkit` is imported. The toolkit continues to work, so this is not a
   blocker for the trio upgrade, but it is technical debt.
3. `InjectedToolCallId`, `Tool`, `Command`, and `ToolMessage` are imported by
   `data_discovery_tools.py` but never used. Their continued upstream availability was
   verified anyway; they can be removed independently.
4. Latest type information exposes two LangChain/LangGraph call-site errors around raw
   dictionary inputs to `agent.ainvoke(...)` and `graph.astream(...)`. Those shapes are
   accepted at runtime and are the documented message/input shape, but the generic
   return types inferred for this project are narrower. These are typing clean-up items,
   not observed runtime incompatibilities. The full type checker also reports many
   unrelated, pre-existing errors, so it is not currently a green migration gate.

## Compatibility by source-code usage

### 1. Agent creation, middleware, structured output, and invocation

**SPARQ usage:** `langchain.agents.create_agent`,
`langchain.agents.middleware.TodoListMiddleware`, `system_prompt` as both a string and a
`SystemMessage`, tools as `BaseTool` objects, Pydantic classes passed as
`response_format`, synchronous `invoke`, asynchronous `ainvoke`, and reading the
`structured_response` state key.

**Finding: compatible.** The latest `create_agent` signature still accepts all of
`model`, `tools`, `system_prompt`, `middleware`, `response_format`, and `name`. Its
source explicitly accepts a Pydantic model class as `response_format`, wraps a raw
schema in `AutoStrategy`, parses provider/tool structured output, and writes the parsed
object to `state["structured_response"]`. `TodoListMiddleware()` still has a no-argument
constructor, contributes the `write_todos` tool, and supports both sync and async model
wrappers. The compiled agent remains a LangGraph runnable and therefore retains
`invoke`/`ainvoke`.

**Typing note:** latest stubs reject the inferred type of the literal
`{"messages": [{"role": "user", "content": ...}]}` at the executor's `ainvoke` call.
The factory source and runtime accept the representation. If a zero-error static check
is required, explicitly annotate/cast the value as the agent input state or construct a
`HumanMessage`; no runtime migration change is required.

### 2. Unified model construction and provider integrations

**SPARQ usage:** `langchain.chat_models.init_chat_model(model=...,
model_provider=...)` for OpenAI and Google GenAI; direct `ChatOpenAI` construction for
OpenRouter; direct `ChatBedrockConverse` construction for Bedrock; and
`with_structured_output(...).invoke(...)` on the resulting chat model.

**Finding: compatible for the requested trio.** The latest `init_chat_model` signature
still has the exact `model` and keyword-only `model_provider` arguments. Its provider
map still includes `openai` and `google_genai` and dynamically imports their integration
packages. The resolver successfully combined latest LangChain/core with the project's
existing integration constraints. Model-network calls were not made because that would
require valid provider credentials.

**Typing note:** `ChatOpenAI(api_key=<str>)` is rejected by the currently pinned
integration's latest-visible type signature, which asks for `SecretStr`, a callable, or
`None`. Runtime construction continues to accept the string. This concerns the direct
OpenRouter factory and the pinned integration, not a removed LangChain 1.4 API.

### 3. Tools and content/artifact results

**SPARQ usage:** `langchain_core.tools.tool` as both `@tool` and a configured decorator,
Pydantic `args_schema`, `response_format="content_and_artifact"`, tool `.invoke(...)`,
and the `InjectedToolCallId` export (currently unused).

**Finding: compatible.** The latest decorator still accepts `args_schema` and the two
response formats. For `content_and_artifact`, the source still requires the two-tuple
that SPARQ returns. `InjectedToolCallId` remains exported and retains its injected
argument semantics. The repository's REPL tool invocation tests passed on the latest
stack.

### 4. Messages, prompt templates, and structured chat-model output

**SPARQ usage:** `AnyMessage`, `SystemMessage`, `HumanMessage`, `ToolMessage` (unused),
`BasePromptTemplate`, `PromptTemplate.from_template(...).partial(...)`, prompt
`.invoke({}).to_string()`, and chat-model `with_structured_output(...).invoke(...)`.

**Finding: compatible.** All imports remain public exports. The prompt source retains
`from_template`, `partial`, and the Runnable `invoke` path; the returned prompt value
still supports `to_string`. Current message classes and the `AnyMessage` discriminated
union remain available. Structured-output calls are supplied by the chat model
interface and were import/type inspected against core 1.6.12.

### 5. Graph state, message reducer, graph construction, retries, and streaming

**SPARQ usage:** `StateGraph(state_schema=State)` with a Pydantic state model,
`Annotated[list[AnyMessage], add_messages]`, `START`, `END`, `add_node`, `add_edge`,
`add_conditional_edges`, `compile`, `RetryPolicy(max_attempts=..., retry_on=...)`, and
compiled-graph `astream(..., config=..., stream_mode="values")`.

**Finding: runtime-compatible.** Latest `StateGraph` still accepts a schema class and
extracts reducers from `Annotated` fields. `add_messages` still merges message lists and
updates equal IDs. The graph builder retains every method used by SPARQ, and compiled
graphs explicitly expose `astream`. `RetryPolicy` retains both configured fields with
compatible meanings and types. `START` and `END` remain exported from
`langgraph.graph`.

`Command` is still present in `langgraph.types`, but SPARQ only imports it and never
uses it. The latest source was inspected to rule out an import break.

**Typing note:** the checker rejects `{"query": user_query}` as the inferred
`State | Command | None` argument to `astream`, because `State` is a Pydantic model even
though partial dictionary graph inputs are supported at runtime. An explicit input
schema or a narrow cast can remove this static-only finding.

### 6. LangSmith tracing

**SPARQ usage:** no direct LangSmith Python API. Configuration is indirect through
`LANGSMITH_API_KEY`, `LANGSMITH_TRACING`, and `LANGSMITH_PROJECT`, which
`ENVSettings` copies to `os.environ`; LangChain/LangGraph callbacks then trace
implicitly. Documentation also mentions legacy `LANGCHAIN_*` spellings.

**Finding: compatible.** LangSmith 0.14.11's environment lookup checks the
`LANGSMITH` namespace first and the `LANGCHAIN` namespace second, so both the active
settings and legacy documentation remain understood. Its project selection uses
`PROJECT` through the same dual-namespace helper. The settings test passed with the
latest SDK. A test run attempted to send traces because the local template contains a
placeholder key and tracing enabled; the proxy rejected the external request. That is
an environment/configuration issue, not an SDK compatibility failure.

### 7. Community file-management toolkit

**SPARQ usage:** `langchain_community.agent_toolkits.FileManagementToolkit`,
`selected_tools`, `root_dir`, and `.get_tools()`.

**Finding: works, but deprecated at the distribution level.** Construction and the
filesystem tests passed. Importing `langchain_community` emits an explicit warning that
the project is being sunset and directs users to its migration issue. There is no
replacement in the latest LangChain/core trio that must be adopted for this upgrade;
treat replacement as a separate migration.

## Exactly what upstream material was read

The following inventory is intentionally precise. “Lines” refer to the source files in
the published wheels installed from PyPI for the stated version. The matching published
source archives and release pages are linked so the material can be independently
retrieved.

### Package/release metadata

- PyPI JSON metadata (`info.version`, release files, and dependency metadata) for
  [`langchain`](https://pypi.org/pypi/langchain/json),
  [`langchain-core`](https://pypi.org/pypi/langchain-core/json),
  [`langgraph`](https://pypi.org/pypi/langgraph/json),
  [`langsmith`](https://pypi.org/pypi/langsmith/json),
  [`langchain-community`](https://pypi.org/pypi/langchain-community/json),
  [`langchain-google-genai`](https://pypi.org/pypi/langchain-google-genai/json),
  [`langchain-openai`](https://pypi.org/pypi/langchain-openai/json),
  [`langchain-ollama`](https://pypi.org/pypi/langchain-ollama/json),
  [`langchain-aws`](https://pypi.org/pypi/langchain-aws/json), and
  [`langgraph-prebuilt`](https://pypi.org/pypi/langgraph-prebuilt/json), fetched on
  2026-10-19.
- Installed `METADATA` `Requires-Dist` entries for `langchain==1.4.6`,
  `langchain-core==1.6.12`, `langgraph==1.2.16`, and `langsmith==0.14.11`.
- [LangChain's public release/versioning statement](https://github.com/langchain-ai/langchain/issues/38437) describing stable 1.x semantic
  versioning and compatibility expectations.

### `langchain==1.4.6`

Repository/source: [`langchain-ai/langchain`](https://github.com/langchain-ai/langchain), plus the [exact PyPI 1.4.6 source archive](https://files.pythonhosted.org/packages/d1/19/f4fd7d8ea845558bdc75c249bf9392250c67f38b7a242a9a3854c4f8e32e/langchain-1.4.6.tar.gz) (SHA-256 `ad3ccfdc50005eb06bf1fcbae61aa69ad98fe8bc7d7c1b4ad88323b5bda4cbac`) and published wheel.

- `langchain/agents/factory.py`: lines 822-914 (typing overloads and full
  `create_agent` signature), 915-971 (arguments relevant to SPARQ), 1066-1082 (raw
  response schema normalization), 1247-1332 (provider/tool structured-response
  parsing), 1401-1480 (effective response strategy and model binding), 1501-1593
  (sync/async model invocation and structured response propagation), and 2035-2062
  (structured-response graph exit routing).
- `langchain/agents/middleware/todo.py`: lines 174-260 (`TodoListMiddleware` class,
  constructor, generated tool, and synchronous wrapper) plus its adjacent asynchronous
  wrapper implementation.
- `langchain/chat_models/base.py`: lines 232-470 (`init_chat_model` signature,
  documented arguments, provider mapping, and integration requirements) and the
  provider dispatch/import portion of that function.

### `langchain-core==1.6.12`

Repository/source: [`langchain-ai/langchain`](https://github.com/langchain-ai/langchain), plus the [exact PyPI 1.6.12 source archive](https://files.pythonhosted.org/packages/12/c2/9779d64322d4e270f6d5392bc86b30f735581a87b5eefd6ad15a0828a169/langchain_core-1.6.12.tar.gz) (SHA-256 `dc6b2a965c351b4cc94df1c80220bcf3c5472e14437d2186c8290ea22f4d0219`) and published wheel.

- `langchain_core/tools/__init__.py`: lines 1-85 (public exports and lazy import map).
- `langchain_core/tools/convert.py`: lines 77-190 (`tool` signature and documented
  `args_schema` / `response_format` contract) and its decorator conversion branch.
- `langchain_core/tools/base.py`: lines 1756-1780 (`InjectedToolCallId`).
- `langchain_core/tools/simple.py`: `Tool` class declaration and invocation wrapper.
- `langchain_core/messages/__init__.py` and
  `langchain_core/messages/utils.py`: public message exports and the `AnyMessage`
  discriminated union/conversion path.
- `langchain_core/messages/system.py`: lines 9 onward (`SystemMessage`).
- `langchain_core/messages/human.py`: lines 9 onward (`HumanMessage`).
- `langchain_core/messages/tool.py`: lines 26 onward (`ToolMessage`).
- `langchain_core/prompts/base.py`: lines 38-330 (`BasePromptTemplate`, `invoke`, and
  `partial`).
- `langchain_core/prompts/prompt.py`: lines 24-180 (`PromptTemplate`) and lines
  257 onward (`from_template`).
- `langchain_core/prompt_values.py`: the `StringPromptValue.to_string` implementation.
- `langchain_core/language_models/chat_models.py`: the `with_structured_output`
  interface and Runnable `invoke`/`ainvoke` path relevant to SPARQ.
- `langchain_core/runnables/config.py`: the `RunnableConfig` definition and
  `configurable` field used by the executor node.

### `langgraph==1.2.16`

Repository/source: [`langchain-ai/langgraph` tag `1.2.16`](https://github.com/langchain-ai/langgraph/tree/1.2.16/libs/langgraph), the [exact PyPI 1.2.16 source archive](https://files.pythonhosted.org/packages/db/65/69de99a24546c1e3b75a67905dbcc17ed6cd0ed81694d4cc021ed688a60d/langgraph-1.2.16.tar.gz) (SHA-256 `e4dd63480bfd3ee14a6ab80ecf1311698780519693d8e3b67a28f3486e69e1f7`), and the published wheel.

- `langgraph/graph/__init__.py`: the complete file (public `START`, `END`,
  `StateGraph`, `MessagesState`, and `add_messages` exports).
- `langgraph/graph/state.py`: lines 131-300 (`StateGraph` contract and schema/reducer
  model), 376-735 (`add_node` overloads and implementation), 928-1025 (`add_edge` and
  `add_conditional_edges`), and 1177 onward (`compile`).
- `langgraph/graph/message.py`: lines 45-220 (`add_messages` wrapper, signature,
  merge semantics, and graph-state example).
- `langgraph/types.py`: lines 427-445 (`RetryPolicy`) and 832-878 (`Command`).
- `langgraph/pregel/main.py`: lines 3154 onward (compiled graph `astream` signature,
  `input`, `config`, and stream-mode handling).
- `langgraph/constants.py`: `START` and `END` definitions.

### `langsmith==0.14.11`

Repository/source: [`langchain-ai/langsmith-python`](https://github.com/langchain-ai/langsmith-python), the [exact PyPI 0.14.11 source archive](https://files.pythonhosted.org/packages/f8/c3/a9a98d7adf573fdd5ea5c92cadbc26d7c597ab36f4dc31c093eb19507808/langsmith-0.14.11.tar.gz) (SHA-256 `d9e7ffe587e12ce51573cab654d02e84feb924d3b1f582274c4b1412ebeabf4c`), and the published wheel.

- `langsmith/utils.py`: lines 369-435 (`is_env_var_truish`, dual-namespace
  `get_env_var`, and project selection).
- `langsmith/run_helpers.py`: lines 450-500 (tracing enablement contract around
  `traceable`).
- `langsmith/_client.py`: lines 97-122 and 389-414 (API-key environment resolution).
- `langsmith/client.py`: lines 1025-1065 (`Client` initialization environment
  behavior).
- `langsmith/env/_runtime_env.py`: lines containing the legacy `LANGCHAIN_API_KEY`,
  `LANGCHAIN_TRACING_V2`, and `LANGCHAIN_PROJECT` compatibility names.
- `langsmith/run_trees.py`: lines 194-205 (project/tracing baggage keys).

### `langchain-community==0.4.2` (adjacent dependency)

Repository/source: [`langchain-ai/langchain-community`](https://github.com/langchain-ai/langchain-community), the [exact PyPI 0.4.2 source archive](https://files.pythonhosted.org/packages/ea/0c/e3aca1f2b1c5b95f8b87cb2b6e81a6f20d538c07a128419dc01cef0617b6/langchain_community-0.4.2.tar.gz) (SHA-256 `a99308160d53d7e9b5965ee665e5173709914338210089fd5788ad724432c21e`), and the published wheel.

- `langchain_community/agent_toolkits/file_management/toolkit.py`: lines 31 onward
  (`FileManagementToolkit`, `root_dir`, `selected_tools`, and `get_tools`).
- `langchain_community/tools/__init__.py` and the resolved
  `langchain_core.tools.simple.Tool` export used by the currently dead import.
- The package import warning and [upstream sunset/migration issue #674](https://github.com/langchain-ai/langchain-community/issues/674).

## Verification performed

1. Queried PyPI for all ten relevant distribution versions.
2. Resolved and installed the repository into a fresh Python 3.13.13 virtual
   environment. Resolution succeeded with latest LangChain, core, LangGraph, and
   LangSmith together.
3. Imported every LangChain/LangGraph/community symbol used by active `src/` code and
   resolved each to its installed source file.
4. Ran the unit suite on that environment. 59 tests passed, one was skipped, four
   unrelated package-manager/Matplotlib assertions failed, and one test module could
   not import because `tiktoken` attempted a proxy-blocked encoding download. The
   LangChain-facing REPL tool, filesystem toolkit, settings, and graph-adjacent tests
   that did run passed.
5. Ran BasedPyright 1.40.2. It reported 77 existing errors. The latest framework types
   specifically surfaced the two dictionary-input findings described above; no missing
   LangChain, LangGraph, or LangSmith imports were reported.

## Recommended migration sequence

1. Change the direct minimums to the audited versions if the project intends to state a
   minimum supported baseline: `langchain>=1.4.6`, `langchain-core>=1.6.12`,
   `langgraph>=1.2.16`, and `langsmith>=0.14.11`.
2. Regenerate `uv.lock` and verify the resolved versions. Do not assume changing only
   the TOML updates a committed lockfile.
3. Run tests with `LANGSMITH_TRACING=false` and with the `cl100k_base` tokenizer cache
   pre-populated (or network access to its official blob host).
4. Perform one credentialed smoke test per configured provider and one end-to-end graph
   run. Source compatibility cannot prove provider-side model access or credentials.
5. Handle the `langchain-community` replacement and the wider integration-package
   upgrades as separate changes; they are not required to move the requested trio.
