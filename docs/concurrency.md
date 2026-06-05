# Concurrency Notes

## Multiple sparq invocations

### Sequential invocations — safe

Each `Agentic_system.run()` call generates a fresh `uuid4` as `run_id`, which keys a separate temp pickle file for the REPL namespace. `run_dir` is timestamp-based (`%d-%m-%Y_%H-%M-%S`), so outputs land in separate directories. The namespace pickle is cleaned up in the `finally` block.

### Concurrent invocations — not safe

Two overlapping `run()` calls on the same `Agentic_system` instance will race because `run()` mutates `self`:

```python
# system.py — both lines overwrite instance attributes
self._get_node_definitions()  # sets self.graph, self.executor_node_partial, etc.
self._build_graph()           # sets self.graph
```

Request A can be mid-stream on `self.graph.astream(...)` while Request B's `_build_graph()` replaces `self.graph` underneath it.

Secondary issue: `run_dir` has 1-second timestamp resolution, so two runs starting within the same second share an output directory and can intermix files.

The REPL namespace pickle files (`_ns_paths` dict in `namespace.py`) are UUID-keyed and do not conflict under concurrency.

## Fix if exposing as a web API

**Easy:** instantiate `Agentic_system` per request — it's lightweight (just loads prompts and settings).

```python
@app.post("/query")
async def query(user_query: str):
    system = Agentic_system()
    return await system.run(user_query)
```

**Cleaner:** move `_get_node_definitions()` and `_build_graph()` into `__init__` (the graph structure is static across runs), and pass `run_dir` through the LangGraph config rather than baking it into the partial at call time.
