# Bedrock model auditing

Discover regional models and check compatibility with SPARQ's model interfaces.
Run from the repository root after `uv sync`:

```bash
uv run python -m utils.bedrock_utils
```

Results go to `outputs/<profile>/<region>/audits/<timestamp>/`, with per-model
folders under `runs/`. Generated outputs are ignored by Git. The reusable
inference-profile mapping remains in the OS user cache.

See the [audit guide](../../docs/guides/utils/bedrock_utils/README.md) for options and checks.
