# Evaluation

- `datasets/questions.json`: versioned evaluation questions.
- [References](references/README.md): preserved run bundles and their review status.
- `results/batch_eval/`: generated batches, ignored by Git.
- `results/bedrock_models/`: generated regional model sweeps, ignored by Git.

See [evaluation result organization](../docs/guides/evaluation-results.md) for
bundle layouts, explicit output directories, and moving or preserving results.

Run from a repository checkout after `uv sync`:

```bash
uv run python -m sparq.evaluation.batch -n 10 -k 1 --evaluation-id baseline-v1
```

The batch runner is packaged under `sparq.evaluation`; the question set and saved
references are repository assets, not wheel contents. Historical batches have
been consolidated here from `eval/results/`. See
[MIGRATION.md](MIGRATION.md) for path mappings and verification. Preserved reference
metadata, hashes, and external run symlinks remain historical evidence.
