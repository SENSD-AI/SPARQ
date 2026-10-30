# Benchmarks

- `datasets/questions.json`: versioned evaluation questions.
- [References](references/README.md): preserved run bundles and their review status.
- `results/batch_eval/`: generated batches, ignored by Git.

Run from a repository checkout after `uv sync`:

```bash
uv run python -m sparq.evaluation.batch -n 10 -k 1 --evaluation-id baseline-v1
```

The batch runner is packaged under `sparq.evaluation`; the question set and saved
references are repository assets, not wheel contents. Existing historical batches
under `eval/results/` remain at their original paths. Preserved reference metadata,
hashes, and external run symlinks are historical evidence and are not rewritten
when the bundle moves.
