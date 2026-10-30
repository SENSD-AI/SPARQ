# Data setup

Request access to the Hugging Face datasets listed in the repository README and
configure `HF_TOKEN` in your local environment. Download them from the repository:

```bash
uv run python -m sparq.utils.download_data
```

Bundled manifests and dataset descriptions live in `src/sparq/data/`. Downloaded
data and user configuration live outside the repository in the platform-specific
SPARQ configuration directory. The benchmark question set is separate, under
`benchmarks/datasets/questions.json`.

The canonical bundled full description is `data_summaries_full.json`; planning
uses `data_summaries_short.json`. Setup copies these to the user data directory.
The obsolete duplicate `data_summaries.json` is no longer bundled; existing user
copies are left untouched.
