# SPARQ

SPARQ is a multi-agent LangGraph pipeline for Salmonella epidemiological research. It routes a natural-language query through a planner, executor, and aggregator to produce a structured analytical report.

> **Recommendation:** Use a UNIX-based OS (Linux, macOS) or WSL on Windows.

---

## Setup

1. Clone the repository and `cd` into it.
2. Install [Python 3.13.3](https://www.python.org/downloads/) if you don't have it.
3. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it.
4. Install dependencies:
   ```bash
   uv sync
   ```

---

## Get access to the data

Request access to the datasets on [HuggingFace (zayanhugsAI)](https://huggingface.co/zayanhugsAI):

1. Pulsenet
2. National Outbreak Reporting System (NORS)
3. Social Vulnerability Index (SVI)
4. Map The Meal Gap
5. Census Population

You will need a HuggingFace account and an access token (`HF_TOKEN`) set in your `.env` file.

Once access is granted, download the data:

```bash
uv run python -m sparq.utils.download_data
```

---

## Configuration

SPARQ is configured via `config_v1.toml` in the project root. Create this file to override the defaults. The only required entries are the models you want to use:

> **User config and data directory** (where SPARQ stores its config and downloaded data outside the project):
> - **Linux / macOS:** `~/.config/sparq/`
> - **Windows:** `AppData\Local\sparq\` (i.e. `C:\Users\<username>\AppData\Local\sparq\`)

```toml
[llm_config.router]
model = "gemini-2.5-flash"
provider = "google_genai"

[llm_config.planner]
model = "gemini-2.5-flash"
provider = "google_genai"

[llm_config.executor]
model = "gemini-2.5-flash"
provider = "google_genai"

[llm_config.aggregator]
model = "gemini-2.5-flash"
provider = "google_genai"
```

Supported providers: `google_genai`, `openai`, `aws_bedrock`, `openrouter`. For AWS Bedrock setup see [docs/aws/README.md](docs/aws/README.md).

You can also override the output directory and set a custom test query:

```toml
test_query = "What are the main factors contributing to salmonella rates in Missouri?"

[paths]
output_dir = "~/my/output/dir"
```

---

## API Keys

Create a `.env` file in the project root (or at `~/.config/sparq/.env`) with the keys for whichever providers you use:

```
GOOGLE_API_KEY=...
OPENAI_API_KEY=...
HF_TOKEN=...

# LangSmith tracing (optional)
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=SPARQ

# AWS Bedrock (if using aws_bedrock provider)
AWS_PROFILE=...
AWS_REGION=us-east-1
```

---

## Running

Run with a predefined test query:

```bash
uv run sparq -t
```

Run interactively:

```bash
uv run sparq
```

---

## Outputs

Each run writes to a timestamped directory under `output_dir` (default `~/tmp/sparq/output`):

```
output/02-06-2026_11-28-17/
├── executor/          # plots and files generated during execution
├── trace.json         # full step-by-step trace
└── final_answer.json  # synthesised answer
```

Where `~` resolves to:
- **Linux / macOS:** `/home/<username>/tmp/sparq/output`
- **Windows:** `C:\Users\<username>\tmp\sparq\output`
- **WSL:** `/home/<username>/tmp/sparq/output` inside WSL — accessible from Windows Explorer at `\\wsl$\<distro>\home\<username>\tmp\sparq\output`

Override the output location in `config_v1.toml`:

```toml
[paths]
output_dir = "~/Documents/sparq_output"   # or any absolute path
```

LangSmith tracing is also available if configured — see [API Keys](#api-keys).

---

## Versioning & future work

SPARQ is structured to support multiple pipeline architectures side by side. The current implementation lives under `src/sparq/architectures/v1/` and is configured via `config_v1.toml` — the `v1` suffix exists to leave room for a v2 architecture without breaking existing configs or requiring code reorganisation.

A v2 pipeline is planned. It will introduce automated data cleaning and validation, DAG-based query decomposition, speculative parallel execution across multiple methodological strategies, and an editorial review gatekeeper for publication-grade outputs. See [docs/improvements.md](docs/improvements.md) for the full specification.

---

## FAQ

**Does this download any models to my computer?**
No. All models are accessed via API.

**What is LangGraph / LangSmith?**
LangGraph is a Python library for building systems of interconnected LLM agents. LangSmith is a platform for inspecting the step-by-step trace of such systems.

**What graph does SPARQ run?**
```
START → router → [needs analysis] → planner → executor → aggregator → saver → END
                 [direct answer]  → saver → END
```
