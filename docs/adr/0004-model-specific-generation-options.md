# Model-specific generation options

- Date: 2026-11-04
- Status: Adopted; configured end-to-end smoke test verified
- Authority: User requested this ADR after discussing model compatibility and
  the planner's reported temperature error.

## Context

SPARQ can assign different models to the router, planner, executor, aggregator,
and vision roles. LangChain exposes common generation parameters, but each
underlying model determines which parameters and values it accepts.

The shared configuration previously supplied `temperature = 0.0`. Changing a
role's model in `config_v1.toml` retained that inherited setting. The user reported
a temperature deprecation error when running the planner with Sonnet 5.
Anthropic's [Sonnet 5 documentation, "Good to know"](https://platform.claude.com/docs/en/models/sonnet-5/overview#good-to-know)
states that setting `temperature`, `top_p`, or `top_k` returns a 400 error.

TOML has no null value. Omitting a key from an override file does not remove a
value inherited from the shared configuration, so a Python `None` default alone
cannot undo an inherited sampling setting.

## Decision

Keep shared generation defaults minimal. Do not set `temperature`, `top_p`, or
`top_k` in `default_config.toml`. Their optional fields in `LLMSetting` default to
Python `None` when no configuration source supplies a value.

Set sampling options explicitly in a role's configuration only when the chosen
model supports them. When changing a model, review its generation options at the
same time. Configure `max_tokens` within that model's supported output range;
there is no universal token budget suitable for every model.

Keep `get_llm()` as a simple function: forward explicitly supplied generation
values, preserve valid zero values, and omit options whose value is `None`.
Configuration owns SPARQ's defaults. Do not add duplicate generation defaults,
a capability registry, or automatic model-name-based parameter filtering for
this decision. Unsupported explicit settings should remain visible as errors.

Model compatibility also depends on the assigned role. The planner requires
structured output, and worker agents require tool calling. A shared construction
interface does not guarantee that every model can perform every SPARQ role.

## Consequences

Models can use their own sampling behavior when no sampling options are set.
Users can still configure supported controls without another abstraction.
Model selection and model-specific options must be maintained together; this
decision does not promise automatic support for all models.

## Implementation and verification

At the time of writing, source inspection confirms that:

- [Shared configuration](../../src/sparq/architectures/v1/default_config.toml)
  contains no sampling settings. Its existing `max_tokens = 100000` values remain
  and must be checked against any newly selected model.
- [LLMSetting](../../src/sparq/settings.py) defaults the three sampling fields
  to `None`.
- [get_llm()](../../src/sparq/utils/get_llm.py) omits `None` options and forwards
  explicit values using `is not None` checks.

On 2026-11-04, the user reported a successful run of `scripts/smoke_test.py`
after removing the shared sampling settings. Subsequent inspection of
`/Users/zayan/tmp/sparq/output/04-11-2026_15-10-51/` confirmed supporting artifacts:

- `log.txt` records plan creation, completion of worker step 1, aggregator
  execution, and the saver writing the final report.
- `result.json`, `final_answer.json`, `trace.json`, and `events.jsonl` exist.
- Both `report/report.md` and `report/report.pdf` exist.

This verifies completion of the configured dataset-catalog smoke scenario. It
does not establish compatibility with every model or validate the factual
accuracy or rendered layout of the generated report. No additional live model
calls were made when inspecting these artifacts and updating this ADR.
