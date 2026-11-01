# Try SPARQ with every regional Bedrock model

Run one basic SPARQ query per model and collect which ones work and why others fail:

```bash
uv run python scripts/aws/check_models.py
```

This performs paid model inference. It uses `AWS_PROFILE` and `AWS_REGION` loaded
through SPARQ's normal environment settings, falling back to the AWS default
profile and its configured region. It does not edit project configuration.

To choose a profile/region or output location:

```bash
uv run python scripts/aws/check_models.py \
  --profile my-profile --region us-east-1 \
  --output benchmarks/results/bedrock-check
```

To discover the targets first without inference:

```bash
uv run python scripts/aws/check_models.py --list-only
```

Use a new output directory for the actual sweep. Existing reports are not overwritten.
`--model-id ID` can select an exact discovered ID (repeat the option for multiple
IDs). `--timeout 300` changes the hard per-model deadline from its default 180 seconds.

## What the script does

1. Lists the complete regional foundation-model catalog and all inference-profile
   pages. Direct IDs and profile IDs are separate targets, so a model requiring a
   profile can fail under its direct ID and pass under the profile ID.
2. Records non-text foundation models as skipped, with their incompatible
   modalities as the reason. Other discovered targets receive a real SPARQ run.
3. Uses the candidate for router, planner, executor and aggregator. The dedicated
   vision model stays as configured; there is no separate image/capability test.
4. Runs the same small synthetic Salmonella query asking Python to sum and average
   2, 4 and 6. SPARQ's normal planner still receives its configured dataset
   descriptions. No special prompts, model adapters or graph nodes are substituted.
5. Accepts a pass only when the analytical route runs, all planned executor steps
   report success, the final response is nonempty, and JSON/Markdown/PDF outputs
   exist. It does not impose an additional CSV task or evaluate scientific quality.
6. Captures exceptions, original wrapped AWS error codes, step-failure reasons,
   worker crashes and timeouts, then continues to the next target.

Each run has an isolated process and output directory. On POSIX, a timeout or Ctrl-C
terminates the worker's process group, including spawned REPL processes. Windows
uses worker termination; descendant cleanup is not verified there.

## Results

Default location: `benchmarks/results/bedrock_models/<UTC timestamp>/`.

- `summary.md`: list of models that passed, followed by all models and reasons.
- `results.csv`: one row per target with status, error code, reason, artifacts and duration.
- `report.json`: complete catalog/results, profile, region, query, error chains,
  and discovery errors.
- `runs/<target hash>/`: worker log, job description, check result and SPARQ artifacts.

Results are saved initially and after every model. Ctrl-C retains finished results, records the active run as interrupted, and leaves
remaining targets as `not_tested`. Start a new report with `--model-id` to retry
selected targets. A failed or timed-out run means this attempt did not work in
this account/configuration; it does not establish permanent model incompatibility.
The configured vision model, dataset setup, SDK version and other SPARQ code can
also cause a run to fail. Check the recorded error and per-run log.

Coverage is the catalog and profiles visible in the selected region, not all AWS
regions or undeployed custom/Marketplace/provisioned endpoints. The script does not create provisioned endpoints or edit AWS policies.
Catalog permission/network failures are recorded as incomplete discovery rather
than treated as an empty successful scan.

Exit codes: `0` means the sweep/report finished (individual models may fail),
`2` means discovery was incomplete or arguments invalid, `130` means interrupted.
No paid sweep has been run as part of implementing this script.

## Sources read for implementation

- [ListFoundationModels](https://docs.aws.amazon.com/boto3/latest/reference/services/bedrock/client/list_foundation_models.html): request and response syntax, identifiers and modalities.
- [ListInferenceProfiles](https://docs.aws.amazon.com/boto3/latest/reference/services/bedrock/client/list_inference_profiles.html): pagination, profile identifiers/types and underlying models.
- [Converse](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html): invocation identifiers, tool use and required invocation permissions.

Implementation behavior was checked against SPARQ's current model factory,
settings loader, graph, executor and report writer. The script uses live catalog
responses, not a hard-coded list copied from documentation or search results.
