# Latest SPARQ run review — 2026-11-01

**Verdict: execution completed, but the answer is incomplete and its central risk conclusion is unsupported.** The largest analytical error is inclusion of food isolates in population-based comparisons: restricting the same source data to human isolates reverses Boone County's apparent elevation. The aggregator also stopped at its output limit, omitting the requested actions, while the batch recorded success.

This is an observational review, with independent local recomputation. No application code, configuration, or original run artifact was changed. Remedies below are proposals, not implemented changes or adopted decisions.

## Run and evidence

- Batch: `2026-11-01T19-20-26Z_ffaf1cc8`; run: `3bd1d1ad-c1a4-45d1-9dbb-4b92c426e9a2`.
- Question: “What is the current salmonella risk level in Boone county, and what actions should I take?”
- November 1, 2026, 13:20:26–13:41:11 America/Chicago (CST); 1,244.4 seconds for the run, approximately 20m 44s.
- One question, one iteration; batch and run both `completed`; all nine worker results say `success=true`. There is no saved judge score or judge review.
- Router: `us.anthropic.claude-haiku-4-5-20251001-v1:0`; planner/executor/aggregator: literal identifier `us.anthropic.claude-sonnet-5`; vision: `us.anthropic.claude-sonnet-4-5-20250929-v1:0`. These are recorded configuration identifiers, not an independent verification of model identity behind the endpoint.
- Recorded source commit: `1a640a8aacf9cc0117299b6456b8f8c25a755ed9`, dirty. Current checkout: `7229b05`. The inspected aggregator, executor, saver, system, LLM factory, and batch runner have no current diff against that recorded commit; the run's dirty working tree was not snapshotted.
- The current question-set SHA-256 matches the batch manifest: `295351abcee12791e964dcf2627fa48900f43d574cee289901260253824968b0`.

[Batch manifest](../../evaluation/results/batch_eval/2026-11-01T19-20-26Z_ffaf1cc8/batch.json), [saved result](../../evaluation/results/batch_eval/2026-11-01T19-20-26Z_ffaf1cc8/runs/iteration_001/question_001/3bd1d1ad-c1a4-45d1-9dbb-4b92c426e9a2/result.json), [trace](../../evaluation/results/batch_eval/2026-11-01T19-20-26Z_ffaf1cc8/runs/iteration_001/question_001/3bd1d1ad-c1a4-45d1-9dbb-4b92c426e9a2/trace.json), and [report](../../evaluation/results/batch_eval/2026-11-01T19-20-26Z_ffaf1cc8/runs/iteration_001/question_001/3bd1d1ad-c1a4-45d1-9dbb-4b92c426e9a2/report/report.md) are the primary run evidence. Generated results are ignored by Git; these links depend on retaining the local batch bundle.

Scope actually inspected: the complete saved answer and report, trace fields and all nine structured worker results, the log, all six generated CSV tables, seven text/Markdown artifacts, the artifact index, all four figures, and all five rendered PDF pages. Independent source checks used the full `Export_2020-2023.xlsx`, the PulseNet dictionary, NORS `NORSMain`/`Counties`/`GenEtiology`, both SVI county CSVs, the Census population sheet, and the Map the Meal Gap `County` sheet. The other five PulseNet export workbooks were not independently reloaded; older-year counts below come from the saved extract/table. Worker tool conversations and executed Python source are not preserved in the saved trace.

## Findings, in priority order

### 1. Critical: nonhuman isolates reverse the population-based comparison

The saved Boone extract contains 189 isolates: 132 HUMAN and 57 FOOD. For 2020–2022, the report uses all 78 isolates, including 27 FOOD isolates, as the numerator over resident population. The PulseNet dictionary explicitly defines nonhuman `SourceCounty` as where the isolate was collected; it is not a resident human case location. These records cannot establish human disease burden per resident.

Re-reading the original 2020–2023 workbook and normalizing `SourceType`, `SourceState`, and `SourceCounty` with uppercase/strip gives:

| 2020–2022 pooled measure | Boone, MO | Missouri | United States |
|---|---:|---:|---:|
| Report numerator, all source types | 78 | 2,044 | 102,662 |
| Human-only numerator | 51 | 1,957 | 101,786 |
| Population-years | 557,883 | 18,501,987 | 996,847,321 |
| Reported isolates /100,000/year | 13.98 | 11.05 | 10.30 |
| Recomputed human isolates /100,000/year | **9.14** | **10.58** | **10.21** |

Boone's annual human counts are 9, 26, and 16; food counts are 9, 9, and 9. Human-only rate ratios against inclusive state/national rates are **0.864 and 0.895**, rather than the report's **1.266 and 1.358**. Comparisons against the rest of Missouri/US, excluding Boone from each comparator, yield RR **0.861** (95% exact conditional CI **0.638–1.136**, two-sided p **0.321**) and **0.895** (CI **0.667–1.177**, p **0.466**).

The claimed statistically significant elevation does not survive this correction. This is not evidence that present-day Boone risk is low: these remain historical isolate records, not a validated current risk estimate or necessarily unique illness episodes.

The raw historical jump also mixes source populations: the saved extract has no food isolates in 2011–2018, but 28 in 2019 and 27 in 2020–2022. Thus a major observed composition change must be addressed before assigning that jump to disease incidence or surveillance methodology. Human-only counts in the saved extract imply 65/4 versus 62/8 isolates/year, a ratio of **2.10**, versus the mixed-source **3.87**; this is still a historical descriptive comparison with ascertainment and population limitations.

**Proposed remedy:** define and enforce the target population before aggregation; report human and nonhuman surveillance separately; rederive every dependent table, figure, statistical comparison, synthesis, and recommendation.

### 2. Critical: output truncation is accepted as successful completion

The final assistant message records `stopReason: max_tokens`, **4,096 output tokens**, and 34,882 input tokens for that final call. The response ends literally: “Panel 2 compares historical”. The same incomplete body appears in `result.json`, `final_answer.json`, Markdown, and the five-page PDF. This is generation truncation, not a PDF-only rendering defect.

The report contains no recommended-actions section, complete figure caption, concluding synthesis, or references. Step 9 did save a substantial action document, but the final answer never reaches it. The user's second explicit request is therefore unanswered in the delivered response.

The code explains acceptance: `aggregator_node()` returns the last message without checking its stop reason; the saver renders any nonempty final assistant text; `run()` checks the message type; the batch marks completion when no exception is raised. `get_llm()` receives model/provider only, so configured `LLMSetting.max_tokens` is not forwarded to the constructor. In the aggregator, that setting is instead used as an input-results budget. Merely editing TOML would not resolve that wiring.

Sources: [aggregator](../../src/sparq/architectures/v1/nodes/aggregator.py), [LLM factory](../../src/sparq/utils/get_llm.py), [saver](../../src/sparq/architectures/v1/nodes/saver.py), [system](../../src/sparq/architectures/v1/system.py), [batch runner](../../src/sparq/evaluation/batch.py).

**Proposed remedy:** distinguish complete answers from generation cutoffs; forward an explicit output budget through the model factory; recover or regenerate truncated output; verify requested-answer coverage before declaring answer success. Increasing the output allowance alone does not fix finding 1.

### 3. High: historical surveillance is labeled “current risk”

At the November 2026 question date, the primary comparison ends in 2022, NORS ends in 2021, food insecurity ends in 2020, and the isolate extract contains only partial 2023 data. A caveat acknowledges these limits, but the title and opening still assert a current Low-to-Moderate risk classification and say no active real-time outbreak was identified “in any dataset used.” These sources cannot establish current outbreak presence or absence.

**Proposed remedy:** answer that current risk cannot be determined from the available historical inputs. Present dated surveillance context separately. Current advice requires current official sources; this review did not perform a live Boone outbreak search and makes no claim about present outbreak status.

### 4. High: the composite risk score is an unvalidated, model-created rubric

Step 7 chooses weights 0.40/0.20/0.25/0.15 for isolate rate, NORS, SVI, and food insecurity, along with Low/Moderate/High thresholds of 0.67 and 1.33. No calibration, validation, or external basis is supplied. Stating weights makes an arithmetic recipe inspectable; it does not establish predictive validity or justify a public-health risk category.

The main 0.65 score is arithmetically consistent with assigned domain scores, but its 0.40 isolate contribution rests on finding 1. The food-insecurity domain is labeled Low-Moderate in the CSV but assigned zero in the calculation. Sensitivity documentation also conflicts: giving the raw trend a final weight of 0.30 and proportionally reducing other weights gives `0.30*2 + 0.70*0.65 = 1.055`, whereas adding an unnormalized 0.30 and dividing all weights by 1.30 gives `(0.65 + 0.30*2)/1.30 = 0.961538`. The latter reproduces the saved 0.96; those are different weighting statements, though both land in the rubric's Moderate band.

**Proposed remedy:** remove the clinical/current-risk label. If an exploratory index is retained for research, label it unvalidated, specify the scoring rules exactly, and evaluate sensitivity to corrected inputs and plausible weights without presenting the resulting band as established risk.

### 5. High: p-values do not reproduce from the stated tests

Using SciPy 1.18.1 `binomtest` for conditional exact comparisons:

| Comparison using the report's mixed-source counts | Saved claim | Recomputed two-sided conditional exact p |
|---|---:|---:|
| 120 isolates /4 years vs 62 /8 years | Step 4: 2.84e-19; Step 8/final: approximately 0.0006 | **3.6771e-19** |
| Boone vs inclusive Missouri | 0.027 | **0.04557** |
| Boone vs inclusive US | 0.0057 | **0.009853** |

For the historical comparison, the conditional null probability is `4/(4+8)` and the total is 182. The final answer adopts the much larger p-value as “independently recomputed” without resolving how it was produced. For nested county/state/nation comparisons, disjoint county-versus-rest groups are preferable; their mixed-source p-values are 0.03824 and 0.009845. The corrected human-only findings in section 1 are the relevant comparisons.

A one-sided, fixed-benchmark Poisson upper tail reproduces the national 0.005695, suggesting a possible alternate formulation, but the executed worker code is unavailable, so its actual origin remains unverified. No confidence intervals accompany the report's principal rate ratios.

**Proposed remedy:** retain counts, exposures, test function/arguments, direction, confidence intervals, and executed analysis source. Reconcile discrepancies computationally rather than selecting a “conservative” result because both are significant. These tests remain illustrative count-model checks, not validation of surveillance completeness or isolate independence.

### 6. High: geography clarification is identified but never resolved

The planner explicitly says it needs the state, meaning of “current,” and user role. Step 1 finds eight Boone counties and substitutes Missouri because it is largest. The assumption is disclosed, which is useful, but population size does not identify the user's intended county. The current graph proceeds directly from planning to execution and provides no clarification turn.

**Proposed remedy:** make ambiguous geography an explicit unresolved prerequisite for county-specific advice. Historical context under a declared assumption may be useful, but should not become an asserted answer for the user's location.

### 7. Medium: NORS absence is overstated and missing years become zero

The original NORS workbook confirms 38 Missouri records, no direct Boone exposure match, no Boone exposure-flag match through linked counties, and seven suppressed county records. Two Boone county-detail rows are residence-only, correctly excluded from an exposure-location count.

However, this is the Salmonella-focused NORS extract; absence here does not establish absence of outbreaks of “any other etiology,” as the report claims. Exposure geography and resident illness burden are also different outcomes. The generated time-series CSV assigns NORS zero to **2022 and 2023**, beyond its 2021 coverage, and the dashboard extends the flat-zero series across those years. Missing surveillance must not appear as observed zero.

**Proposed remedy:** restrict the conclusion to identified Salmonella exposure outbreaks in this extract and interval; distinguish exposure/residence; use missing values beyond coverage.

### 8. Medium: SVI numbers are correct, but interpretation and version labels need correction

Both original SVI CSVs reproduce the saved Boone values: overall 0.3819/0.4244 and housing/transportation 0.7820/0.7776 for 2020/2022. The 2022 overall percentile is **below**, not “slightly above,” the national median. A high housing percentile also does not, by itself, show that university housing causally drives disease risk or “almost entirely” drives the overall index.

The report and figures reuse older Theme 2/3 labels for 2022. CDC's 2022 service labels these **Household Characteristics** and **Racial & Ethnic Minority Status**. Matching theme numbers across releases is not proof of identical constructs. [CDC 2022 SVI service](https://onemap.cdc.gov/OneMapServices/rest/services/SVI/CDC_ATSDR_Social_Vulnerability_Index_2022_USA/MapServer).

### 9. Medium: plausible surveillance explanations become claimed verification

CDC independently confirms the national PFGE-to-WGS transition in 2019. The report's Missouri “reporting lapse,” and its claim that this explains the county's jump, are not independently verified by the saved evidence. Observing a drop and a later rise supports a hypothesis, not a documented cause. Same-era comparisons likewise do not establish equal ascertainment across geographies. Food-source composition is an additional observed explanation requiring analysis. [CDC PulseNet timeline](https://www.cdc.gov/pulsenet/php/anniversary/timeline.html).

### 10. Medium: the saved actions need their own evidence review

The general Clean/Separate/Cook/Chill recommendations broadly align with [CDC prevention guidance](https://www.cdc.gov/food-safety/prevention/). But recommendations for a lower testing threshold and targeted escalation are justified using the invalid elevated-rate finding. No live advisory check or external retrieval provenance supports the document's “verified” guidance claim.

The action document says seek care for symptoms lasting more than three days; CDC's Salmonella-specific page says contact a doctor for diarrhea or vomiting lasting more than **two days**, as well as blood, fever over 102°F, or dehydration. The saved text includes other warning signs, so this is a threshold discrepancy rather than complete omission. [CDC Salmonella symptoms, “When to talk to your doctor”](https://www.cdc.gov/salmonella/signs-symptoms/).

### 11. Medium: reproducibility and evaluation telemetry are incomplete

`trace.json` retains top-level input/final messages and structured worker summaries, not worker Python code, tool outputs, individual call usage, or aggregator inspection history. The live REPL namespaces are cleaned up at run end. Thus claims of exact recomputation, visual verification, and artifact reading cannot all be audited from persisted tool evidence. Central numbers can be independently checked, as done here; execution summaries alone are not that verification.

`result.json` has null cost, output-token count, judge score, and judge review. The final aggregator call has usage metadata, but 34,882 input/4,096 output tokens are **not the full-run totals**. Model settings and dataset paths are recorded, but the run lacks a complete dirty-code/prompt/input snapshot. This single question/iteration cannot establish benchmark reliability, repeatability, or comparative model quality. The stored conditional follow-up was not executed; the runner submits the main question text only.

### 12. Low to medium: figures and artifact packaging need refinement

All **17 indexed executor artifacts exist** (four figures, six tables, seven documents); the report image resolves and appears in the PDF. The two action files, `actions_placeholder.txt` and `boone_recommended_actions.md`, are byte-identical. There is no actual empty placeholder, but the misleading name and duplication should be avoided.

All five PDF pages are legible and their tables render without clipping. Page 5 contains the dashboard followed by the incomplete caption; the same partial caption pollutes the generated list of figures on page 1. The dashboard's summary text is small at PDF scale. Scientific defects remain visible despite clean rendering: mixed-source rates, inconsistent p-values, and out-of-coverage NORS zeros.

The standalone composite-indicator chart puts outbreak count, relative rate excess, percentile, and proportion on one shared y-axis. Its note discloses different scales, but bar heights still invite an invalid magnitude comparison. Separate panels or a compact indicator table would communicate these distinct quantities better. The standalone time-series chart also uses the full 2011–2022 mean as its “historical” reference, whereas the trend test uses 2011–2018; these should be labeled as different baselines.

## Execution behavior and what worked

The dependency waves ran in order: 1; then 2/3/5/6; then 4; then 7/8; then 9. All nine results returned success, and the log shows no terminal worker failure. The manifest's concurrency limit of three is for independent runs; four workers appearing together within this one run is not evidence of a batch-concurrency violation.

Approximate worker durations: step 1 59s; step 2 145s; step 3 121s; step 5 124s; step 6 48s; step 4 **432s**; step 7 **346s**; step 8 144s; step 9 129s. Executor wall time was approximately 18m 31s. Step 4 and the synthetic-score step dominate the critical path. Logs warn of 54 inherited variables in step 4, 162 in step 7, 158 in step 8, and 192 in step 9. This demonstrates growing context, not a measured memory bottleneck or a proven cause of the analytical errors.

Useful verified behavior includes explicit county ambiguity, partial-2023 exclusion from primary pooled rates, population denominator arithmetic, suppression caveats, correct SVI retrieval, and correct food-insecurity retrieval. The original food-insecurity sheet confirms 11.5% overall, 11.8% child, 20,670 people, and 12.8% in 2019; the 2020 county median of 12.3% and Boone percentile near 41.8 also reproduce. Those correct extractions do not rescue the current-risk conclusion.

## Recommended next work

1. Treat this run as **analytically unsupported and delivery-incomplete**, while preserving its artifacts as a regression case.
2. Fix human/nonhuman population contracts and enforce data-coverage boundaries before calculating risk metrics.
3. Detect generation cutoffs and requested-answer omissions, propagate output settings, and track answer quality separately from exception-free execution.
4. Resolve county identity and temporal scope before county-specific classification; avoid unsupported composite risk bands.
5. Save executable analysis provenance and reconcile exact tests, comparator definitions, and confidence intervals.
6. Rerun this exact question after corrections; validate human-only counts/rates, missing-year handling, complete actions, and terminal stop reason. Repeat runs and an independent review are needed before broader performance claims.

The defensible corrected interpretation is: **assuming Boone County, Missouri, available 2020–2022 human-isolate surveillance does not show the claimed elevated rate relative to state or national benchmarks; these historical data cannot determine November 2026 risk.** This is a review conclusion, not a replacement current-health advisory.

## Verification limits and continuity

Recomputations used the existing local Python environment without LLM inference, AWS calls, dataset downloads, or a full pipeline rerun. The original source workbook also contains unexpected source-type values such as state abbreviations; source schema quality warrants review before treating even human-only surveillance as complete. No malformed records were silently repaired for this audit. The five older PulseNet source workbooks and patient-level deduplication across the full historical corpus were not independently audited. Snapshot paths identify the current local inputs, but the run does not prove immutable input content at execution time.

Official pages actually read for this review: CDC PulseNet timeline (2019 transition), CDC Salmonella symptoms (“When to talk to your doctor”), CDC food-safety prevention, CDC 2022 SVI service theme list, and SciPy `binomtest` parameters/alternative/confidence-interval documentation. [SciPy documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html). External information verifies those specific definitions and guidance; it does not verify this run's local calculations or current Boone outbreak status.

Continuity owner: `agentic-epidemiology`, based on the registered SPARQ repository and verified project Home. This repository review is the delivery record. External Obsidian research-log filing remains pending because that destination is outside this session's writable roots; no external filing is claimed.
