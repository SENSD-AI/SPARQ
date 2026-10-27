# Factual audit of the dataset inventory report

- Reviewer: assistant
- Audit date: 27 October 2026
- Human adjudication: pending
- Original run ID: `07dd64b0-3469-4fd8-9e10-311f02e8a015`
- Original query: What datasets do we have and what do they represent?
- Original run directory: `/Users/zayan/tmp/sparq/output/27-10-2026_13-07-52`
- Saved duration: 438.014856 seconds
- Report: [Markdown](run/report/report.md) and [PDF](run/report/report.pdf)

## Assessment

The report contains clear factual errors and unsupported quantitative claims and
should not be treated as a verified dataset inventory. Its broad inventory is
recognizable: direct checks confirmed 582,022 PulseNet rows across six exports,
3,950 NORS main-sheet records, 39 NORS sheets, and a nominal common data year of
2020 based on the stated coverage. These correct facts do not validate the invented
schema details, numerical assessments, or claims of comprehensive verification.

## Inspection scope

The full generated Markdown was read and text was extracted from all 10 PDF pages.
All three source figures were inspected visually, as was the rendered PDF page
containing the quality-assessment figure. The saved state, result metadata, and
artifact inventory were inspected; relevant sections of the worker documents were
read. The combined initial worker-document read was truncated, so this audit does
not claim a complete read of both worker documents. The original run log was
searched, not read as a complete tool transcript.

Local verification included:

- All six PulseNet exports scanned for nonempty rows and aggregate date/location/
  source fields; headers checked. Non-datetime date values were not fully parsed.
- NORS `NORSMain` loaded in full for row count, schema, year range, unique CDCID,
  and PrimaryMode counts; the complete workbook sheet-name list was checked.
- Census national workbook loaded using `header=3`; county rows distinguished from
  its national aggregate and footer text.
- Both SVI CSVs loaded for shape, identifiers, and indicator counts; raw FIPS text
  checked; Connecticut identifiers compared between versions.
- All 12 MMG workbook/sheet inventories inspected; detailed rate-field, row/year,
  and FIPS-cell checks sampled the 2009, 2018, and combined 2019/2020 workbooks.
- NACo local file inventory checked for tabular data.
- Official CDC/ATSDR SVI introductions and relevant variable/theme material and
  Feeding America methodology material checked. The complete SVI documentation
  PDFs were not read. See [public sources](evidence/public_sources.json).

No new model pipeline run or empirical geographic join was performed. Local source
fingerprints were recorded when the bundle was collected. See the structured
[verified facts](evidence/verified_facts.json) and [input manifest](evidence/input_files_manifest.json).

## Findings

### F01: NORS schema contains nonexistent fields and sheets

- Classification: `confirmed_error`
- Severity: high (assistant judgment, not calibrated)
- Location: [original report](run/report/report.md), lines 85–87

**Claim:** IllCount, HospCount, DeathCount; Foods, Settings, Water

**Correction or qualification:** Use the actual workbook schema: TotalCases, HospitalNum, DeathsNum; FB_FoodMain, FB_FoodImplicated, Setting.

**Origin assessment:** generated worker documentation; not present in the supplied short summary. This is an inference from the saved materials, not a recovered tool-call history.

### F02: All NORS records called foodborne outbreaks

- Classification: `confirmed_error`
- Severity: high (assistant judgment, not calibrated)
- Location: [original report](run/report/report.md), lines 83–83

**Claim:** 3,950 documented foodborne disease outbreak events

**Correction or qualification:** Report 3,950 records, of which 3,191 are Food, 266 Animal Contact, and 493 Indeterminate/Other/Unknown.

**Origin assessment:** generated worker documentation; supplied summary said Salmonella outbreaks. This is an inference from the saved materials, not a recovered tool-call history.

### F03: Census loaded rows mislabeled as counties

- Classification: `confirmed_error`
- Severity: medium (assistant judgment, not calibrated)
- Location: [original report](run/report/report.md), lines 101–101

**Claim:** 3,151 US counties

**Correction or qualification:** There are 3,144 county rows, one national aggregate, and six footer rows.

**Origin assessment:** inherited from supplied summary and repeated as a verified county count. This is an inference from the saved materials, not a recovered tool-call history.

### F04: Incorrect SVI factor count, themes, and uniform row count

- Classification: `confirmed_error`
- Severity: medium (assistant judgment, not calibrated)
- Location: [original report](run/report/report.md), lines 132–134

**Claim:** 15 social factors; 3,143 counties for both years; older theme names

**Correction or qualification:** Use 16 factors; 3,143 county rows in 2020 and 3,144 in 2022; Household Characteristics and Racial & Ethnic Minority Status for themes 2 and 3.

**Origin assessment:** row count inherited from summary; factor/theme descriptions added in worker documentation. This is an inference from the saved materials, not a recovered tool-call history.

### F05: Numerical quality and join scores lack documented derivation

- Classification: `unsupported_quantification`
- Severity: high (assistant judgment, not calibrated)
- Location: [original report](run/report/report.md), lines 175–183

**Claim:** 0–1 data-quality scores and Poor/Fair/Excellent join reliability scores

**Correction or qualification:** Supply a reproducible rubric and measurements or label these explicitly as subjective illustrations; do not treat them as measured quality.

**Origin assessment:** generated figure and worker documentation; calculation provenance not retained. This is an inference from the saved materials, not a recovered tool-call history.

### F06: Unsupported empirical Census match-rate claim

- Classification: `unsupported_quantification`
- Severity: high (assistant judgment, not calibrated)
- Location: [original report](run/report/report.md), lines 216–216

**Claim:** typically yields <95% match rates

**Correction or qualification:** Remove the percentage unless a measured join result or suitable source supports it.

**Origin assessment:** the worker document recommends aiming above 95%; the report turns this into a typical observed rate. This is an inference from the saved materials, not a recovered tool-call history.

### F07: FIPS source storage conflated with parser inference

- Classification: `confirmed_error`
- Severity: medium (assistant judgment, not calibrated)
- Location: [original report](run/report/report.md), lines 128–145

**Claim:** SVI is stored as integer; MMG is consistently zero-padded strings

**Correction or qualification:** The checked SVI CSVs contain 01001; default pandas inference converts it. MMG cells vary between strings and numbers, so configure and validate imports consistently.

**Origin assessment:** supplied summary partly conflates dtype inference with source representation. This is an inference from the saved materials, not a recovered tool-call history.

### F08: Undefined combined record total

- Classification: `invalid_aggregation`
- Severity: medium (assistant judgment, not calibrated)
- Location: [original report](run/report/report.md), lines 39–39

**Claim:** ~592,000 total records across five datasets

**Correction or qualification:** Report counts separately by observational unit and year. The table mixes isolates, outbreaks, county rows, and county-year counts; its naive sum is approximately 595,409, not 592,000.

**Origin assessment:** worker-generated total repeated downstream. This is an inference from the saved materials, not a recovered tool-call history.

### F09: Calendar-year counts inconsistent with inclusive counting

- Classification: `confirmed_error`
- Severity: low (assistant judgment, not calibrated)
- Location: [original report](run/report/report.md), lines 167–168

**Claim:** 1996–2023: 27 years; 1998–2021: 23 years

**Correction or qualification:** These ranges contain 28 and 24 calendar years respectively. A 27- or 23-year elapsed difference should not be mixed with inclusive counts used elsewhere.

**Origin assessment:** worker documentation arithmetic repeated downstream. This is an inference from the saved materials, not a recovered tool-call history.

### F10: MMG suggested column search misses a real rate field

- Classification: `confirmed_error`
- Severity: medium (assistant judgment, not calibrated)
- Location: [original report](run/report/report.md), lines 118–124

**Claim:** Find rate columns by searching for insecurity

**Correction or qualification:** The checked 2009 workbook uses FI Rate, which does not contain insecurity. Use inspected, year-specific field mappings or validated alternatives.

**Origin assessment:** inherited from supplied summary. This is an inference from the saved materials, not a recovered tool-call history.

### F11: Integration advice omits geographic-vintage reconciliation

- Classification: `methodological_limitation`
- Severity: medium (assistant judgment, not calibrated)
- Location: [original report](run/report/report.md), lines 195–206

**Claim:** FIPS standardization makes geographic integration straightforward

**Correction or qualification:** Reconcile the eight Connecticut county FIPS in 2020 with nine planning-region FIPS in 2022 before treating cross-vintage joins as comparable. This is a missing qualification, not a failed measured join.

**Origin assessment:** omission in recommended integration workflow. This is an inference from the saved materials, not a recovered tool-call history.

### F12: Methods overstate inspection and verification

- Classification: `unsubstantiated_verification`
- Severity: medium (assistant judgment, not calibrated)
- Location: [original report](run/report/report.md), lines 49–49

**Claim:** Each dataset was inspected for ... data quality ... completeness

**Correction or qualification:** Distinguish supplied summaries from direct checks and preserve tool-level evidence. Saved trace messages contain only human/AI final messages; specific schema/count errors contradict blanket verification.

**Origin assessment:** saved worker result also asserts all numeric claims were verified; exact tool history unavailable. This is an inference from the saved materials, not a recovered tool-call history.

## Additional observations

- PulseNet's report lists a `PatientAge` field. Actual exports use
  `PatientAgeDays`, `PatientAgeMonths`, and `PatientAgeYears`.
- Figure 2 draws a continuous 2020–2022 SVI interval despite the local collection
  containing only the 2020 and 2022 snapshots. Figure 1 represents snapshots more
  accurately. The report also calls the gray concurrent-availability line
  “cumulative,” although that line decreases after 2020.
- The 158 SVI columns are not all derived vulnerability measures: they also include
  identifiers, estimates, margins of error, rankings, and flags.
- The SVI reference URL printed in the report returned HTTP 404 during the audit.
  The Feeding America URL could not be accessed by the browsing tool; that failure
  alone does not establish a dead link. Updated working documentation links are
  listed in `evidence/public_sources.json`.
- “No local NACo tabular data” is supported. This describes the local cache and
  should not be generalized to an assertion that NACo has no data anywhere.

## Provenance and evaluation limits

The supplied short summaries already contain the Census row count presented as
county count, a single SVI row count, MMG column-search advice, and a misleading
SVI integer-storage description. The worker documents add further assertions,
including nonexistent schema details and unsupported assessments; the final report
repeats or strengthens some of them. In particular, a suggested >95% validation
target in the worker document becomes a claim about typical <95% performance in
the final report.

The saved worker result says all numerical claims were verified and no synthetic
data were used. That statement is not independent verification. Exact numerical
quality scores lack a saved derivation; their unsupported status should not be
conflated with proof of deliberate fabrication. Definitive causal attribution
requires the missing worker/aggregator tool-call history.

This audit distinguishes confirmed errors, unsupported quantification, invalid
aggregation, and methodological limitations. It is a reference candidate requiring
human review before becoming a scoring standard. The original report, run outputs,
and source datasets have not been edited.
