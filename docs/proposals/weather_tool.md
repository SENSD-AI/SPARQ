# Weather Tool — Design Notes and Complexities

Design notes for adding a weather-lookup capability to the executor's tool list, so
SPARQ can address weather-related questions (e.g. correlating outbreak timing with
historical conditions). Nothing here is implemented yet — this documents the open
questions and constraints surfaced while scoping the work.

---

## Plain tool, not MCP

`architectures/v1/nodes/executor.py` builds a flat list of `@tool`-decorated LangChain
functions and hands them to `create_agent` (see `filesystemtools`, `data_discovery_tools`,
`python_repl_tool`). A weather capability fits the same pattern: a small `@tool` function
calling a REST API, added to that list.

Consuming an external weather *MCP server* (via `langchain-mcp-adapters`) is unnecessary
extra surface — process management, auth — for what is a single stateless HTTP call.
Only worth it if reusing a specific existing MCP server is a hard requirement.

**Recommended API:** [Open-Meteo](https://open-meteo.com/) — no API key, no signup,
generous free-tier rate limits. Has two relevant endpoints:
- `api.open-meteo.com` — forecast / current conditions
- `archive-api.open-meteo.com` — historical weather (ERA5 reanalysis, back to 1940)

For SPARQ's epidemiology use case, the historical/archive endpoint is the one that
matters — queries are almost always "what was the weather in X around date Y," not
current conditions.

---

## Complexity: no dataset carries lat/long

Checked every dataset in `src/sparq/data/data_summaries_short.json`. None expose
coordinates directly — all are keyed by county/state name or FIPS code:

| Dataset | Location fields | Date granularity |
|---|---|---|
| `pulsenet` | `SourceState`, `SourceCounty` | `IsolatDate` — real date |
| `nors` | `ExposureState`, `ExposureCounty` | `yearfirstill`, `monthfirstill` — **no day** |
| `social_vulnerability_index` | `FIPS`, `STATE`, `COUNTY` | year only (2020, 2022 snapshots) |
| `census_population` | county name (no FIPS) | year only |
| `map_the_meal_gap` | `FIPS`, `State`, `County` | year only |

Open-Meteo's APIs take lat/long, not place names. So the tool needs a geocoding step
before it can call the weather API — it can't just be handed a county/state string.

**Options for geocoding:**
- Open-Meteo's free Geocoding API (`geocoding-api.open-meteo.com`) — matches by name,
  ambiguous for common county names ("Springfield", "Washington") unless queried as
  `"<county>, <state>"` and the correct match is picked from results.
- A static FIPS→centroid lookup table (e.g. bundled US Census county centroids) — more
  reliable since it's keyed by the same FIPS code several datasets already use, no
  network call, no name-matching ambiguity.

Leaning toward the static FIPS lookup for datasets that have FIPS (`social_vulnerability_index`,
`map_the_meal_gap`), and geocoding-by-name as a fallback for `pulsenet`/`nors`, which
only carry county/state strings, not FIPS.

---

## Complexity: date granularity mismatch across datasets

`pulsenet` has a real date (`IsolatDate`), so a day-level historical weather query is
possible. `nors` only has `yearfirstill`/`monthfirstill` — day-level weather isn't
meaningful there; the tool would need to return a monthly aggregate (e.g. mean
temperature/precipitation over the month) rather than a single day's reading.

This means the tool signature should accept either an exact date or a `(year, month)`
pair, and branch internally between a single-day lookup and a monthly-aggregate lookup —
the executor agent shouldn't have to know which dataset it's working from to pick the
right call shape.

---

## Complexity: planner/router don't know this capability exists

- **Planner** (`architecture/v1/nodes/planner.py`) reasons only over `data_manifest.json`
  / `data_summaries_short.json` when deciding plan steps. It won't propose a weather
  step unless the planner's system prompt is told the capability exists.
- **Router** (`architecture/v1/nodes/router.py`) classifies queries as needing data
  analysis (→ planner/executor) or answerable directly (→ no tool access at all). A
  pure weather question ("what's the weather in Miami today") needs to route to the
  `True`/executor branch even though it touches none of the epi datasets — the router
  prompt likely needs updating so it doesn't assume "needs data" means "needs one of
  the manifest datasets."

---

## Open questions (not yet decided)

- One combined tool that hides geocoding internally, vs. two tools
  (`geocode_county` + `get_historical_weather`) that the executor agent chains itself.
- Whether to bundle a static FIPS-centroid table as a data file vs. calling the
  geocoding API every time.
- Whether monthly-aggregate weather (for `nors`) is computed by calling the archive API
  once per day in the month and averaging, or whether Open-Meteo exposes a monthly
  aggregate endpoint directly (needs checking).
