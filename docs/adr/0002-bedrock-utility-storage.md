# Bedrock utilities, cache and audit evidence

- Date: 2026-10-31
- Status: Adopted in implementation
- Authority: User requested convenient persistent mappings and utility ownership,
  then delegated the storage choice to engineering best practices.

## Decision

Keep Bedrock discovery and interface auditing in `sparq.utils.bedrock`, with matching
tests. Preserve the existing script wrapper and support `python -m sparq.utils.bedrock`.

Store the latest complete inference-profile mapping in the OS-native user cache,
using `platformdirs`, scoped by AWS profile and region. This mapping is generated,
replaceable metadata rather than user-authored configuration. Each command still
discovers fresh metadata; the stored mapping is a reusable lookup, not an authority
for availability or permission. Incomplete discovery preserves the previous complete
cache. A cache-write failure warns while preserving the per-audit snapshot.

Store new audit evidence in the Git-ignored project directory
`artifacts/bedrock/<profile>/<region>/audits/<id>/`. Keep a relative latest-audit
pointer beside the audit folders. Explicit `--output` destinations retain their
exact-directory semantics; `--cache-dir` overrides the OS cache root.

## Consequences

The reusable mapping survives branch and checkout changes. Audit evidence stays
with the project and remains available for inspection. Existing historical bundles
under `evaluation/results/bedrock_models/` are preserved. Tests isolate all cache
writes in temporary directories. Concurrent cache refreshes use unique temporary
files and atomic replacement.

Sources read: [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir/latest/)
(cache versus configuration) and [platformdirs API](https://platformdirs.readthedocs.io/en/latest/api.html#user-cache-directory)
(OS-native user-cache paths). Project-local audit evidence is an implementation
choice for this research repository, not an XDG requirement.
