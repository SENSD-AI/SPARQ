# Bedrock utilities, cache and audit evidence

- Date: 2026-10-31
- Updated: 2026-11-01
- Status: Adopted in implementation
- Authority: User requested convenient persistent mappings and utility ownership,
  then explicitly selected repository-level `utils/bedrock_utils/` with local `outputs/`.

## Decision

Keep Bedrock discovery and interface auditing in `utils.bedrock_utils`, with matching
tests. Preserve the existing script wrapper and support `python -m utils.bedrock_utils`.

Store the latest complete inference-profile mapping in the OS-native user cache,
using `platformdirs`, scoped by AWS profile and region. This mapping is generated,
replaceable metadata rather than user-authored configuration. Each command still
discovers fresh metadata; the stored mapping is a reusable lookup, not an authority
for availability or permission. Incomplete discovery preserves the previous complete
cache. A cache-write failure warns while preserving the per-audit snapshot.

Store new audit evidence in the Git-ignored project directory
`utils/bedrock_utils/outputs/<profile>/<region>/audits/<id>/`. Keep a relative latest-audit
pointer beside the audit folders. Explicit `--output` destinations retain their
exact-directory semantics; `--cache-dir` overrides the OS cache root.

## Consequences

The reusable mapping survives branch and checkout changes. Audit evidence stays
beside the utility code and remains available for inspection. Previous audit bundles
were deleted at the user’s request before this relocation. The utility is not part
of the installed SPARQ library; module commands run from the checkout root. Tests isolate all cache
writes in temporary directories. Concurrent cache refreshes use unique temporary
files and atomic replacement.

Sources read: [XDG Base Directory Specification](https://specifications.freedesktop.org/basedir/latest/)
(cache versus configuration) and [platformdirs API](https://platformdirs.readthedocs.io/en/latest/api.html#user-cache-directory)
(OS-native user-cache paths). Project-local audit evidence is an implementation
choice for this research repository, not an XDG requirement.
