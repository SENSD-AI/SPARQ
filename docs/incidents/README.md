# Runtime incidents

These records distinguish observed behavior, implementation findings, user reports, and proposed fixes. Recording an incident does not mean it has been resolved.

- [2026-11-04: recursive event-log growth](2026-11-04-recursive-event-log-growth.md): a worker read the live event log through Python; returned namespace artifacts embedded the log back into itself. Final observed size: 24,299,523,616 bytes.
