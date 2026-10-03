# Alignment auditor 2.0.2-rc1 status contract

This document describes the supported `check` interface in 2.0.2-rc1. The
comparison and loading functions are unchanged from 2.0.1-rc1. Only the module
documentation, version constant and CLI registration changed: the legacy
`benchmark` command is retired. Historical v1 commands and v2.0.1 evidence keep
their original meanings and hashes. `selfstats` and `make-bundle` are auxiliary
commands, not governed by this `check` result/exit-code contract.

## Output fields

Every `check` JSON result reports four separate concepts:

| Field | Meaning |
|---|---|
| `structural_status` | `valid`, `invalid`, or `invalid_or_unreadable`; label/count consistency and numeric-domain checks only. |
| `identity_evidence` | `exact_match_in_scope`, `within_tolerance`, `mismatch`, `reference_incompatible`, `insufficient_information`, `unresolved`, or `not_evaluated`. |
| `overall_status` | Combined decision. Any structural failure takes precedence over a positive identity result. |
| `scope` | Number of matched gene columns and cells actually fingerprinted, plus `max_cells`. |

`exact_match_in_scope` means every compared matrix entry matched for the
reported matched genes and fingerprinted cells. It is not a claim about
unmatched genes, untested cells, or the full source file. `within_tolerance`
means a nonzero informative mismatch fraction at or below the configured
tolerance. It must never be described as exact identity.

If the union of nonzero entries is empty, identity evidence is
`insufficient_information`; matching zeros do not establish identity. Order,
summary-statistic, annotation, and distribution checks remain supporting
evidence and cannot independently promote a result to exact identity.

## Exit codes (`check`)

| Code | Contract |
|---:|---|
| 0 | Structurally valid and exact match within the reported scope. |
| 1 | Structural failure or supported identity mismatch. |
| 2 | Insufficient support, within-tolerance result, incompatible reference, or unresolved comparison. |
| 3 | Input parsing, file access, or runtime failure. |

The JSON status fields are authoritative for interpretation; the exit code
groups outcomes for shell automation. A malformed input emits a JSON error
object when the CLI can start and returns code 3.

## Serialization and provenance

`check` writes strict JSON (`NaN` and `Infinity` are rejected). Undefined
statistics are represented by `null` with an explanatory status/reason.
Results record the selected H5AD count matrix (`layers/counts` or `X`), the
selected identifier keys, or bundle member paths, SHA256, byte size, dtype,
and shape. Bundle member hashes bind the result to array contents, not just to
the JSON path specification.

The v1 command remains available with its historical semantics. RC1 is a
separate executable and the exit-code change is intentionally breaking.
