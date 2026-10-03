# W2 delivery: thresholds and boundary regressions

Date: 2026-10-03. Synthetic tests in this report are development challenges,
not an independent external validation set.

## Decision thresholds carried into v2

| Rule | Value in v2 | Interpretation |
|---|---:|---|
| Minimum matched genes | 50 | Below this, no fingerprint. |
| Minimum matched cells | 3 | Below this, no fingerprint. |
| Fingerprint cells | 300 by default | Evenly spaced over matched cell pairs; this does not limit file loading or full-file structural checks. |
| Gene-label offset search | −3 through +3 | Both endpoints included; values outside this search are not identified as a specific offset. |
| Candidate offset in-range coverage | ≥90% | Lower coverage makes that offset inapplicable. |
| Tolerated informative mismatch | ≤5% | A nonzero result is `within_tolerance`, never exact. |
| Positional cell-order Spearman | ≥0.99 | Only when positional matching is explicitly enabled and identifier overlap is insufficient. |
| Normalized order identity Spearman | ≥0.80, within 0.05 of best tested order | Supporting order evidence; it does not establish exact count identity. |
| Normalized order offset | nonidentity best ≥0.80 and ≥0.05 above identity | Candidate order shift, still a proxy rather than exact integer-count validation. |
| Normalized order mismatch | identity and best <0.50 | Order disagreement evidence. |
| Per-cell value multiset fraction | ≥0.50 | Supporting evidence for value-preserving label mismatch; not sufficient by itself. |
| Cell outlier cutoff | `max(0.25, 5 × median per-cell informative mismatch)` | Supporting cell-label mismatch diagnostic. |
| Integer offset evidence | identity mismatch informative fraction ≥0.20 and best nonidentity fraction ≤5% of identity fraction | A specific offset may explain the observed identity mismatch. |

The offset is first selected by minimum raw mismatch count across the tested
scope, then assessed using each candidate's own informative-union denominator.
The denominator and both raw and informative fractions are retained in the
output. With zero nonzero-union support, the result is insufficient and the
best offset is unset.

## Actual CLI boundary challenges

`scripts/verify_alignment_v2_boundary_cases.py` generates a fixed-seed,
4-cell × 200-reference-gene artificial matrix and runs eight cases through
the CLI. The latest run passes 8/8; outputs are in
`D:/mito_alignment_cache/paper_revision_v2_20261003/boundary_cases/`.

| Challenge | Result | What it establishes |
|---|---|---|
| Exact supported identity | exit 0, exact_match_in_scope | Exactness uses compared matrix values and matched labels. |
| +1 offset | exit 1, gene_label_offset(+1) | Correctly detects the simulated offset within the configured search. |
| One altered count in 400 supported entries | exit 2, within_tolerance | A nonzero 0.25% discrepancy is not called exact. |
| Three pairwise count-column permutations with unique IDs retained | exit 1, gene_label_mismatch | Detects a local value-preserving relabeling above the 5% informative threshold. |
| Independent values | exit 2, reference_incompatible | Does not attribute an incompatible matrix to a label error. |
| Legal gene-column reordering with labels following the columns | exit 0, exact_match_in_scope | Identifier matching is invariant to valid column order. |
| Shift beyond the ±3 search with changed matched-scope values | exit 2, reference_incompatible | The tool does not claim a specific unsearched offset. |
| Values scaled by 0.5 | exit 2, unresolved; fingerprint `aligned_order` | Rank agreement remains supporting evidence and is not promoted to exact identity. |

This suite verifies expected behavior of the current implementation on the
stated inputs. It does not estimate sensitivity, false-positive rates, or
performance on a population of public datasets. The offset-outside-window case
is deliberately reported as reference-incompatible because the observed
matched-scope per-cell value multisets differ; it is not proof of a real
offset beyond the search window.

## Paired reproduction of review counterexamples

The reviewer-created CLI fixtures have now been rerun through v2. Their v1
outputs are preserved in `docs/reviews_20261003/utility_evidence/` and
`docs/reviews_20261003/methodology_repro/`.

| Counterexample | v1 behavior | v2 behavior |
|---|---|---|
| Duplicate identifiers with otherwise identical data | Fingerprint `aligned`, exit 0, despite `passes_all_applicable=false`; JSON contains nonstandard NaN for undefined correlations. | `identity_evidence=exact_match_in_scope` retained as a scoped comparison fact, but `structural_status=invalid`, `overall_status=structural_failure`, exit 1; strict JSON. |
| All-zero pair | `aligned`, exit 0, `best_offset=-3`, no informative support, nonstandard NaN. | `insufficient_information`, exit 2, best offset unset, strict JSON. |
| Duplicate gene identifier review fixture | Top-level `aligned`, exit 0 while structural checks fail. | Structural failure takes precedence, exit 1. |
| Bundle shape contradicts its labels | V1 loader rejects the bundle before producing an integrity status. | Classified as invalid structure with identity not evaluated, overall structural failure, exit 1. |

The original reviewer fixtures plus the new synthetic boundary set and three
real files have therefore been rerun through the v2 CLI. The nine small contract
cases pass 9/9 and the boundary set passes 8/8. These are targeted regressions,
not a prevalence estimate or an external validation cohort. Future changes to
the stated contract require an explicitly versioned update to the expected
behavior and its evidence.
