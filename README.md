# Gene-identity alignment audit (2.0.2-rc1)

Reference-supported auditing of gene–count alignment in reusable single-cell
expression matrices. The tool checks whether the count columns of a processed
matrix correspond to the gene identifiers claimed for them, using an
independently obtained reference matrix.

It reports evidence, not a repair: exit 0 = supported exact agreement in scope,
1 = structural failure or a supported identity mismatch, 2 = non-exact,
incompatible or unresolved, 3 = read/runtime error. Strict JSON output; undefined
statistics are null. The tool never relabels source data.

## Install and run

Tested on Windows with Python 3.14 (Linux execution not yet established).

```bash
python -m pip install -r requirements-alignment-core.txt
python scripts/verify_alignment_v2_contract.py --out replay/contract_regression.json
python scripts/verify_alignment_v2_boundary_cases.py --out-dir replay/boundary_cases
```

Expected: 10/10 contract cases and 8/8 boundary cases pass. Each test launches the
actual CLI and checks the expected status and exit code; a deliberately failing
case reports exit 1 while the suite itself exits 0.

For a real comparison, obtain a provenance-matched reference, preserve the
original files, then:

```bash
python scripts/audit_release_alignment_v2.py check --help
python scripts/audit_release_alignment_v2.py check \
    --derived <processed.h5ad> --reference <reference-bundle.json> \
    --max-cells 300 --json <result.json>
```

Inspect the matrix layer and identifier keys actually used, the matched scope, and
the three reported components (`structural_status`, `identity_evidence`,
`overall_status`). `selfstats` is an auxiliary internal-statistics diagnostic and
does not certify count identity; `make-bundle` prepares reference inputs. The
older `benchmark` subcommand is retired from the supported CLI.

## What is in this repository

- `scripts/` — the auditor, an independent baseline comparison, and the contract
  and boundary verification suites.
- `replay/` — synthetic inputs, reference bundles and expected outputs that
  reproduce the reported contract and boundary results.
- `docs/` — status contract, output schema, threshold notes and review fixtures.
- `package_manifest.json` — file list with content hashes.

All measurement matrices here are artificial. **No third-party single-cell data
and no human measurements are redistributed in this repository.** Real-data
comparisons in the associated manuscript are reproduced from their public sources;
the derived reference arrays are not included.

## Data and reference requirements

A trustworthy result needs an appropriate reference for the same biological
material. A self-consistent processed matrix cannot certify its own identifiers,
and normalized values support ordering diagnostics only, not exact count
identity. Scope, sampling and reference provenance must accompany any reused
result.

## Citation

See `CITATION.cff`. Manuscript in preparation: *Reference-supported auditing of
gene–count alignment in reusable single-cell datasets*.

Repository: https://github.com/ztmatrix/gene-identity-audit (author to confirm the account before publication).

## License

MIT — see `LICENSE`. Copyright (c) 2026 Tao Zhang.

## Reproducibility note

The committed JSONs under `replay/` are a snapshot of one run in this repository
root. Re-running the suites changes three things that are not reproducibility
signals: the recorded wall-clock `runtime_s` of each check, absolute path strings,
and the content hashes derived from those paths (reference bundle JSONs embed the
array paths). The stable, checkable outcome is the reported statuses and exit
codes for the fixed-seed inputs: 10/10 contract cases and 8/8 boundary cases pass.
Byte-identical output is not expected and is not the acceptance criterion.
