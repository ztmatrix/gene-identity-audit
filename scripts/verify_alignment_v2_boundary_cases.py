"""Run frozen, synthetic CLI challenges for alignment v2 decision boundaries."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


def write_bundle(folder: Path, cells, genes, matrix):
    folder.mkdir(parents=True, exist_ok=True)
    x = csr_matrix(matrix)
    for key, arr in [('data', x.data), ('indices', x.indices), ('indptr', x.indptr),
                     ('genes', np.asarray(genes, dtype=str)), ('cells', np.asarray(cells, dtype=str))]:
        np.save(folder / f'{key}.npy', arr)
    spec = {'format': 'alignment-bundle-v1',
            'counts': {'data': str(folder / 'data.npy'),
                       'indices': str(folder / 'indices.npy'),
                       'indptr': str(folder / 'indptr.npy'),
                       'shape': list(x.shape), 'orientation': 'csr'},
            'genes': str(folder / 'genes.npy'), 'cells': str(folder / 'cells.npy')}
    (folder / 'bundle.json').write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    return folder / 'bundle.json'


def strict_loads(text):
    return json.loads(text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out-dir', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.project_root.resolve(), args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    script = root / 'scripts' / 'audit_release_alignment_v2.py'
    cells = [f'cell{i}' for i in range(4)]
    ref_genes = [f'g{i:03d}' for i in range(200)]
    rng = np.random.default_rng(7132026)
    ref = rng.integers(1, 100, size=(4, 200), dtype=np.int64)
    ref_path = write_bundle(out / 'reference', cells, ref_genes, ref)
    base = ref[:, 50:150].copy()
    labels = ref_genes[50:150]
    cases = []

    def add(name, matrix, names, status, identity, code, tolerance=0.05):
        derived = write_bundle(out / name, cells, names, matrix)
        cases.append((name, derived, status, identity, code, tolerance))

    add('exact', base, labels, 'exact_match_in_scope', 'exact_match_in_scope', 0)
    add('offset_plus1', ref[:, 51:151], labels, 'identity_mismatch', 'mismatch', 1)
    tiny = base.copy(); tiny[0, 0] += 1
    add('within_tolerance', tiny, labels, 'within_tolerance', 'within_tolerance', 2)
    local = base.copy()
    for a, b in [(0, 1), (10, 11), (20, 21)]:
        local[:, [a, b]] = local[:, [b, a]]
    add('unique_id_local_reordering', local, labels, 'identity_mismatch', 'mismatch', 1)
    independent = rng.integers(101, 200, size=(4, 100), dtype=np.int64)
    add('reference_incompatible', independent, labels, 'reference_incompatible',
        'reference_incompatible', 2)
    permutation = np.arange(100)[::-1]
    add('legal_gene_column_reordering', base[:, permutation], [labels[i] for i in permutation],
        'exact_match_in_scope', 'exact_match_in_scope', 0)
    beyond = ref[:, 54:154]
    beyond_labels = [ref_genes[i] for i in range(50, 150)]
    add('offset_outside_search_window', beyond, beyond_labels, 'reference_incompatible',
        'reference_incompatible', 2)
    add('normalized_proxy', base.astype(float) * 0.5, labels, 'unresolved', 'unresolved', 2)

    records = []
    for name, derived, expected_overall, expected_identity, expected_code, tolerance in cases:
        result_path = out / f'{name}.json'
        command = [sys.executable, str(script), 'check', '--derived', str(derived),
                   '--reference', str(ref_path), '--mismatch-frac', str(tolerance),
                   '--json', str(result_path)]
        run = subprocess.run(command, cwd=root, text=True, capture_output=True)
        payload = strict_loads(run.stdout)
        fp = payload.get('fingerprint', {})
        passed = (run.returncode == expected_code
                  and payload.get('overall_status') == expected_overall
                  and payload.get('identity_evidence') == expected_identity)
        records.append({'case': name, 'exit_code': run.returncode,
                        'expected_exit_code': expected_code,
                        'structural_status': payload.get('structural_status'),
                        'identity_evidence': payload.get('identity_evidence'),
                        'overall_status': payload.get('overall_status'),
                        'fingerprint_verdict': fp.get('verdict'),
                        'identity_informative_mismatch_fraction': fp.get('identity_informative_mismatch_fraction'),
                        'best_offset': fp.get('best_offset'),
                        'best_offset_informative_mismatch_fraction': fp.get('best_offset_informative_mismatch_fraction'),
                        'stderr_excerpt': run.stderr[-300:], 'passed': passed})
    summary = {'suite': 'alignment_v2_boundary_cases', 'seed': 7132026,
               'matrix_shape_reference': [4, 200], 'derived_columns': 100,
               'cases': records, 'all_passed': all(r['passed'] for r in records)}
    (out / 'boundary_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'case_count': len(records), 'all_passed': summary['all_passed'],
                      'summary': str(out / 'boundary_summary.json')}))
    return 0 if summary['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
