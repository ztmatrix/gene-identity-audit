"""Run small actual-subprocess regressions for the alignment v2 check contract."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import shutil
from pathlib import Path

import h5py
import numpy as np


def strict_loads(text: str):
    def reject(token):
        raise ValueError(f'non-standard JSON numeric constant: {token}')
    return json.loads(text, parse_constant=reject)


def build_explicit_selection_fixture(out_dir: Path):
    genes = np.asarray([f'g{i:03d}' for i in range(50)], dtype='<U4')
    cells = np.asarray(['cell-a', 'cell-b', 'cell-c'], dtype='<U6')
    data = np.ones(150, dtype=np.int64)
    indices = np.tile(np.arange(50, dtype=np.int32), 3)
    indptr = np.arange(0, 151, 50, dtype=np.int32)
    x_path = out_dir / 'selector_fixture.h5ad'
    ref_dir = out_dir / 'selector_reference'
    ref_dir.mkdir(parents=True, exist_ok=True)
    np.save(ref_dir / 'data.npy', data)
    np.save(ref_dir / 'indices.npy', indices)
    np.save(ref_dir / 'indptr.npy', indptr)
    np.save(ref_dir / 'genes.npy', genes, allow_pickle=True)
    np.save(ref_dir / 'cells.npy', cells, allow_pickle=True)
    bundle = {'format': 'alignment-bundle-v1',
              'counts': {'data': str(ref_dir / 'data.npy'),
                         'indices': str(ref_dir / 'indices.npy'),
                         'indptr': str(ref_dir / 'indptr.npy'),
                         'shape': [3, 50], 'orientation': 'csr'},
              'genes': str(ref_dir / 'genes.npy'), 'cells': str(ref_dir / 'cells.npy')}
    bundle_path = ref_dir / 'bundle.json'
    bundle_path.write_text(json.dumps(bundle, indent=2) + '\n', encoding='utf-8')
    with h5py.File(x_path, 'w') as f:
        for source, matrix_data in [('X', data), ('layers/counts', np.asarray([], dtype=np.int64))]:
            group = f.require_group(source)
            group.attrs['encoding-type'] = 'csr_matrix'
            group.attrs['shape'] = [3, 50]
            group.create_dataset('data', data=matrix_data)
            group.create_dataset('indices', data=indices if len(matrix_data) else np.asarray([], dtype=np.int32))
            group.create_dataset('indptr', data=indptr if len(matrix_data) else np.asarray([0, 0, 0, 0], dtype=np.int32))
        obs = f.create_group('obs')
        obs.create_dataset('cell_barcode', data=cells.astype(h5py.string_dtype('utf-8')))
        var = f.create_group('var')
        var.create_dataset('ENSEMBL', data=genes.astype(h5py.string_dtype('utf-8')))
    return x_path, bundle_path


def build_invalid_value_fixture(out_dir: Path, name: str, data):
    folder = out_dir / name
    folder.mkdir(parents=True, exist_ok=True)
    indices = np.tile(np.arange(50, dtype=np.int32), 3)
    indptr = np.arange(0, 151, 50, dtype=np.int32)
    genes = np.asarray([f'g{i:03d}' for i in range(50)], dtype='<U4')
    cells = np.asarray(['cell-a', 'cell-b', 'cell-c'], dtype='<U6')
    for key, arr in [('data', data), ('indices', indices), ('indptr', indptr),
                     ('genes', genes), ('cells', cells)]:
        np.save(folder / f'{key}.npy', arr)
    spec = {'format': 'alignment-bundle-v1',
            'counts': {'data': str(folder / 'data.npy'),
                       'indices': str(folder / 'indices.npy'),
                       'indptr': str(folder / 'indptr.npy'),
                       'shape': [3, 50], 'orientation': 'csr'},
            'genes': str(folder / 'genes.npy'), 'cells': str(folder / 'cells.npy')}
    path = folder / 'bundle.json'
    path.write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    root = args.project_root.resolve()
    script = root / 'scripts' / 'audit_release_alignment_v2.py'
    fixture = root / 'docs' / 'reviews_20261003' / 'utility_evidence'
    methodology_fixture = root / 'docs' / 'reviews_20261003' / 'methodology_repro'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    selector_h5ad, selector_bundle = build_explicit_selection_fixture(args.out.parent)
    auto_index_h5ad = args.out.parent / 'auto_index_fixture.h5ad'
    shutil.copy2(selector_h5ad, auto_index_h5ad)
    with h5py.File(auto_index_h5ad, 'r+') as f:
        del f['layers/counts']
        del f['obs/cell_barcode']
        f['obs'].create_dataset('_index', data=np.asarray(['cell-a', 'cell-b', 'cell-c'], dtype=h5py.string_dtype('utf-8')))
        f['obs'].create_dataset('index', data=np.asarray(['NA'] * 3, dtype=h5py.string_dtype('utf-8')))
    malformed_spec = json.loads(selector_bundle.read_text(encoding='utf-8'))
    malformed_spec['counts']['shape'] = [4, 50]
    malformed_bundle = selector_bundle.parent / 'malformed_shape.json'
    malformed_bundle.write_text(json.dumps(malformed_spec, indent=2) + '\n', encoding='utf-8')
    nonfinite_bundle = build_invalid_value_fixture(args.out.parent, 'nonfinite_values',
                                                   np.r_[np.nan, np.ones(149)])
    negative_bundle = build_invalid_value_fixture(args.out.parent, 'negative_values',
                                                 np.r_[-1, np.ones(149, dtype=np.int64)])
    cases = [
        ('auto_anndata_index_before_annotation', auto_index_h5ad, selector_bundle,
         0, 'valid', 'exact_match_in_scope'),
        ('exact', fixture / 'ones_reference' / 'bundle.json',
         fixture / 'ones_reference' / 'bundle.json', 0, 'valid', 'exact_match_in_scope'),
        ('duplicate_cell_ids', fixture / 'duplicate_cell_derived' / 'bundle.json',
         fixture / 'ones_reference' / 'bundle.json', 1, 'invalid', 'structural_failure'),
        ('all_zero', fixture / 'zero_matrix' / 'bundle.json',
         fixture / 'zero_matrix' / 'bundle.json', 2, 'valid', 'insufficient_information'),
        ('unreadable', fixture / 'does_not_exist.json', None, 3,
         'invalid_or_unreadable', 'error'),
        ('review_duplicate_gene_repro', methodology_fixture / 'duplicate.json',
         methodology_fixture / 'reference.json', 1, 'invalid', 'structural_failure'),
        ('non_finite_values', nonfinite_bundle, fixture / 'ones_reference' / 'bundle.json',
         1, 'invalid', 'structural_failure'),
        ('negative_values', negative_bundle, fixture / 'ones_reference' / 'bundle.json',
         1, 'invalid', 'structural_failure'),
        ('malformed_bundle_shape', malformed_bundle,
         fixture / 'ones_reference' / 'bundle.json', 1, 'invalid', 'structural_failure'),
        ('explicit_h5ad_selection', selector_h5ad, selector_bundle, 0,
         'valid', 'exact_match_in_scope',
         ['--matrix-source', 'X', '--gene-key', 'ENSEMBL', '--cell-key', 'cell_barcode']),
    ]
    results = []
    for case in cases:
        name, derived, reference, expected_code, expected_structural, expected_overall = case[:6]
        extra = case[6] if len(case) > 6 else []
        cmd = [sys.executable, str(script), 'check', '--derived', str(derived)]
        if reference is not None:
            cmd.extend(['--reference', str(reference)])
        cmd.extend(extra)
        run = subprocess.run(cmd, cwd=root, text=True, capture_output=True)
        (args.out.parent / f'contract_{name}.json').write_text(run.stdout, encoding='utf-8')
        payload = strict_loads(run.stdout)
        passed = (run.returncode == expected_code
                  and payload.get('structural_status') == expected_structural
                  and payload.get('overall_status') == expected_overall)
        if name == 'all_zero':
            passed = passed and payload.get('fingerprint', {}).get('best_offset') is None
        if name == 'auto_anndata_index_before_annotation':
            passed = passed and payload['effective_input_selection']['derived']['cell_key'] == 'obs/_index'
        if name in ('non_finite_values', 'negative_values', 'malformed_bundle_shape'):
            passed = passed and payload.get('identity_evidence') == 'not_evaluated'
        results.append({'case': name, 'exit_code': run.returncode,
                        'structural_status': payload.get('structural_status'),
                        'identity_evidence': payload.get('identity_evidence'),
                        'overall_status': payload.get('overall_status'),
                        'best_offset': payload.get('fingerprint', {}).get('best_offset'),
                        'effective_input_selection': payload.get('effective_input_selection'),
                        'expected_exit_code': expected_code,
                        'passed': passed,
                        'stderr_excerpt': run.stderr[-400:]})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'tool': 'alignment_v2_contract_regression',
                                    'cases': results,
                                    'all_passed': all(x['passed'] for x in results)},
                                   ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                        encoding='utf-8')
    print(json.dumps({'all_passed': all(x['passed'] for x in results),
                      'case_count': len(results),
                      'output': str(args.out)}, ensure_ascii=False))
    return 0 if all(x['passed'] for x in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
