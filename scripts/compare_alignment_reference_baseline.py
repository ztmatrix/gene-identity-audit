"""Independent exact stable-ID join baseline; does not import the auditor."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


def sha256_file(path: Path, block=1 << 24):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def read_labels(node):
    if hasattr(node, 'asstr'):
        return np.asarray(node.asstr()[:], dtype=str)
    if 'categories' in node and 'codes' in node:
        cats = node['categories'].asstr()[:]
        codes = np.asarray(node['codes'][:])
        return np.asarray([cats[c] if 0 <= c < len(cats) else '' for c in codes], dtype=str)
    raise ValueError('unsupported H5AD label encoding')


def read_h5ad(path: Path):
    with h5py.File(path, 'r') as f:
        matrix_key = 'layers/counts' if 'layers/counts' in f else 'X'
        node = f[matrix_key]
        if isinstance(node, h5py.Dataset):
            matrix = csr_matrix(node[:])
        else:
            shape = tuple(int(v) for v in node.attrs['shape'])
            data = np.asarray(node['data'][:])
            indices = np.asarray(node['indices'][:])
            indptr = np.asarray(node['indptr'][:])
            if 'csc' in str(node.attrs.get('encoding-type', '')):
                matrix = csc_matrix((data, indices, indptr), shape=shape).tocsr()
            else:
                matrix = csr_matrix((data, indices, indptr), shape=shape)
        gene_key = next((f'var/{key}' for key in
                         ('ENSEMBL', 'gene_ids', 'ensembl_id', 'gene_id', 'index', '_index')
                         if f'var/{key}' in f), None)
        cell_key = next((f'obs/{key}' for key in
                         ('cell_barcode', 'cell_barcodes', 'barcode', 'barcodes',
                          'bc_wells', 'cell_id', 'index', '_index')
                         if f'obs/{key}' in f), None)
        if gene_key is None or cell_key is None:
            raise ValueError('could not locate H5AD gene and cell identifiers')
        genes, cells = read_labels(f[gene_key]), read_labels(f[cell_key])
    if matrix.shape != (len(cells), len(genes)):
        raise ValueError('count matrix shape does not match H5AD identifiers')
    return matrix, genes, cells, matrix_key, gene_key, cell_key


def read_bundle(path: Path):
    spec = json.loads(path.read_text(encoding='utf-8'))
    counts = spec['counts']
    arrays = {k: np.load(Path(counts[k]), mmap_mode='r') for k in ('data', 'indices', 'indptr')}
    genes = np.asarray(np.load(Path(spec['genes']), allow_pickle=True), dtype=str)
    cells = np.asarray(np.load(Path(spec['cells']), allow_pickle=True), dtype=str)
    shape = tuple(int(v) for v in counts['shape'])
    if counts.get('orientation', 'csc') == 'csc':
        matrix = csc_matrix((arrays['data'], arrays['indices'], arrays['indptr']), shape=shape)
        if shape != (len(genes), len(cells)):
            raise ValueError('CSC bundle shape does not match gene/cell labels')
        orientation = 'csc_genes_by_cells'
    else:
        matrix = csr_matrix((arrays['data'], arrays['indices'], arrays['indptr']), shape=shape)
        if shape != (len(cells), len(genes)):
            raise ValueError('CSR bundle shape does not match cell/gene labels')
        orientation = 'csr_cells_by_genes'
    return matrix, genes, cells, orientation


def stable_gene_id(value):
    text = str(value)
    if text.startswith('ENSG') and '.' in text:
        return text.split('.', 1)[0]
    return text


def unique_positions(labels, normalizer=lambda value: str(value)):
    positions = {}
    for i, label in enumerate(labels):
        positions.setdefault(normalizer(label), []).append(i)
    return {label: ix[0] for label, ix in positions.items() if len(ix) == 1}


def compare(derived_path: Path, reference_path: Path, max_cells=300,
           v2_result_path: Path | None = None):
    started = time.perf_counter()
    derived, dgenes, dcells, matrix_key, gene_key, cell_key = read_h5ad(derived_path)
    reference, rgenes, rcells, orientation = read_bundle(reference_path)
    dgene_unique = unique_positions(dgenes, stable_gene_id)
    rgene_unique = unique_positions(rgenes, stable_gene_id)
    dcell_unique, rcell_unique = unique_positions(dcells), unique_positions(rcells)
    gene_pairs = [(dpos, rgene_unique[name]) for name, dpos in dgene_unique.items()
                  if name in rgene_unique]
    gene_pairs.sort()
    cell_pairs = [(dpos, rcell_unique[name]) for name, dpos in dcell_unique.items()
                  if name in rcell_unique]
    cell_pairs.sort()
    subsampled = False
    if len(cell_pairs) > max_cells:
        picks = np.linspace(0, len(cell_pairs) - 1, max_cells).round().astype(int)
        cell_pairs = [cell_pairs[i] for i in picks]
        subsampled = True
    drows = [p[0] for p in cell_pairs]
    rrows = [p[1] for p in cell_pairs]
    dcols = [p[0] for p in gene_pairs]
    rcols = [p[1] for p in gene_pairs]
    dblock = derived[drows, :][:, dcols].toarray()
    if orientation == 'csc_genes_by_cells':
        rblock = reference[:, rrows][rcols, :].toarray().T
    else:
        rblock = reference[rrows, :][:, rcols].toarray()
    if dblock.shape != rblock.shape:
        raise ValueError(f'comparison block shapes differ: {dblock.shape} vs {rblock.shape}')
    mismatch = dblock != rblock
    informative = (dblock != 0) | (rblock != 0)
    mismatch_n = int(mismatch.sum())
    informative_n = int(informative.sum())
    derived_hash = sha256_file(derived_path)
    reference_hash = sha256_file(reference_path)
    member_manifest = None
    if v2_result_path is not None:
        v2 = json.loads(v2_result_path.read_text(encoding='utf-8'))
        if (Path(v2['derived']['path']).resolve() != derived_path.resolve()
                or Path(v2['reference']['path']).resolve() != reference_path.resolve()
                or v2['derived']['sha256'] != derived_hash
                or v2['reference']['sha256'] != reference_hash):
            raise ValueError('baseline input paths or hashes differ from the v2 result')
        member_manifest = v2.get('effective_input_selection', {}).get('reference', {}).get('members')
    result = {
        'method': 'stable_gene_ID_and_literal_cell_ID_exact_identity_comparison',
        'baseline_script_sha256': sha256_file(Path(__file__)),
        'derived_path': str(derived_path), 'reference_path': str(reference_path),
        'derived_sha256': derived_hash,
        'reference_spec_sha256': reference_hash,
        'reference_bundle_member_manifest_from_v2': member_manifest,
        'matrix_source': matrix_key, 'gene_key': gene_key, 'cell_key': cell_key,
        'reference_orientation': orientation,
        'matched_gene_ids': len(gene_pairs), 'matched_cell_ids_total': len(dcell_unique.keys() & rcell_unique.keys()),
        'compared_cells': len(cell_pairs), 'cell_subsampled': subsampled,
        'compared_entries': int(dblock.size), 'mismatched_entries': mismatch_n,
        'informative_union_entries': informative_n,
        'raw_mismatch_fraction': mismatch_n / dblock.size if dblock.size else None,
        'informative_mismatch_fraction': mismatch_n / informative_n if informative_n else None,
        'exact_identity_match': bool(mismatch_n == 0 and informative_n > 0),
        'decision': 'exact_match_in_scope' if mismatch_n == 0 and informative_n > 0
        else ('insufficient_information' if informative_n == 0 else 'nonidentity_mismatch'),
        'scope': 'matched stable Ensembl gene IDs and literal cell IDs; selected cells only; identity only; no offset/tolerance/order search',
        'runtime_s': round(time.perf_counter() - started, 3),
        'python': platform.python_version(), 'numpy': np.__version__,
    }
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--derived', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--max-cells', type=int, default=300)
    ap.add_argument('--v2-result', type=Path,
                    help='optional v2 JSON used only to verify the same input file hashes')
    ap.add_argument('--json', type=Path, required=True)
    args = ap.parse_args()
    result = compare(args.derived, args.reference, args.max_cells, args.v2_result)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                         encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('decision', 'matched_gene_ids',
                                             'compared_cells', 'informative_mismatch_fraction',
                                             'runtime_s')}))


if __name__ == '__main__':
    main()
