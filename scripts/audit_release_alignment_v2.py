"""Count--annotation alignment checker v2 (generic, standalone).

Two layers:
  L1  conventional integrity checks  -- what standard practice already does
      (shape, uniqueness, value domain, per-cell totals, declared checksum).
  L2  alignment fingerprint          -- exact-integer comparison of a derived
      matrix against an authoritative reference under candidate alignments
      (gene-label row offset k, identity, permuted counts), plus a cell-level
      agreement profile.

Design discipline (see docs/data_audit_checker_plan_20261002.md):
  * Only exact integer comparison decides a row-offset verdict. Non-integer
    derived values are reported as inconclusive, never as an error.
  * Genes/cells are matched by identifier, never by position, so a legitimate
    column reorder must come out as aligned.
  * Output describes the alignment relationship between two files only. It does
    not name a responsible party, a conversion step, or an effect on published
    analyses.

Usage:
  python scripts/audit_release_alignment_v2.py check \
      --derived <h5ad|bundle.json> --reference <h5ad|bundle.json> [--json out.json]

  python scripts/audit_release_alignment_v2.py make-bundle \
      --project-descriptor <pilot_v6_rna_matrix_descriptor.json> --json out.json

The versioned status and strict-JSON contract applies to `check` only.
`check` exits: 0 = supported exact match in scope; 1 = structural failure or
supported identity mismatch; 2 = tolerance/support/incompatible/unresolved;
3 = read/parse/runtime failure. Inspect the component states and scope.
`selfstats` is an auxiliary internal-consistency check, not identity validation.
The legacy `benchmark` command is not exposed in this release; historical
experiments remain available through the preserved v1 executable and artifacts.
"""
import argparse
import gzip
import hashlib
import json
import sys
import time
import platform
import warnings
from pathlib import Path

import numpy as np

DERIVED_GENE_KEYS = ['ENSEMBL', 'gene_ids', 'ensembl_id', 'gene_id', 'index', '_index']
DERIVED_CELL_KEYS = ['cell_barcode', 'cell_barcodes', 'barcode', 'barcodes', 'bc_wells',
                     'cell_id', '_index', 'index']
BUNDLE_FORMAT = 'alignment-bundle-v1'
TOOL_VERSION = '2.0.2-rc1'


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


TOOL_SHA256 = sha256_file(__file__)


def stable_id(value):
    """Strip an Ensembl-style version suffix; keep everything else unchanged."""
    text = str(value)
    if text.startswith('ENSG') and '.' in text:
        return text.split('.')[0]
    return text


def _as_path(value, root):
    p = Path(value)
    return p if p.is_absolute() else (root / p)


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #
class Matrix:
    """cells x genes integer counts, with labels."""

    def __init__(self, cells, genes, symbols, csr_data, csr_indices, csr_indptr, shape):
        self.cells = np.asarray(cells, dtype=object)
        self.genes = np.asarray(genes, dtype=object)
        self.symbols = np.asarray(symbols, dtype=object) if symbols is not None else None
        self.data = csr_data
        self.indices = csr_indices
        self.indptr = csr_indptr
        self.shape = tuple(int(x) for x in shape)

    @property
    def n_cells(self):
        return self.shape[0]

    @property
    def n_genes(self):
        return self.shape[1]

    def block(self, cell_positions, gene_positions):
        """Dense (len(cell_positions), len(gene_positions)) submatrix.

        float64 is used unconditionally: integer counts are represented exactly
        below 2**53, while normalised releases hold fractional values that would
        be silently truncated to zero by an integer accumulator.
        """
        out = np.zeros((len(cell_positions), len(gene_positions)), dtype=np.float64)
        col_of = {int(g): j for j, g in enumerate(gene_positions)}
        for i, c in enumerate(cell_positions):
            lo, hi = int(self.indptr[c]), int(self.indptr[c + 1])
            for g, v in zip(self.indices[lo:hi], self.data[lo:hi]):
                j = col_of.get(int(g))
                if j is not None:
                    out[i, j] = v
        return out

    def row_totals(self, cell_positions):
        return np.array([int(self.data[int(self.indptr[c]):int(self.indptr[c + 1])].sum())
                         for c in cell_positions], dtype=np.int64)

    def subset_totals(self, cell_positions, gene_positions):
        """Per-cell sums restricted to the given gene positions.

        Both sides of a comparison must use the same gene subset, otherwise the
        totals compare different quantities (e.g. a 5,838-gene panel against a
        58,347-gene whole matrix).
        """
        wanted = set(int(g) for g in gene_positions)
        out = np.zeros(len(cell_positions), dtype=np.int64)
        for i, c in enumerate(cell_positions):
            lo, hi = int(self.indptr[c]), int(self.indptr[c + 1])
            total = 0
            for g, v in zip(self.indices[lo:hi], self.data[lo:hi]):
                if int(g) in wanted:
                    total += v
            out[i] = total
        return out

    def density(self):
        return len(self.data) / max(1, self.shape[0] * self.shape[1])


def load_bundle(path, root):
    spec = json.loads(Path(path).read_text(encoding='utf-8'))
    if spec.get('format') != BUNDLE_FORMAT:
        raise ValueError(f'unsupported bundle format: {spec.get("format")!r}')
    counts = spec['counts']
    data = np.load(_as_path(counts['data'], root), mmap_mode='r')
    indices = np.load(_as_path(counts['indices'], root), mmap_mode='r')
    indptr = np.load(_as_path(counts['indptr'], root), mmap_mode='r')
    genes = np.load(_as_path(spec['genes'], root), allow_pickle=True)
    cells = np.load(_as_path(spec['cells'], root), allow_pickle=True)
    shape = tuple(int(x) for x in counts['shape'])
    orientation = counts.get('orientation', 'csc')
    # Both orientations are addressed the same way here: indptr is indexed by the
    # cell (CSC column or CSR row) and indices hold gene positions. For CSC the
    # declared shape is (genes, cells), for CSR it is (cells, genes).
    if orientation == 'csc' and shape == (len(genes), len(cells)):
        return Matrix(cells, genes, spec.get('symbols'), data, indices, indptr,
                      (len(cells), len(genes)))
    if orientation == 'csr' and shape == (len(cells), len(genes)):
        return Matrix(cells, genes, spec.get('symbols'), data, indices, indptr,
                      (len(cells), len(genes)))
    raise ValueError(f'bundle shape {shape} does not match orientation {orientation!r} '
                     f'and labels ({len(cells)} cells, {len(genes)} genes)')


def _read_labels(node):
    """Read a string/categorical obs or var column into an object array."""
    if hasattr(node, 'asstr'):
        return np.asarray(node.asstr()[:], dtype=object)
    if 'categories' in node and 'codes' in node:          # categorical encoding
        cats = node['categories'].asstr()[:]
        codes = np.asarray(node['codes'][:])
        return np.array([cats[c] if 0 <= c < len(cats) else '' for c in codes], dtype=object)
    raise ValueError('unsupported label encoding')


def load_h5ad(path, matrix_source='auto', gene_key='auto', cell_key='auto'):
    import h5py
    from scipy.sparse import csc_matrix, csr_matrix
    gene_key_name = gene_key.removeprefix('var/') if gene_key != 'auto' else 'auto'
    cell_key_name = cell_key.removeprefix('obs/') if cell_key != 'auto' else 'auto'
    with h5py.File(path, 'r') as f:
        if matrix_source == 'auto':
            selected_matrix_source = 'layers/counts' if 'layers/counts' in f else 'X'
        else:
            selected_matrix_source = matrix_source
        if selected_matrix_source not in ('X', 'layers/counts'):
            raise ValueError('matrix source must be X, layers/counts, or auto')
        if selected_matrix_source == 'layers/counts' and 'layers/counts' not in f:
            raise ValueError('requested matrix source layers/counts is absent')
        layer = f[selected_matrix_source]
        encoding = str(layer.attrs.get('encoding-type', 'csr_matrix'))
        if isinstance(layer, h5py.Dataset):
            # Dense array encoding ('array'): wrap as CSR.  Memory-heavy, so it is
            # only attempted when the file is not stored sparsely.
            dense = layer[:]
            sparse = csr_matrix(np.asarray(dense))
            data, indices, indptr = sparse.data, sparse.indices, sparse.indptr
            declared = tuple(int(x) for x in sparse.shape)
            del dense
        else:
            shape = tuple(int(x) for x in layer.attrs['shape'])
            data = np.asarray(layer['data'][:])
            indices = np.asarray(layer['indices'][:])
            indptr = np.asarray(layer['indptr'][:])
            # AnnData always declares shape as (n_obs, n_vars) whatever the encoding.
            # CSR therefore indexes cells in indptr (matching our access pattern),
            # while CSC indexes genes in indptr and must be transposed first.
            if 'csc' in encoding:
                sparse = csc_matrix((data, indices, indptr), shape=shape).tocsr()
            else:
                sparse = csr_matrix((data, indices, indptr), shape=shape)
            data, indices, indptr = sparse.data, sparse.indices, sparse.indptr
            declared = shape
        genes = None
        for key in DERIVED_GENE_KEYS:
            if (gene_key_name == 'auto' and f'var/{key}' in f) or gene_key_name == key:
                try:
                    genes = _read_labels(f[f'var/{key}'])
                    selected_gene_key = f'var/{key}'
                    break
                except ValueError:
                    if gene_key != 'auto':
                        raise
                    continue
        if genes is None:
            if gene_key_name not in ('auto', '_index'):
                raise ValueError(f'requested gene key {gene_key!r} is absent or unreadable')
            genes = _read_labels(f['var/_index']) if 'var/_index' in f else None
            selected_gene_key = 'var/_index' if genes is not None else None
        if genes is None:
            raise ValueError('cannot locate gene identifiers in var/')
        symbols = None
        for key in ['rawSYMBOL', 'gene_symbols', 'symbol', 'gene_name', 'gene_symbol']:
            if f'var/{key}' in f:
                try:
                    symbols = _read_labels(f[f'var/{key}'])
                    break
                except ValueError:
                    continue
        cells = None
        for key in DERIVED_CELL_KEYS:
            if (cell_key_name == 'auto' and f'obs/{key}' in f) or cell_key_name == key:
                try:
                    cells = _read_labels(f[f'obs/{key}'])
                    selected_cell_key = f'obs/{key}'
                    break
                except ValueError:
                    if cell_key != 'auto':
                        raise
                    continue
        if cells is None and cell_key != 'auto':
            raise ValueError(f'requested cell key {cell_key!r} is absent or unreadable')
        if cells is None:
            raise ValueError('cannot locate cell identifiers in obs/')
    matrix = Matrix(cells, genes, symbols, data, indices, indptr, declared)
    matrix.input_selection = {'matrix_source': selected_matrix_source,
                              'gene_key': selected_gene_key,
                              'cell_key': selected_cell_key}
    return matrix


def load_matrix(path, root, matrix_source='auto', gene_key='auto', cell_key='auto'):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in ('.h5ad', '.h5'):
        return load_h5ad(path, matrix_source, gene_key, cell_key)
    if (matrix_source, gene_key, cell_key) != ('auto', 'auto', 'auto'):
        raise ValueError('H5AD selection options cannot be used with a bundle input')
    return load_bundle(path, root)


# --------------------------------------------------------------------------- #
# L1 conventional checks
# --------------------------------------------------------------------------- #
def conventional_checks(m, reference=None, matched_cells=None, matched_genes=None,
                        declared_md5=None, path=None):
    values = m.data
    finite = bool(np.isfinite(values).all())
    integer = bool(finite and np.all(values == np.floor(values)))
    raw_genes = list(map(str, m.genes))
    stable_genes = [stable_id(g) for g in raw_genes]
    raw_dupes = len(raw_genes) - len(set(raw_genes))
    stable_dupes = len(stable_genes) - len(set(stable_genes))
    report = {
        'shape_declared': list(m.shape),
        'shape_matches_labels': bool(m.shape == (len(m.cells), len(m.genes))),
        'unique_gene_labels': bool(raw_dupes == 0),
        # Informational only: after version stripping, one stable id released with
        # two versions (45 such ids in the sciPlex annotation vintage) makes two
        # rows share a stable id. Reported, but not part of the pass/fail gate.
        'duplicate_stable_gene_ids': int(stable_dupes),
        'unique_cell_ids': bool(len(set(map(str, m.cells))) == len(m.cells)),
        'values_nonnegative': bool(values.size == 0 or values.min() >= 0),
        'values_finite': finite,
        'values_integer': integer,
        'sparsity': round(float(m.density()), 6),
    }
    if declared_md5 is not None and path is not None:
        report['declared_md5'] = declared_md5
        report['actual_md5'] = hashlib.md5(Path(path).read_bytes()).hexdigest()
        report['checksum_matches'] = bool(report['actual_md5'] == declared_md5.replace('md5:', ''))
    if reference is not None and matched_cells and matched_genes:
        d_tot = m.subset_totals([c[0] for c in matched_cells], [g[0] for g in matched_genes])
        r_tot = reference.subset_totals([c[1] for c in matched_cells], [g[1] for g in matched_genes])
        equal = d_tot == r_tot
        report['per_cell_total_equals_reference'] = bool(equal.all())
        report['per_cell_total_equal_fraction'] = round(float(equal.mean()), 6)
    report['passes_all_applicable'] = bool(
        report['shape_matches_labels'] and report['unique_gene_labels']
        and report['unique_cell_ids']
        and report['values_nonnegative'] and report['values_finite']
        and report.get('checksum_matches', True)
    )
    return report


# --------------------------------------------------------------------------- #
# L2 alignment fingerprint
# --------------------------------------------------------------------------- #
def match_entities(derived, reference):
    """Map derived genes/cells onto reference indices by identifier (by label)."""
    ref_gene_pos = {}
    for i, g in enumerate(map(str, reference.genes)):
        ref_gene_pos.setdefault(stable_id(g), []).append(i)
    ref_gene_unique = {k: v[0] for k, v in ref_gene_pos.items() if len(v) == 1}
    derived_gene_pos = {}
    for j, g in enumerate(map(str, derived.genes)):
        derived_gene_pos.setdefault(stable_id(g), []).append(j)
    gene_pairs = []          # (derived_column, reference_row)
    for key, cols in derived_gene_pos.items():
        if len(cols) == 1 and key in ref_gene_unique:
            gene_pairs.append((cols[0], ref_gene_unique[key]))
    gene_pairs.sort()

    ref_cell_pos = {}
    for i, c in enumerate(map(str, reference.cells)):
        ref_cell_pos.setdefault(c, i)
    cell_pairs = []
    seen_derived = set()
    for j, c in enumerate(map(str, derived.cells)):
        c = str(c)
        if c in ref_cell_pos and c not in seen_derived:
            seen_derived.add(c)
            cell_pairs.append((j, ref_cell_pos[c]))
    cell_pairs.sort()
    return gene_pairs, cell_pairs


def sample(positions, limit):
    if limit is None or len(positions) <= limit:
        return list(positions), False
    idx = np.linspace(0, len(positions) - 1, limit).round().astype(int)
    idx = sorted(set(int(i) for i in idx))
    return [positions[i] for i in idx], True


def reference_dense_rows(reference, rows_needed, cell_positions):
    """Dense (len(rows_needed_in_order_unique), n_cells) for the given reference rows."""
    rows_unique = np.unique(np.asarray(rows_needed, dtype=np.int64))
    row_lookup = {int(r): i for i, r in enumerate(rows_unique)}
    out = np.zeros((len(rows_unique), len(cell_positions)), dtype=np.int64)
    by_row = {}
    for i, c in enumerate(cell_positions):
        lo, hi = int(reference.indptr[c]), int(reference.indptr[c + 1])
        for g, v in zip(reference.indices[lo:hi], reference.data[lo:hi]):
            pos = row_lookup.get(int(g))
            if pos is not None:
                by_row.setdefault(pos, []).append((i, v))
    for pos, entries in by_row.items():
        cols = [e[0] for e in entries]
        vals = [e[1] for e in entries]
        out[pos, cols] = vals
    return out, row_lookup


def positional_cell_pairs(derived, reference, min_rho=0.99):
    """Row-order check for companions that ship no cell identifiers.

    Compares the number of detected genes per cell, which is invariant under
    library-size normalisation and log transforms.  Per-cell total counts are
    NOT usable here: against a normalised derived matrix they correlate weakly
    even when the row order is identical.
    """
    if derived.n_cells != reference.n_cells:
        return None, {'applicable': False,
                      'reason': f'cell counts differ ({derived.n_cells} vs {reference.n_cells})'}
    from scipy.stats import spearmanr
    d_nnz = np.diff(np.asarray(derived.indptr, dtype=np.int64))
    r_nnz = np.diff(np.asarray(reference.indptr, dtype=np.int64))
    rho = float(spearmanr(d_nnz, r_nnz).statistic)
    ok = bool(np.isfinite(rho) and rho >= min_rho)
    return ([(i, i) for i in range(derived.n_cells)] if ok else None), {
        'applicable': True, 'statistic': 'detected genes per cell (normalisation-invariant)',
        'spearman': round(rho, 6), 'threshold': min_rho, 'validated': ok}


def alignment_fingerprint(derived, reference, max_cells=300, offsets=range(-3, 4),
                          mismatch_frac=0.05, min_cells=3, positional_cells=False):
    started = time.time()
    gene_pairs, cell_pairs = match_entities(derived, reference)
    result = {
        'mapped_gene_columns': len(gene_pairs),
        'mapped_cells_total': len(cell_pairs),
        'derived_genes': int(derived.n_genes), 'derived_cells': int(derived.n_cells),
        'reference_genes': int(reference.n_genes), 'reference_cells': int(reference.n_cells),
        'cell_matching': 'identifier',
    }
    if positional_cells and len(cell_pairs) < min_cells:
        pairs, check = positional_cell_pairs(derived, reference)
        result['cell_matching'] = 'positional' if pairs else 'none'
        result['positional_order_check'] = check
        if pairs:
            cell_pairs = pairs
            result['mapped_cells_total'] = len(pairs)
            result['positional_cell_matching'] = (
                'cells matched by row order after a normalisation-invariant order check '
                f"(detected genes per cell, Spearman {check.get('spearman')})")
    if len(gene_pairs) < 50 or len(cell_pairs) < min_cells:
        result.update(verdict='unresolved_no_overlap',
                      reason='insufficient identifier overlap for a fingerprint',
                      runtime_s=round(time.time() - started, 2))
        return result
    cell_pairs, subsampled = sample(cell_pairs, max_cells)
    result['fingerprint_cells'] = len(cell_pairs)
    result['cell_subsampled'] = subsampled
    result['fingerprint_gene_columns'] = len(gene_pairs)
    d_cols = [p[0] for p in gene_pairs]
    ref_rows = np.array([p[1] for p in gene_pairs], dtype=np.int64)
    d_cells = [p[0] for p in cell_pairs]
    r_cells = [p[1] for p in cell_pairs]

    derived_block = derived.block(d_cells, d_cols)
    # No integer cast here: normalised releases hold fractional values, and
    # casting would truncate them to zero and silently invalidate the order
    # statistic. Integral counts are represented exactly in float64.
    # Totals restricted to the mapped gene set on both sides, so that a value-
    # preserving pair (including a reordered one) is comparable.
    d_totals = derived.subset_totals(d_cells, d_cols)
    r_totals = reference.subset_totals(r_cells, ref_rows)
    total_equal = d_totals == r_totals
    rel = np.abs(d_totals - r_totals) / np.maximum(1, r_totals)
    result['per_cell_total_equal_fraction'] = round(float(total_equal.mean()), 6)
    result['per_cell_total_median_relative_difference'] = round(float(np.median(rel)), 6)

    n_ref_genes = reference.n_genes
    offset_report = {}
    mismatch_by_offset = {}
    order_by_offset = {}
    valid_offsets = []
    gathered0 = None
    for k in offsets:
        rows_k = ref_rows + int(k)
        inside = (rows_k >= 0) & (rows_k < n_ref_genes)
        if inside.sum() < 0.9 * len(rows_k):
            offset_report[str(k)] = {'applicable': False, 'reason': 'outside reference gene range'}
            continue
        dense, lookup = reference_dense_rows(reference, rows_k, r_cells)
        gathered = dense[[lookup[int(r)] for r in rows_k], :].T
        if int(k) == 0:
            gathered0 = gathered
        mismatch = int((derived_block != gathered).sum())
        # Sparse matrices compare equal almost everywhere simply because both
        # sides are zero. Restrict the informative comparison to the union of
        # non-zero entries, otherwise a wrong alignment looks almost aligned.
        support = int(((derived_block != 0) | (gathered != 0)).sum())
        # Order statistic: the per-gene mean profile must track the reference at
        # the correct alignment. Ranks survive normalisation, so this is what
        # makes normalised (non-integer) releases checkable at all.
        from scipy.stats import spearmanr
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            order_r_raw = float(spearmanr(derived_block.mean(axis=0), gathered.mean(axis=0)).statistic)
        order_r = order_r_raw if np.isfinite(order_r_raw) else None
        mismatch_by_offset[int(k)] = (mismatch, support, int(derived_block.size))
        valid_offsets.append(int(k))
        if order_r is not None:
            order_by_offset[int(k)] = order_r
        offset_report[str(k)] = {
            'applicable': True,
            'mismatched_entries': mismatch,
            'total_entries': int(derived_block.size),
            'mismatch_fraction': round(mismatch / derived_block.size, 6),
            'non_zero_union_entries': support,
            'informative_mismatch_fraction': round(mismatch / support, 6) if support else None,
            'gene_mean_spearman_vs_reference': round(order_r, 6) if order_r is not None else None,
        }
    result['offset_fingerprint'] = offset_report
    result['informative_definition'] = ('mismatch / (entries non-zero on either side); '
                                        'raw fractions are dominated by sparse zeros')

    values = derived.data
    integer = bool(values.size == 0 or np.all(values == np.floor(values)))
    if not valid_offsets:
        result.update(verdict='unresolved_no_overlap', reason='no applicable offsets',
                      runtime_s=round(time.time() - started, 2))
        return result

    if not integer:
        # Normalised (non-integer) release: the exact-count fingerprint does not
        # apply, but the per-gene mean ORDER must still track the reference.
        identity_r = order_by_offset.get(0)
        if not order_by_offset:
            result.update(verdict='insufficient_information',
                          reason='gene mean profiles are constant; order statistic is undefined',
                          runtime_s=round(time.time() - started, 2))
            return result
        best_ok = max(order_by_offset, key=lambda k: order_by_offset[k])
        best_r = order_by_offset[best_ok]
        result['order_fingerprint'] = {str(k): round(v, 6) for k, v in order_by_offset.items()}
        result['order_identity_spearman'] = round(identity_r, 6) if identity_r is not None else None
        result['order_best_shift'] = int(best_ok)
        result['order_best_spearman'] = round(best_r, 6)
        if identity_r is None:
            verdict, reason = 'unresolved_no_overlap', 'order statistic unavailable'
        elif identity_r >= 0.8 and identity_r >= best_r - 0.05:
            verdict = 'aligned_order'
            reason = (f'normalised values; per-gene mean order matches the reference at '
                      f'identity (Spearman {identity_r:.4f}, best other shift {best_r:.4f})')
        elif best_ok != 0 and best_r >= 0.8 and best_r >= identity_r + 0.05:
            verdict = f'order_offset({best_ok:+d})'
            reason = (f'normalised values; per-gene mean order matches the reference only '
                      f'when labels are shifted by {best_ok:+d} '
                      f'(Spearman {best_r:.4f} vs identity {identity_r:.4f})')
        else:
            verdict = 'inconclusive_order'
            reason = (f'normalised values; order statistic does not separate alignments '
                      f'(identity {identity_r:.4f}, best {best_r:.4f} at {best_ok:+d})')
            if identity_r < 0.5 and best_r < 0.5:
                verdict = 'order_mismatch'
                reason = ('normalised values; per-gene mean order does not track the '
                          'reference at any tested alignment '
                          f'(identity Spearman {identity_r:.4f}, best {best_r:.4f})')
        result.update(verdict=verdict, reason=reason,
                      runtime_s=round(time.time() - started, 2))
        return result

    total = mismatch_by_offset[0][2]
    identity = mismatch_by_offset.get(0, (None, None, total))[0]
    identity_support = mismatch_by_offset.get(0, (None, 0, total))[1]
    best_k = min(valid_offsets, key=lambda k: mismatch_by_offset[k][0])
    best_m = mismatch_by_offset[best_k][0]
    best_support = mismatch_by_offset[best_k][1]
    result['identity_mismatch_fraction'] = round(identity / total, 6)
    result['identity_informative_mismatch_fraction'] = round(identity / identity_support, 6) if identity_support else None
    result['best_offset'] = int(best_k)
    result['best_offset_mismatch_fraction'] = round(best_m / total, 6)
    result['best_offset_informative_mismatch_fraction'] = round(best_m / best_support, 6) if best_support else None

    # Per-cell multiset of non-zero values: preserved by any reordering of the
    # same values, destroyed when the two matrices are not the same data.
    multiset_fraction = None
    cell_level = None
    if gathered0 is not None:
        same_multiset = 0
        for i in range(derived_block.shape[0]):
            a = np.sort(derived_block[i][derived_block[i] != 0])
            b = np.sort(gathered0[i][gathered0[i] != 0])
            same_multiset += int(a.shape == b.shape and np.array_equal(a, b))
        multiset_fraction = round(same_multiset / derived_block.shape[0], 6)
        support0 = (derived_block != 0) | (gathered0 != 0)
        per_cell_mismatch = (derived_block != gathered0).sum(axis=1)
        per_cell_support = support0.sum(axis=1)
        frac = np.where(per_cell_support > 0,
                        per_cell_mismatch / np.maximum(1, per_cell_support), 0.0)
        median_fraction = float(np.median(frac))
        outlier_threshold = max(0.25, 5 * median_fraction)
        outliers = int((frac > outlier_threshold).sum())
        cell_level = {
            'identity_median_informative_mismatch_fraction': round(median_fraction, 6),
            'identity_max_informative_mismatch_fraction': round(float(frac.max()), 6),
            'outlier_cells': outliers,
            'outlier_threshold_fraction': round(float(outlier_threshold), 6),
        }
    result['per_cell_value_multiset_equal_fraction'] = multiset_fraction
    result['cell_level'] = cell_level

    if identity_support == 0:
        result['best_offset'] = None
        result['best_offset_mismatch_fraction'] = None
        result['best_offset_informative_mismatch_fraction'] = None
        result.update(verdict='insufficient_information',
                      reason='no non-zero entries in either matched matrix; identity is not testable',
                      runtime_s=round(time.time() - started, 2))
        return result
    identity_informative = identity / identity_support
    best_informative = best_m / best_support if best_support else 1.0
    outliers = (cell_level or {}).get('outlier_cells', 0)
    offset_explains = (best_k != 0 and identity_informative >= 0.2
                       and best_informative <= 0.05 * identity_informative)
    if identity == 0:
        verdict, reason = 'exact_match_in_scope', 'identity alignment reproduces the reference exactly within the reported scope'
    elif offset_explains:
        verdict = f'gene_label_offset({best_k:+d})'
        reason = (f'counts in derived column j equal the reference counts of the gene at '
                  f'reference row ref_row(label_j) {best_k:+d}; this explains '
                  f'{100 * (1 - best_informative / identity_informative):.1f}% of the identity '
                  f'mismatch ({identity} of {identity_support} non-zero comparisons)')
    elif identity_informative > mismatch_frac and (multiset_fraction or 0) >= 0.5:
        verdict = 'gene_label_mismatch'
        reason = ('non-zero entries do not line up at identity, but each cell keeps the same '
                  'multiset of values; consistent with labels not tracking their counts '
                  '(e.g. counted columns reordered)')
    elif identity_informative > mismatch_frac:
        verdict = 'reference_incompatible'
        reason = ('non-zero entries do not line up and per-cell value multisets differ; '
                  'the two matrices are not the same data, no misalignment inferred')
    elif outliers > 0:
        verdict = 'cell_label_mismatch'
        reason = (f'{outliers} cell(s) disagree with the reference at identity while the rest '
                  f'agree; consistent with swapped or mismatched cell labels')
    elif identity == 0:
        verdict, reason = 'exact_match_in_scope', 'all compared entries match at identity in the reported scope'
    elif identity_informative <= mismatch_frac:
        verdict = 'within_tolerance'
        reason = (f'identity informative mismatch fraction {identity_informative:.4f} '
                  f'within configured tolerance {mismatch_frac}; exact identity was not established')
    else:
        verdict, reason = 'inconclusive', 'comparison did not meet a positive or mismatch decision rule'

    result.update(verdict=verdict, reason=reason, runtime_s=round(time.time() - started, 2))
    return result


def _remap(m, new_indices, new_genes=None, new_cells=None, new_data=None, keep_shape=False):
    genes = new_genes if new_genes is not None else m.genes
    cells = new_cells if new_cells is not None else m.cells
    shape = m.shape if keep_shape else (len(cells), len(genes))
    return Matrix(cells, genes, None,
                  new_data if new_data is not None else m.data,
                  np.asarray(new_indices, dtype=np.int64),
                  m.indptr, shape)


def _subset_cells(m, keep):
    keep = np.asarray(keep)
    positions = np.where(keep)[0]
    data, indices, indptr = [], [], [0]
    for c in positions:
        lo, hi = int(m.indptr[c]), int(m.indptr[c + 1])
        data.extend(m.data[lo:hi])
        indices.extend(m.indices[lo:hi])
        indptr.append(len(data))
    return Matrix(m.cells[positions], m.genes, None, np.asarray(data, np.int64),
                  np.asarray(indices, np.int64), np.asarray(indptr, np.int64),
                  (len(positions), len(m.genes)))


def _subset_genes(m, keep):
    keep = np.asarray(keep)
    positions = np.where(keep)[0]
    remap = -np.ones(len(m.genes), dtype=np.int64)
    remap[positions] = np.arange(len(positions))
    data, indices, indptr = [], [], [0]
    for c in range(m.n_cells):
        lo, hi = int(m.indptr[c]), int(m.indptr[c + 1])
        cols = m.indices[lo:hi]
        vals = m.data[lo:hi]
        sel = remap[cols] >= 0
        data.extend(vals[sel])
        indices.extend(remap[cols][sel])
        indptr.append(len(data))
    return Matrix(m.cells, m.genes[positions], None, np.asarray(data, np.int64),
                  np.asarray(indices, np.int64), np.asarray(indptr, np.int64),
                  (m.n_cells, len(positions)))


def _permute_cells(m, order):
    """Rebuild the matrix with cell rows in the given order (labels follow)."""
    order = np.asarray(order)
    data, indices, indptr = [], [], [0]
    for c in order:
        lo, hi = int(m.indptr[c]), int(m.indptr[c + 1])
        data.extend(m.data[lo:hi])
        indices.extend(m.indices[lo:hi])
        indptr.append(len(data))
    return Matrix(m.cells[order], m.genes, None, np.asarray(data, np.float64),
                  np.asarray(indices, np.int64), np.asarray(indptr, np.int64),
                  (len(order), m.n_genes))


def build_base_matrix(reference, panel_rows, cell_positions):
    """Self-consistent (counts, labels) base taken from the authoritative side."""
    row_of = {int(r): j for j, r in enumerate(panel_rows)}
    data, indices, indptr = [], [], [0]
    for c in cell_positions:
        lo, hi = int(reference.indptr[c]), int(reference.indptr[c + 1])
        for g, v in zip(reference.indices[lo:hi], reference.data[lo:hi]):
            if v == 0:
                continue
            j = row_of.get(int(g))
            if j is not None:
                indices.append(j)
                data.append(int(v))
        indptr.append(len(data))
    return Matrix(reference.cells[cell_positions], reference.genes[panel_rows], None,
                  np.asarray(data, np.int64), np.asarray(indices, np.int64),
                  np.asarray(indptr, np.int64), (len(cell_positions), len(panel_rows)))


def run_case(name, layer, expectation, derived, reference, args, positional=False):
    conventional = conventional_checks(derived)
    fp = alignment_fingerprint(derived, reference, max_cells=args.max_cells,
                               offsets=range(args.offsets[0], args.offsets[1]),
                               mismatch_frac=args.mismatch_frac,
                               positional_cells=positional)
    verdict = fp['verdict']
    by_fingerprint = bool(verdict.startswith('gene_label') or verdict.startswith('order_offset')
                          or verdict in ('cell_label_mismatch', 'order_mismatch')
                          or (fp.get('cell_level') or {}).get('outlier_cells', 0) > 0)
    by_conventional = not conventional['passes_all_applicable']
    detected = bool(by_fingerprint or by_conventional)
    return {
        'case': name, 'layer': layer, 'expected': expectation,
        'verdict': verdict,
        'identity_informative_mismatch_fraction': fp.get('identity_informative_mismatch_fraction'),
        'best_offset': fp.get('best_offset'),
        'best_offset_informative_mismatch_fraction': fp.get('best_offset_informative_mismatch_fraction'),
        'per_cell_value_multiset_equal_fraction': fp.get('per_cell_value_multiset_equal_fraction'),
        'cell_level_outlier_cells': (fp.get('cell_level') or {}).get('outlier_cells'),
        'detected_by_fingerprint': bool(by_fingerprint),
        'detected_by_conventional': bool(by_conventional),
        'detected': detected,
        'conventional_passes_all': conventional['passes_all_applicable'],
        'conventional_miss': bool(conventional['passes_all_applicable'] and by_fingerprint),
        'reason': fp.get('reason'),
        'runtime_s': fp.get('runtime_s'),
    }


def cmd_benchmark(args):
    root = Path(args.root).resolve()
    reference = load_matrix(args.reference, root)
    rng = np.random.default_rng(args.seed)

    if args.panel_from_h5ad:
        panel_source = load_matrix(args.panel_from_h5ad, root)
        wanted = {stable_id(g) for g in map(str, panel_source.genes)}
        panel_rows = np.array([i for i, g in enumerate(map(str, reference.genes))
                               if stable_id(g) in wanted], dtype=np.int64)
    else:
        panel_rows = np.sort(rng.choice(reference.n_genes,
                                        size=min(args.n_genes, reference.n_genes),
                                        replace=False))
    stride = max(1, reference.n_cells // args.n_cells)
    cell_positions = np.arange(0, reference.n_cells, stride)[:args.n_cells]
    base = build_base_matrix(reference, panel_rows, cell_positions)

    results = []
    # G1 base self-check: a self-consistent pair must come out aligned.
    results.append(run_case('base_selfcheck', 'control', 'aligned', base, reference, args))

    n_genes = base.n_genes
    n_cells = base.n_cells
    perm = rng.permutation(n_genes)

    # Fault family A: labels repointed to a neighbouring *reference annotation
    # row*, which is the off-by-one form actually observed in case 1.
    def relabel_ref_rows(mask, shift):
        genes = base.genes.copy()
        for j in np.where(mask)[0]:
            r = int(panel_rows[j]) + int(shift)
            if 0 <= r < reference.n_genes:
                genes[j] = reference.genes[r]
        return _remap(base, base.indices, new_genes=genes)

    # Fault family B: labels shifted within the panel list itself (cyclic, so no
    # duplicate identifiers are introduced). Detectable as an exact reference-row
    # offset only when the panel rows are contiguous.
    def relabel_panel_positions(shift):
        order = np.roll(np.arange(n_genes), -shift)
        return _remap(base, base.indices, new_genes=base.genes[order])

    all_mask = np.ones(n_genes, dtype=bool)
    results.append(run_case('F1_label_repointed_ref_row_plus1', 'fault', 'misalignment',
                            relabel_ref_rows(all_mask, 1), reference, args))
    results.append(run_case('F2_label_repointed_ref_row_minus1', 'fault', 'misalignment',
                            relabel_ref_rows(all_mask, -1), reference, args))
    results.append(run_case('F3_label_repointed_ref_row_plus2', 'fault', 'misalignment',
                            relabel_ref_rows(all_mask, 2), reference, args))
    results.append(run_case('F1b_label_shifted_panel_position', 'fault', 'misalignment',
                            relabel_panel_positions(1), reference, args))

    moved = _remap(base, perm[base.indices])                      # counts move, labels stay
    results.append(run_case('F4_count_columns_permuted', 'fault', 'misalignment',
                            moved, reference, args))

    swapped_cells = base.cells.copy()
    if n_cells >= 4:
        i, j = 1, n_cells - 2
        swapped_cells[i], swapped_cells[j] = swapped_cells[j], swapped_cells[i]
    results.append(run_case('F5_barcode_swap', 'fault', 'misalignment',
                            _remap(base, base.indices, new_cells=swapped_cells), reference, args))

    dup_genes = base.genes.copy()
    if n_genes > 12:
        dup_genes[10] = dup_genes[11]
    results.append(run_case('F6_duplicate_gene_id', 'fault', 'misalignment',
                            _remap(base, base.indices, new_genes=dup_genes), reference, args))

    results.append(run_case('F7_gene_label_row_dropped', 'fault', 'misalignment',
                            _remap(base, base.indices, new_genes=base.genes[:-1],
                                   keep_shape=True), reference, args))

    partial = rng.random(n_genes) < 0.30                           # documented boundary case
    results.append(run_case('F8_partial_30pct_ref_row_shift', 'fault_boundary',
                            'not_detected_expected',
                            relabel_ref_rows(partial, 1), reference, args))

    keep_cells = rng.random(n_cells) < 0.70
    results.append(run_case('C1_legitimate_cell_subset', 'control', 'not_flagged',
                            _subset_cells(base, keep_cells), reference, args))
    keep_genes = rng.random(n_genes) < 0.70
    results.append(run_case('C2_legitimate_gene_subset', 'control', 'not_flagged',
                            _subset_genes(base, keep_genes), reference, args))

    totals = np.array([int(base.data[int(base.indptr[c]):int(base.indptr[c + 1])].sum())
                       for c in range(n_cells)], dtype=np.float64)
    cpm = np.zeros_like(base.data, dtype=np.float64)
    pos = 0
    for c in range(n_cells):
        lo, hi = int(base.indptr[c]), int(base.indptr[c + 1])
        cpm[lo:hi] = base.data[lo:hi] / max(1.0, totals[c]) * 1e6
    results.append(run_case('C3_cpm_normalised', 'control', 'not_flagged',
                            _remap(base, base.indices, new_data=cpm), reference, args))

    # Normalised faults: the exact-count fingerprint cannot run, but the order
    # fingerprint must still catch a label shift or a counted-column reorder.
    ref_row_plus1 = np.array([reference.genes[min(reference.n_genes - 1,
                                                int(panel_rows[j]) + 1)] for j in range(n_genes)],
                             dtype=object)
    results.append(run_case('F9_cpm_label_repointed_ref_row_plus1', 'fault', 'misalignment',
                            _remap(base, base.indices, new_genes=ref_row_plus1, new_data=cpm),
                            reference, args))
    results.append(run_case('F10_cpm_count_columns_permuted', 'fault', 'misalignment',
                            _remap(base, perm[base.indices], new_data=cpm), reference, args))

    # Small-magnitude normalisation (log1p of CPM/1e4) reproduces real released
    # files, where most values are below 1.  A cell subset that only sees large
    # normalised values would not exercise this regime.
    lognorm = np.zeros_like(base.data, dtype=np.float64)
    pos = 0
    for c in range(n_cells):
        lo, hi = int(base.indptr[c]), int(base.indptr[c + 1])
        lognorm[lo:hi] = np.log1p(base.data[lo:hi] / max(1.0, totals[c]) * 1e4)
    results.append(run_case('C3b_log_normalised_small_values', 'control', 'not_flagged',
                            _remap(base, base.indices, new_data=lognorm), reference, args))
    results.append(run_case('F11_log_normalised_label_repointed_ref_row_plus1', 'fault',
                            'misalignment',
                            _remap(base, base.indices, new_genes=ref_row_plus1, new_data=lognorm),
                            reference, args))

    results.append(run_case('C4_reorder_labels_follow_counts', 'control', 'not_flagged',
                            _remap(base, perm[base.indices], new_genes=base.genes[np.argsort(perm)]),
                            reference, args))

    results.append(run_case('C5_barcode_prefix', 'control', 'not_flagged',
                            _remap(base, base.indices,
                                   new_cells=np.array(['S1_' + str(c) for c in base.cells],
                                                      dtype=object)), reference, args))

    cell_tot = np.array([int(base.data[int(base.indptr[c]):int(base.indptr[c + 1])].sum())
                         for c in range(n_cells)])
    keep = cell_tot > 0
    results.append(run_case('C6_drop_zero_count_cells', 'control', 'not_flagged',
                            _subset_cells(base, keep), reference, args))

    # Companions without cell identifiers: row-order matching must work when the
    # order really corresponds, and must be refused (not silently accepted) when
    # it does not.  Reference here is the same cells in the same order, which is
    # what a barcode-less companion of the same sample looks like.
    ref_mask = np.zeros(reference.n_cells, dtype=bool)
    ref_mask[cell_positions] = True
    ref_sub = _subset_cells(reference, ref_mask)
    renamed = np.array([f'D{i}' for i in range(n_cells)], dtype=object)
    results.append(run_case('C7_positional_order_matches', 'control', 'not_flagged',
                            _remap(base, base.indices, new_cells=renamed), ref_sub, args,
                            positional=True))
    shuffled = rng.permutation(n_cells)
    results.append(run_case('F12_positional_order_shuffled', 'fault_boundary',
                            'not_detected_expected',
                            _permute_cells(_remap(base, base.indices, new_cells=renamed), shuffled),
                            ref_sub, args, positional=True))

    faults = [r for r in results if r['layer'] == 'fault']
    boundary = [r for r in results if r['layer'] == 'fault_boundary']
    controls = [r for r in results if r['layer'] == 'control' and r['case'] != 'base_selfcheck']
    summary = {
        'base_selfcheck_aligned': results[0]['verdict'] == 'aligned',
        'faults_total': len(faults),
        'faults_fingerprint_recall': round(sum(r['detected_by_fingerprint'] for r in faults) / len(faults), 4),
        'faults_any_layer_recall': round(sum(r['detected'] for r in faults) / len(faults), 4),
        'conventional_miss_count': sum(r['conventional_miss'] for r in faults),
        'conventional_miss_cases': [r['case'] for r in faults if r['conventional_miss']],
        'faults_detected_by_fingerprint': [r['case'] for r in faults if r['detected_by_fingerprint']],
        'faults_detected_by_conventional_only': [r['case'] for r in faults
                                                 if r['detected_by_conventional']
                                                 and not r['detected_by_fingerprint']],
        'faults_undetected': [r['case'] for r in faults if not r['detected']],
        'controls_total': len(controls),
        'control_false_positives': sum(r['detected'] for r in controls),
        'control_false_positive_cases': [r['case'] for r in controls if r['detected']],
        'controls_not_flagged_but_undecidable': [r['case'] for r in controls
                                                 if not r['detected']
                                                 and r['verdict'] in ('inconclusive_non_integer',
                                                                      'inconclusive_order',
                                                                      'unresolved_no_overlap',
                                                                      'undetermined')],
        'boundary_cases': [{'case': r['case'], 'verdict': r['verdict'],
                            'detected': r['detected'],
                            'detected_by_fingerprint': r['detected_by_fingerprint'],
                            'detected_by_conventional': r['detected_by_conventional']}
                           for r in boundary],
    }
    payload = {
        'tool': 'audit_release_alignment', 'mode': 'benchmark', 'seed': int(args.seed),
        'reference': {'path': str(args.reference), 'sha256': sha256_file(args.reference)},
        'base': {'cells': int(n_cells), 'genes': int(n_genes),
                 'panel_source': args.panel_from_h5ad or 'random gene sample',
                 'construction': 'authoritative counts with authoritative labels'},
        'summary': summary, 'cases': results,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text, encoding='utf-8')
    if args.out_md:
        Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_md).write_text(render_benchmark_md(payload), encoding='utf-8')
    print(text)
    return 0 if summary['control_false_positives'] == 0 else 1


def render_benchmark_md(payload):
    s = payload['summary']
    lines = [
        '# 计数—注释对齐检查器：已知真值基准结果',
        '',
        f"生成时间基准种子：{payload['seed']}；基座：{payload['base']['cells']} 细胞 × "
        f"{payload['base']['genes']} 基因，取自权威侧（计数与标签自洽）。",
        '',
        '| 用例 | 层 | 判定 | 信息性失配率(identity) | 最优偏移 | 最优偏移失配率 | 常规检查通过 | 指纹检出 | 常规漏检 |',
        '|---|---|---|---:|---:|---:|---|---|---|',
    ]
    for r in payload['cases']:
        ident = r['identity_informative_mismatch_fraction']
        best = r['best_offset_informative_mismatch_fraction']
        lines.append('| {} | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            r['case'], r['layer'], r['verdict'],
            'n/a' if ident is None else f'{ident:.3f}',
            r['best_offset'] if r['best_offset'] is not None else 'n/a',
            'n/a' if best is None else f'{best:.3f}',
            r['conventional_passes_all'], r['detected_by_fingerprint'], r['conventional_miss']))
    lines += [
        '',
        '## 汇总',
        '',
        f"- 基座自检（自洽对应判为 aligned）：{s['base_selfcheck_aligned']}",
        f"- 故障用例指纹召回：{s['faults_fingerprint_recall']}（{s['faults_total']} 例）",
        f"- 任一层召回：{s['faults_any_layer_recall']}",
        f"- **常规检查漏检数：{s['conventional_miss_count']}**，涉及 {', '.join(s['conventional_miss_cases']) or '无'}",
        f"- 仅由常规层检出：{', '.join(s['faults_detected_by_conventional_only']) or '无'}",
        f"- 两层均未检出：{', '.join(s['faults_undetected']) or '无'}",
        f"- 负对照误报：{s['control_false_positives']} / {s['controls_total']}"
        f"（{', '.join(s['control_false_positive_cases']) or '无'}）",
        f"- 负对照中判为不可判定（非误报）：{', '.join(s['controls_not_flagged_but_undecidable']) or '无'}",
        '',
        '## 边界用例（如实报告，不计入主召回）',
        '',
    ]
    for b in s['boundary_cases']:
        lines.append(f"- {b['case']}：判定 {b['verdict']}；指纹层 = {b['detected_by_fingerprint']}，"
                     f"常规层 = {b['detected_by_conventional']}")
        if b['detected_by_conventional'] and not b['detected_by_fingerprint']:
            lines.append('  - 说明：部分重指向必然产生重复的基因标识符，因此由常规层的唯一性检查检出；'
                         '指纹层未检出，属于工具的真实边界。')
    lines += [
        '',
        '## 口径',
        '',
        '- **偏移 k 的含义**：衍生矩阵第 j 列的计数等于「其标签所指基因在参考中的行号 + k」那一行的计数。',
        '  因此 k = 0 表示标签与计数一致；k = +1 表示计数实际来自标签的下一行。',
        '- 信息性失配率 = 失配条目数 / 任一侧非零的条目数；稀疏矩阵下原始失配率被零值主导，会低估错位。',
        '- 偏移判定要求该偏移解释掉 identity 失配的 ≥95%，且 identity 失配本身 ≥0.2，避免用噪声拟合偏移。',
        '- 负对照中 C3（CPM 归一化）预期判为 `inconclusive_non_integer`，C5（条码加前缀）预期判为',
        '  `unresolved_no_overlap`，两者均**不计为误报**，也不计为检出。',
        '- 本基准只检验检测能力与误报控制，不声称任何来源文件存在错误。',
    ]
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_check(args):
    root = Path(args.root).resolve()
    def load_for_check(path):
        if Path(path).suffix.lower() in ('.h5ad', '.h5'):
            return load_matrix(path, root, args.matrix_source, args.gene_key, args.cell_key)
        return load_matrix(path, root)
    try:
        derived = load_for_check(args.derived)
        reference = load_for_check(args.reference) if args.reference else None
    except Exception as exc:
        structural_input_error = (isinstance(exc, ValueError)
                                  and ('bundle shape' in str(exc)
                                       or 'does not match orientation' in str(exc)))
        if structural_input_error:
            payload = {'tool': 'audit_release_alignment_v2', 'version': TOOL_VERSION,
                       'tool_sha256': TOOL_SHA256, 'mode': 'check',
                       'runtime_environment': {'python': platform.python_version(),
                                               'platform': platform.platform(), 'numpy': np.__version__},
                       'effective_parameters': {'max_cells': args.max_cells,
                                                'matrix_source': args.matrix_source,
                                                'gene_key': args.gene_key, 'cell_key': args.cell_key,
                                                'mismatch_fraction_tolerance': args.mismatch_frac},
                       'derived': {'path': str(args.derived)},
                       'reference': {'path': str(args.reference)} if args.reference else None,
                       'conventional_checks': {}, 'verdict': 'structural_failure',
                       'reason': f'{type(exc).__name__}: {exc}',
                       'structural_status': 'invalid', 'identity_evidence': 'not_evaluated',
                       'overall_status': 'structural_failure', 'scope': None}
            print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
            return 1
        payload = {'tool': 'audit_release_alignment_v2', 'version': TOOL_VERSION,
                   'tool_sha256': TOOL_SHA256, 'mode': 'check',
                   'effective_parameters': {'max_cells': args.max_cells,
                                             'matrix_source': args.matrix_source,
                                             'gene_key': args.gene_key, 'cell_key': args.cell_key,
                                             'mismatch_fraction_tolerance': args.mismatch_frac},
                   'runtime_environment': {'python': platform.python_version(),
                                           'platform': platform.platform(), 'numpy': np.__version__},
                   'derived': {'path': str(args.derived)}, 'reference': {'path': str(args.reference)}
                   if args.reference else None,
                   'conventional_checks': {}, 'verdict': 'error',
                   'structural_status': 'invalid_or_unreadable',
                   'identity_evidence': 'not_evaluated', 'overall_status': 'error',
                   'reason': f'{type(exc).__name__}: {exc}', 'scope': None}
        print(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False))
        return 3
    gene_pairs, cell_pairs = (match_entities(derived, reference) if reference else ([], []))
    conventional = conventional_checks(derived, reference, cell_pairs, gene_pairs,
                                       args.declared_md5, args.derived)
    payload = {
        'tool': 'audit_release_alignment_v2', 'version': TOOL_VERSION,
        'tool_sha256': TOOL_SHA256, 'mode': 'check',
        'runtime_environment': {'python': platform.python_version(), 'platform': platform.platform(),
                                'numpy': np.__version__},
        'effective_parameters': {'max_cells': args.max_cells,
                                 'offsets_inclusive': [args.offsets[0], args.offsets[1] - 1],
                                 'mismatch_fraction_tolerance': args.mismatch_frac,
                                 'matrix_source': args.matrix_source,
                                 'gene_key': args.gene_key, 'cell_key': args.cell_key,
                                 'min_matched_genes': 50, 'min_matched_cells': 3,
                                 'cell_subsample_method': 'evenly_spaced_positions'},
        'derived': {'path': str(args.derived), 'sha256': sha256_file(args.derived)},
        'reference': ({'path': str(args.reference), 'sha256': sha256_file(args.reference)}
                      if args.reference else None),
        'conventional_checks': conventional,
        'effective_input_selection': {
            'derived': describe_input_selection(args.derived, root,
                                                getattr(derived, 'input_selection', None)),
            'reference': (describe_input_selection(args.reference, root,
                                                   getattr(reference, 'input_selection', None))
                          if args.reference else None),
        },
    }
    if reference is None:
        payload.update(verdict='unresolved_no_overlap', reason='no reference supplied')
    elif not (conventional['shape_matches_labels'] and conventional['values_finite']
              and conventional['values_nonnegative']):
        payload.update(verdict='not_evaluated_invalid_structure',
                       reason='identity fingerprint skipped because shape or numeric-domain checks failed',
                       fingerprint={'verdict': 'not_evaluated_invalid_structure',
                                    'reason': 'unsafe matrix structure or values'})
    else:
        payload['fingerprint'] = alignment_fingerprint(
            derived, reference, max_cells=args.max_cells,
            offsets=range(args.offsets[0], args.offsets[1]),
            mismatch_frac=args.mismatch_frac,
            positional_cells=args.positional_cells)
        payload['verdict'] = payload['fingerprint']['verdict']
        payload['reason'] = payload['fingerprint'].get('reason')
    fp_verdict = payload.get('fingerprint', {}).get('verdict', 'unresolved_no_overlap')
    structural_ok = conventional['passes_all_applicable']
    if fp_verdict == 'not_evaluated_invalid_structure':
        identity_status = 'not_evaluated'
    elif fp_verdict in ('aligned', 'exact_match_in_scope'):
        identity_status = 'exact_match_in_scope'
    elif fp_verdict == 'within_tolerance':
        identity_status = 'within_tolerance'
    elif fp_verdict == 'insufficient_information':
        identity_status = 'insufficient_information'
    elif fp_verdict == 'reference_incompatible':
        identity_status = 'reference_incompatible'
    elif fp_verdict.startswith(('gene_label', 'order_offset')) or fp_verdict in ('order_mismatch', 'cell_label_mismatch'):
        identity_status = 'mismatch'
    else:
        identity_status = 'unresolved'
    if not structural_ok:
        overall, exit_code = 'structural_failure', 1
    elif identity_status == 'exact_match_in_scope':
        overall, exit_code = 'exact_match_in_scope', 0
    elif identity_status == 'mismatch':
        overall, exit_code = 'identity_mismatch', 1
    elif identity_status == 'reference_incompatible':
        overall, exit_code = 'reference_incompatible', 2
    else:
        overall, exit_code = identity_status, 2
    payload.update(structural_status='valid' if structural_ok else 'invalid',
                   identity_evidence=identity_status, overall_status=overall,
                   scope={'matched_gene_columns': payload.get('fingerprint', {}).get('fingerprint_gene_columns', 0),
                          'matched_cells_compared': payload.get('fingerprint', {}).get('fingerprint_cells', 0),
                          'max_cells': args.max_cells})
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text, encoding='utf-8')
    print(text)
    return exit_code


def describe_input_selection(path, root, selection=None):
    """Record which matrix and label sources were actually selected."""
    path = Path(path)
    record = {'container': path.suffix.lower(), 'matrix_source': None,
              'gene_key': None, 'cell_key': None}
    if path.suffix.lower() in ('.h5ad', '.h5'):
        record.update(selection or {})
    elif path.suffix.lower() == '.json':
        spec = json.loads(path.read_text(encoding='utf-8'))
        names = {'data': spec['counts']['data'], 'indices': spec['counts']['indices'],
                 'indptr': spec['counts']['indptr'], 'genes': spec['genes'], 'cells': spec['cells']}
        members = {}
        for name, raw in names.items():
            member = _as_path(raw, root)
            members[name] = {'path': str(member), 'bytes': member.stat().st_size,
                             'sha256': sha256_file(member)}
            try:
                arr = np.load(member, mmap_mode='r', allow_pickle=name in ('genes', 'cells'))
            except ValueError as exc:
                if 'memory-mapped' not in str(exc):
                    raise
                arr = np.load(member, allow_pickle=name in ('genes', 'cells'))
            members[name].update(dtype=str(arr.dtype), shape=list(arr.shape))
        record.update(matrix_source='bundle', orientation=spec['counts'].get('orientation', 'csc'),
                      members=members, spec_sha256=sha256_file(path))
    return record


def selfstats_payload(path, root, max_shift=3):
    """Reference-free screening: do stored per-gene / per-cell statistics still
    describe the stored counts?  A file whose labels were shifted after its
    statistics were computed fails at identity and matches only at a shift.

    Signals used (no external file needed):
      * var per-gene statistics recomputed from the matrix
      * obs per-cell statistics recomputed from the matrix
      * share of counts in genes whose stored label starts with MT-
    """
    path = Path(path)
    if path.suffix.lower() in ('.h5ad', '.h5'):
        import h5py
        with h5py.File(path, 'r') as f:
            var_fields = {k: f['var'][k][:] for k in f['var']
                          if isinstance(f['var'][k], h5py.Dataset)}
            obs_fields = {k: f['obs'][k][:] for k in f['obs']
                          if isinstance(f['obs'][k], h5py.Dataset)}
            symbols = None
            for key in ('rawSYMBOL', 'gene_symbol', 'gene_symbols', 'gene_name', 'gene',
                        'genes', 'symbol', 'symbols'):
                if f'var/{key}' in f:
                    try:
                        symbols = _read_labels(f[f'var/{key}'])
                        break
                    except ValueError:
                        continue
            if symbols is None and 'var/_index' in f:
                try:
                    symbols = _read_labels(f['var/_index'])
                except ValueError:
                    symbols = None
    else:
        raise ValueError('selfstats currently supports h5ad/HDF5 input only')
    m = load_matrix(path, Path(root))
    n_cells, n_genes = m.n_cells, m.n_genes
    from scipy.sparse import csr_matrix
    x = csr_matrix((np.asarray(m.data), np.asarray(m.indices), np.asarray(m.indptr)),
                   shape=(n_cells, n_genes))
    gene_total = np.asarray(x.sum(axis=0)).ravel()
    gene_ncell = np.asarray((x != 0).sum(axis=0)).ravel().astype(np.int64)
    cell_total = np.asarray(x.sum(axis=1)).ravel()
    cell_ngene = np.asarray((x != 0).sum(axis=1)).ravel().astype(np.int64)

    def align(stored, recomputed, name, kind):
        stored = np.asarray(stored, dtype=float)
        if stored.shape[0] != recomputed.shape[0]:
            return {'field': name, 'kind': kind, 'verdict': 'length_mismatch',
                    'stored_len': int(stored.shape[0]), 'recomputed_len': int(recomputed.shape[0])}
        n = stored.shape[0]
        exact = {}
        for k in range(-max_shift, max_shift + 1):
            idx = np.arange(n) + k
            ok = (idx >= 0) & (idx < n)
            exact[k] = float(np.mean(stored[ok] == recomputed[idx[ok]]))
        best_k = max(exact, key=lambda k: exact[k])
        identity = exact[0]
        ratio = np.median(stored / np.maximum(1e-9, recomputed))
        ratio_cv = float(np.std(stored / np.maximum(1e-9, recomputed)) /
                         max(1e-9, abs(ratio))) if ratio else float('inf')
        if identity >= 0.99:
            verdict = 'consistent'
        elif exact[best_k] >= 0.99 and best_k != 0:
            verdict = f'self_inconsistent_offset({best_k:+d})'
        else:
            # A field may be stored on a different scale (e.g. means of
            # normalised values rather than of counts). If the ranks still agree
            # it is a scale convention, not an inconsistency.
            from scipy.stats import spearmanr
            rho = float(spearmanr(stored, recomputed).statistic)
            if rho >= 0.95:
                verdict = 'consistent_different_scale'
                return {'field': name, 'kind': kind, 'verdict': verdict,
                        'identity_exact_fraction': round(identity, 6),
                        'best_shift': int(best_k),
                        'best_shift_exact_fraction': round(exact[best_k], 6),
                        'rank_spearman_vs_recomputed': round(rho, 6),
                        'scale_ratio_median': round(float(ratio), 6),
                        'scale_ratio_cv': round(ratio_cv, 6)}
            if ratio_cv < 0.25:
                verdict = 'undetermined_stats_from_other_scope'
            else:
                verdict = 'unexplained'
        return {'field': name, 'kind': kind, 'verdict': verdict,
                'identity_exact_fraction': round(identity, 6),
                'best_shift': int(best_k),
                'best_shift_exact_fraction': round(exact[best_k], 6),
                'scale_ratio_median': round(float(ratio), 6),
                'scale_ratio_cv': round(ratio_cv, 6)}

    results = []
    for name, target, kind in [('n_cells', gene_ncell, 'per_gene'),
                               ('ncells', gene_ncell, 'per_gene'),
                               ('n_cells_by_counts', gene_ncell, 'per_gene'),
                               ('total_counts', gene_total, 'per_gene'),
                               ('ncounts', gene_total, 'per_gene'),
                               ('mean_counts', gene_total / max(1, n_cells), 'per_gene'),
                               ('means', gene_total / max(1, n_cells), 'per_gene')]:
        if name in var_fields and np.asarray(var_fields[name]).dtype.kind in 'fiu':
            results.append(align(var_fields[name], target, f'var/{name}', kind))
    for name, target in [('ncounts', cell_total), ('total_counts', cell_total),
                         ('n_genes', cell_ngene), ('n_genes_by_counts', cell_ngene),
                         ('ngenes', cell_ngene)]:
        if name in obs_fields and np.asarray(obs_fields[name]).dtype.kind in 'fiu':
            results.append(align(obs_fields[name], target, f'obs/{name}', 'per_cell'))

    mt = {'symbols_available': symbols is not None}
    if symbols is not None:
        is_mt = np.array([str(s).upper().startswith('MT-') for s in symbols])
        mt_counts = np.asarray(x[:, is_mt].sum(axis=1)).ravel()
        share = 100 * mt_counts / np.maximum(1, cell_total)
        median_share = float(np.median(share))
        mt.update({'mt_labelled_genes': int(is_mt.sum()),
                   'recomputed_mt_share_median_pct': round(median_share, 4)})
        for field in ('percent_mito', 'pct_counts_mt'):
            if field in obs_fields:
                stored = np.asarray(obs_fields[field], dtype=float)
                mt[f'stored_obs_{field}_median'] = round(float(np.median(stored)), 4)
                mt[f'stored_obs_{field}_matches_recomputed'] = bool(
                    stored.shape[0] == n_cells and
                    np.allclose(stored, share, rtol=1e-3, atol=1e-6))
        # A vocabulary that simply contains no mitochondrial genes (common for
        # mouse matrices or filtered references) is NOT evidence of a fault, so
        # it must be reported as not applicable rather than implausible.
        if is_mt.sum() == 0:
            mt['verdict'] = 'not_applicable_no_mt_labels'
            mt['note'] = ('no MT-labelled gene in this vocabulary; mitochondrial sanity '
                          'check cannot be evaluated for this file')
        elif median_share > 25:
            mt['verdict'] = 'implausible_high_share'
        elif is_mt.sum() >= 5 and median_share < 0.5:
            # Present but nearly empty.  Seen in real MT-poor data (e.g. brain
            # tissue with a mitochondrial-depleted protocol, GSE289659), where
            # the companion matrix lacks MT genes too.  Informational, not a flag.
            mt['verdict'] = 'low_share_informational'
            mt['note'] = ('MT-labelled genes carry almost no counts; this is a property of '
                          'MT-poor or MT-depleted data, not by itself an alignment signal')
        else:
            mt['verdict'] = 'plausible'
        mt['implausible_for_labelled_set'] = mt['verdict'].startswith('implausible')
    else:
        mt['verdict'] = 'not_applicable_no_symbols'
        mt['note'] = 'no gene-symbol column found; mitochondrial sanity check not applicable'
        mt['implausible_for_labelled_set'] = False

    verdicts = {r['verdict'] for r in results}
    if any(v.startswith('self_inconsistent') for v in verdicts):
        overall = 'self_inconsistent'
    elif results and all(v.startswith('consistent') for v in verdicts):
        overall = 'self_consistent'
    elif 'undetermined_stats_from_other_scope' in verdicts or not results:
        overall = 'undetermined'
    else:
        overall = 'unexplained'
    return {'tool': 'audit_release_alignment', 'mode': 'selfstats',
            'matrix': str(path), 'sha256': sha256_file(path),
            'shape': [int(n_cells), int(n_genes)], 'nnz': int(len(m.data)),
            'field_results': results, 'mitochondrial': mt,
            'verdict': overall,
            'interpretation': ('self_consistent: stored statistics describe the stored '
                               'counts under identity. self_inconsistent: they match only '
                               'under a shift. undetermined: statistics were computed on a '
                               'different scope (parent object) and cannot be used.')}


def cmd_selfstats(args):
    root = Path(args.root).resolve()
    payload = selfstats_payload(args.matrix, root, args.max_shift)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text, encoding='utf-8')
    print(text)
    overall = payload['verdict']
    if overall == 'self_inconsistent':
        return 1
    if overall in ('consistent', 'self_consistent'):
        return 0
    return 2


def cmd_make_bundle(args):
    root = Path(args.root).resolve()
    descriptor = json.loads(_as_path(args.project_descriptor, root).read_text(encoding='utf-8'))
    arrays = descriptor['array_files']
    bundle = {
        'format': BUNDLE_FORMAT,
        'provenance': {'note': 'built from a project RDS descriptor', 'descriptor': str(args.project_descriptor)},
        'counts': {
            'data': str(_as_path(arrays['x'], root)),
            'indices': str(_as_path(arrays['i'], root)),
            'indptr': str(_as_path(arrays['p'], root)),
            'shape': [int(v) for v in descriptor['shape']],
            'orientation': 'csc',
        },
        'genes': str(_as_path(descriptor['genes_file'], root)),
        'cells': str(_as_path(descriptor['cells_file'], root)),
    }
    text = json.dumps(bundle, indent=2, ensure_ascii=False)
    Path(args.json).write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    sub = parser.add_subparsers(dest='command', required=True)

    p = sub.add_parser('check')
    p.add_argument('--derived', required=True)
    p.add_argument('--reference')
    p.add_argument('--json')
    p.add_argument('--max-cells', type=int, default=300)
    p.add_argument('--matrix-source', choices=['auto', 'X', 'layers/counts'], default='auto',
                   help='H5AD matrix selection; auto prefers layers/counts then X')
    p.add_argument('--gene-key', default='auto', help='H5AD var key name or auto')
    p.add_argument('--cell-key', default='auto', help='H5AD obs key name or auto')
    p.add_argument('--offsets', type=int, nargs=2, default=[-3, 4])
    p.add_argument('--mismatch-frac', type=float, default=0.05)
    p.add_argument('--positional-cells', action='store_true',
                   help='if the companion has no cell identifiers, fall back to row order '
                        'after a normalisation-invariant order check (detected genes per cell)')
    p.add_argument('--declared-md5')
    p.set_defaults(func=cmd_check)

    p = sub.add_parser('make-bundle')
    p.add_argument('--project-descriptor', required=True)
    p.add_argument('--json', required=True)
    p.set_defaults(func=cmd_make_bundle)

    p = sub.add_parser('selfstats')
    p.add_argument('--matrix', required=True)
    p.add_argument('--max-shift', type=int, default=3)
    p.add_argument('--json')
    p.set_defaults(func=cmd_selfstats)

    args = parser.parse_args()
    sys.exit(args.func(args))
