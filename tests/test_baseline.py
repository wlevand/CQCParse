"""
Baseline of parse_from_source (todo.txt T2).

The output of the current code for every example file is saved in tests/baseline/: one <source>.npz per source
(the arrays) and summary.json (the returned keys with their types and shapes, and the errors).
test_baseline parses each source again and compares with the saved output.

The baseline holds today's bugs too, so a difference is not an error by itself. When a change is meant (a fix),
check each difference, give its reason in the commit message, then rewrite the baseline:
    python tests/test_baseline.py
Only the .npz files whose arrays changed are rewritten.
"""
import contextlib
import io
import json
import os

import numpy as np
import pytest

from CQCParse.parsing import parse_from_source
from CQCParse.utils import PKG_ROOT

EXAMPLES = PKG_ROOT + '/CQCParse/files_examples'
BASELINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'baseline')
SUMMARY = os.path.join(BASELINE_DIR, 'summary.json')

# name: (source_type, path in files_examples, lvl_theory, basis_set)
SOURCES = {
    'g16_CO_HF_STO-3G':        ('gaussian', 'co/g16_inputFull_3q.out', 'HF', 'STO-3G'),
    'g16_ACAC_B3LYP_cc-pVQZ':  ('gaussian', 'dftGaussian/ACAC/B3LYPcc-pVQZ/g16_inputFull_3q.out', 'B3LYP', 'cc-pVQZ'),
    'g16_ACDM_B3LYP_cc-pVQZ':  ('gaussian', 'dftGaussian/ACDM/B3LYPcc_pVQZ/g16_b3lyp_corr_inputFull_3q.out',
                                'B3LYP', 'cc-pVQZ'),
    'g16_FORM_B3LYP_cc-pVDZ':  ('gaussian', 'dftGaussian/FORM/B3LYPcc_pVDZ/g16_inputFull_3q.out', 'B3LYP', 'cc-pVDZ'),
    'g16_FORM_B3LYP_cc-pVQZ':  ('gaussian', 'dftGaussian/FORM/B3LYPcc_pVQZ/g16_inputFull_3q.out', 'B3LYP', 'cc-pVQZ'),
    'g16_FORM_HF_cc-pVQZ':     ('gaussian', 'dftGaussian/FORM/HFcc_pVQZ/g16_inputFull_3q.out', 'HF', 'cc-pVQZ'),
    'g16_FORM_HF_cc-pVTZ':     ('gaussian', 'dftGaussian/FORM/HFcc_pVTZ/g16_inputFull_3q.out', 'HF', 'cc-pVTZ'),
    'c4_FORM_CCSDT_cc-pVQZ':   ('cfour', 'refinedc4/CCSDTcc_pVQZ', 'CCSD(T)', 'cc-pVQZ'),
    'c4_FORM_HF_cc-pVQZ':      ('cfour', 'refinedc4/HFcc_pVQZ', 'HF', 'cc-pVQZ'),
    'c4_FORM_HF_cc-pVTZ':      ('cfour', 'refinedc4/HFcc_pVTZ', 'HF', 'cc-pVTZ'),
    'c4_FORM_HF_cc-pVDZ':      ('cfour', 'refinedc4/HFcc_pVDZ', 'HF', 'cc-pVDZ'),
}

# every name a reader knows (cff_rc and qff_rc come only next to cff and qff for now, todo A1)
ALL_NAMES = ['equilibrium_geometry', 'atoms', 'normal_modes',
             'harmonic_states', 'anharmonic_states', 'nc_sqrt_eigval', 'B', 'coriolis',
             'dipgrad', 'diphess', 'polgrad', 'polhess', 'cff', 'cff_rc', 'qff', 'qff_rc', 'modes_mapping']
# what wilson asks for in every request (MolSystemData.load)
BASE = ['nc_sqrt_eigval', 'harmonic_states', 'anharmonic_states']


def parse(source, names):
    source_type, path, lvl_theory, basis_set = SOURCES[source]
    with contextlib.redirect_stdout(io.StringIO()):   # the readers print a lot
        return parse_from_source(requested_data=dict.fromkeys(names), source_type=source_type,
                                 base_file_loc=EXAMPLES + '/' + path,
                                 lvl_theory=lvl_theory, basis_set=basis_set)


def error_text(exc):
    return f'{type(exc).__name__}: {exc}'.replace(PKG_ROOT, '<PKG_ROOT>')


def collect(source, skip=()):
    """
    Parse every name in ALL_NAMES (except skip) in one request. If that fails, find the names that fail, each
    asked for with BASE (as wilson asks), and parse the rest.
    Returns the result dict and {name: error text}.
    """
    names = [n for n in ALL_NAMES if n not in skip]
    try:
        return parse(source, names), {}
    except Exception:
        pass
    errors = {}
    for n in names:
        try:
            parse(source, BASE + [n])
        except Exception as exc:
            errors[n] = error_text(exc)
    ok = [n for n in names if n not in errors]
    return (parse(source, ok) if ok else {}), errors


def key_text(key):
    return ','.join(str(k) for k in key) if isinstance(key, tuple) else str(key)


def flatten(result):
    """
    The result as plain numpy arrays (what an .npz file holds), and a short description of each key.
    A dict becomes two arrays, sorted by key: '<name>.keys' (as text, e.g. '0,1') and '<name>.values'.
    """
    arrays, described = {}, {}
    for name, value in result.items():
        if isinstance(value, dict):
            items = sorted(value.items(), key=lambda kv: key_text(kv[0]))
            arrays[name + '.keys'] = np.array([key_text(k) for k, _ in items], dtype=str)
            values = arrays[name + '.values'] = np.array([v for _, v in items])
            described[name] = f'dict, {len(items)} entries' + (f' of shape {values.shape[1:]}'
                                                               if values.ndim > 1 else '')
        elif isinstance(value, np.ndarray):
            arrays[name] = value
            described[name] = f'ndarray {value.dtype} {value.shape}'
        elif isinstance(value, list):
            arrays[name] = np.array(value)
            described[name] = f'list, {len(value)} items'
        else:
            arrays[name] = np.array(value)
            described[name] = f'{type(value).__name__} {value!r}'
    return arrays, described


def compare(saved, new, rtol=1e-12, atol=1e-18):
    """The differences between two flattened results, as text lines."""
    diffs = []
    for k in sorted(set(saved) | set(new)):
        if k not in new:
            diffs.append(f'{k}: not returned now')
        elif k not in saved:
            diffs.append(f'{k}: returned now, not before')
        elif saved[k].shape != new[k].shape:
            diffs.append(f'{k}: shape {saved[k].shape} -> {new[k].shape}')
        elif saved[k].dtype.kind in 'fc' and new[k].dtype.kind in 'fc':
            if not np.allclose(saved[k], new[k], rtol=rtol, atol=atol):
                diffs.append(f'{k}: largest difference {np.max(np.abs(saved[k] - new[k])):.3g}')
        elif not np.array_equal(saved[k], new[k]):
            diffs.append(f'{k}: values differ')
    return diffs


def load_saved(source):
    with open(SUMMARY) as file:
        errors = json.load(file)[source]['errors']
    with np.load(os.path.join(BASELINE_DIR, source + '.npz'), allow_pickle=False) as data:
        return dict(data), errors


@pytest.mark.parametrize('source', list(SOURCES))
def test_baseline(source):
    saved, saved_errors = load_saved(source)

    result, errors = collect(source, skip=saved_errors)
    diffs = compare(saved, flatten(result)[0])
    diffs += [f'{n}: fails now ({e})' for n, e in errors.items()]
    for n in saved_errors:
        try:
            parse(source, BASE + [n])
            diffs.append(f'{n}: works now (failed before)')
        except Exception:
            pass

    assert not diffs, f'{source} differs from the baseline:\n    ' + '\n    '.join(diffs)


def write_baseline():
    os.makedirs(BASELINE_DIR, exist_ok=True)
    summary = {}
    for source, (source_type, path, lvl_theory, basis_set) in SOURCES.items():
        result, errors = collect(source)
        arrays, described = flatten(result)
        npz = os.path.join(BASELINE_DIR, source + '.npz')
        changed = True
        if os.path.exists(npz):
            with np.load(npz, allow_pickle=False) as data:
                changed = bool(compare(dict(data), arrays, rtol=0, atol=0))
        if changed:
            np.savez_compressed(npz, **arrays)
        summary[source] = {'source_type': source_type, 'file': path,
                           'lvl_theory': lvl_theory, 'basis_set': basis_set,
                           'returned': described, 'errors': errors}
        print(f'{source}: {"written" if changed else "unchanged"} '
              f'({len(described)} keys returned, {len(errors)} errors)')
    with open(SUMMARY, 'w') as file:
        json.dump(summary, file, indent=2)
        file.write('\n')


if __name__ == '__main__':
    write_baseline()
