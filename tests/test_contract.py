"""
Contract test (todo.txt T3): what wilson reads from parse_from_source, checked on every example file.

The checks are those of wilson's check_data_dict (wilson_gv, src/wilson_suite/wilson_system/system_data.py),
copied here so the test does not import wilson:
    modes         nc_sqrt_eigval has the keys 0 .. n-1
    fundamentals  every mode m has its fundamental, key (str(m),), in harmonic_states and anharmonic_states
    ground state  the ground state, key (), if there, is at 0 cm-1
    properties    shape: mode indices first, then x, y, z, e.g. n = 6: polhess (6, 6, 3, 3);
                  symmetric in the mode indices, e.g. cff[0, 2, 2] == cff[2, 0, 2]
wilson looks for the fundamentals only in harmonic_states, and skips cff_rc and qff_rc (not in its
PROP_INDEX_COUNTS). This test checks those too: the contract covers them (todo.txt Rules).

Known failures are marked xfail with the todo item. The marks are strict: once the item is fixed, the test
fails as XPASS, so the mark gets removed.
"""
import functools

import numpy as np
import pytest

from tests.test_baseline import SOURCES, parse

# property: (number of mode indices, number of x, y, z indices), as wilson's PROP_INDEX_COUNTS
INDEX_COUNTS = {'dipgrad': (1, 1), 'diphess': (2, 1), 'polgrad': (1, 2), 'polhess': (2, 2),
                'cff': (3, 0), 'cff_rc': (3, 0), 'qff': (4, 0), 'qff_rc': (4, 0)}
# what wilson asks for in every request (MolSystemData.load), and the properties
NAMES = ['nc_sqrt_eigval', 'harmonic_states', 'anharmonic_states', *INDEX_COUNTS]
# wilson's SYMMETRY_RTOL, SYMMETRY_ATOL
RTOL, ATOL = 1e-12, 1e-18


@functools.cache
def result(source):
    return parse(source, NAMES)


def known_failures(source, name=None):
    reasons = []
    if source == 'c4_FORM_HF_cc-pVQZ':
        reasons.append('todo C2: HFcc_pVQZ has no MOLDEN file, so every request raises FileNotFoundError')
    if SOURCES[source][0] == 'cfour' and name == 'diphess':
        reasons.append('todo C1: CFOUR diphess is not symmetric (each order of a mode pair read on its own)')
    return [pytest.mark.xfail(reason='; '.join(reasons), strict=True)] if reasons else []


SOURCE_PARAMS = [pytest.param(s, marks=known_failures(s)) for s in SOURCES]
PROPERTY_PARAMS = [pytest.param(s, name, marks=known_failures(s, name)) for s in SOURCES for name in INDEX_COUNTS]


@pytest.mark.parametrize('source', SOURCE_PARAMS)
def test_modes(source):
    freqs = result(source)['nc_sqrt_eigval']
    assert isinstance(freqs, dict) and freqs, f'nc_sqrt_eigval: needs a dict, got {freqs!r}'
    assert set(freqs) == set(range(len(freqs))), f'nc_sqrt_eigval: needs the keys 0 .. n-1, got {list(freqs)}'


@pytest.mark.parametrize('source', SOURCE_PARAMS)
@pytest.mark.parametrize('states', ['harmonic_states', 'anharmonic_states'])
def test_fundamentals(source, states):
    data = result(source)
    have = {int(key[0]) for key in data[states] if len(key) == 1}
    missing = [m for m in range(len(data['nc_sqrt_eigval'])) if m not in have]
    assert not missing, f'{states} lacks the fundamentals of modes {missing}'


@pytest.mark.parametrize('source', SOURCE_PARAMS)
@pytest.mark.parametrize('states', ['harmonic_states', 'anharmonic_states'])
def test_ground_state(source, states):
    ground = result(source)[states].get(())
    assert ground in (None, 0.), f'{states}: the ground state () at {ground} cm-1; the energies count from it'


@pytest.mark.parametrize('source, name', PROPERTY_PARAMS)
def test_property(source, name):
    data = result(source)
    assert data.get(name) is not None, f'{name} not returned'
    n = len(data['nc_sqrt_eigval'])
    n_modes, n_xyz = INDEX_COUNTS[name]
    vals = np.asarray(data[name])
    expected = (n,) * n_modes + (3,) * n_xyz
    assert vals.shape == expected, f'{name}: shape {vals.shape}, expected {expected} (mode indices, then x, y, z)'

    for i in range(n_modes - 1):        # symmetric under every swap of neighbours: under every order
        swapped = np.swapaxes(vals, i, i + 1)
        if not np.allclose(vals, swapped, rtol=RTOL, atol=ATOL):
            idx = np.unravel_index(np.argmax(np.abs(vals - swapped)), vals.shape)
            other = list(idx)
            other[i], other[i + 1] = other[i + 1], other[i]
            pytest.fail(f'{name}: not symmetric in its mode indices, {name}{list(map(int, idx))} = {vals[idx]:.6g}, '
                        f'{name}{list(map(int, other))} = {vals[tuple(other)]:.6g}')
