"""
Contract test: what wilson reads from parse_from_source, checked on every example file.

The checks are those of wilson's check_data_dict (wilson_gv, src/wilson_suite/wilson_system/system_data.py),
copied here so the test does not import wilson:
    modes         nc_sqrt_eigval has the keys 0 .. n-1
    fundamentals  every mode m has its fundamental, key (str(m),), in harmonic_states and anharmonic_states
    properties    shape: mode indices first, then x, y, z, e.g. n = 6: polhess (6, 6, 3, 3);
                  symmetric in the mode indices, e.g. cff[0, 2, 2] == cff[2, 0, 2]
wilson looks for the fundamentals only in harmonic_states, and skips cff_rc and qff_rc (not in its
PROP_INDEX_COUNTS). This test checks those too: the contract covers them.
wilson also checks that a ground state, key (), is at 0 cm-1. No reader returns one, so it is not checked here.
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


@pytest.mark.parametrize('source', list(SOURCES))
def test_modes(source):
    freqs = result(source)['nc_sqrt_eigval']
    assert isinstance(freqs, dict) and freqs, f'nc_sqrt_eigval: needs a dict, got {freqs!r}'
    assert set(freqs) == set(range(len(freqs))), f'nc_sqrt_eigval: needs the keys 0 .. n-1, got {list(freqs)}'


@pytest.mark.parametrize('source', list(SOURCES))
@pytest.mark.parametrize('states', ['harmonic_states', 'anharmonic_states'])
def test_fundamentals(source, states):
    data = result(source)
    have = {int(key[0]) for key in data[states] if len(key) == 1}
    missing = [m for m in range(len(data['nc_sqrt_eigval'])) if m not in have]
    assert not missing, f'{states} lacks the fundamentals of modes {missing}'


@pytest.mark.parametrize('source', list(SOURCES))
@pytest.mark.parametrize('name', list(INDEX_COUNTS))
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
