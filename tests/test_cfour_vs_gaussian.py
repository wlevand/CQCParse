"""
CFOUR vs Gaussian, formaldehyde HF/cc-pVTZ and HF/cc-pVQZ (todo T4): the only check here that does not assume
CQCParse is right. It covers the dipole and polarizability derivatives, so it checks their unit conversions
(todo R4), and the harmonic frequencies and states, B and the anharmonic states. The force constants:
test_cfour_matches_gaussian (tests/test_force_constants.py).

Matching the two results:
    - modes: by harmonic frequency (both readers number them by ascending frequency, but two close modes could
      swap between the programs);
    - the axes and the sign of each mode: each program orients the molecule its own way, and the sign of a normal
      coordinate is arbitrary. The axes: the signed permutation of x, y, z (one of 48) that fits best; the mode
      signs: fitted to the first derivatives. For a symmetric molecule several axes fit equally well (a mirror
      plane of the molecule flips an axis and the signs of some modes together); any of them will do.
Neither fit can hide a wrong unit factor, which changes the size of the values.
Measured (October 2026): frequencies within 0.003 cm-1; each derivative tensor within 7e-5 of its largest value;
B within 5e-7 of the value; harmonic states within 0.01 cm-1; anharmonic states of nu1-nu3 within 0.4 cm-1.
"""
import itertools

import numpy as np
import pytest

from tests.test_baseline import parse

NAMES = ['nc_sqrt_eigval', 'harmonic_states', 'anharmonic_states', 'B', 'dipgrad', 'diphess', 'polgrad', 'polhess']
DERIVATIVES = ['dipgrad', 'diphess', 'polgrad', 'polhess']


def signed_permutations():
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            r = np.zeros((3, 3))
            r[range(3), order] = signs
            yield r


def to_gaussian(c4, perm, signs, r):
    """CFOUR's derivative tensors in Gaussian's mode order, mode signs and axes (r: CFOUR x, y, z -> Gaussian's)."""
    s = np.asarray(signs, dtype=float)
    ss = np.outer(s, s)
    return {'dipgrad': s[:, None] * (c4['dipgrad'][perm] @ r.T),
            'diphess': ss[:, :, None] * (c4['diphess'][np.ix_(perm, perm)] @ r.T),
            'polgrad': s[:, None, None] * np.einsum('ij,mjk,lk->mil', r, c4['polgrad'][perm], r),
            'polhess': ss[:, :, None, None] * np.einsum('ij,mnjk,lk->mnil', r, c4['polhess'][np.ix_(perm, perm)], r)}


def relative_gap(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.fixture(scope='module', params=['cc-pVTZ', 'cc-pVQZ'])
def pair(request):
    g16 = parse(f'g16_FORM_HF_{request.param}', NAMES + ['modes_mapping'])
    c4 = parse(f'c4_FORM_HF_{request.param}', NAMES)
    n = len(g16['nc_sqrt_eigval'])
    w_g16 = np.array([g16['nc_sqrt_eigval'][m] for m in range(n)])
    w_c4 = np.array([c4['nc_sqrt_eigval'][m] for m in range(n)])
    perm = [int(np.argmin(np.abs(w_c4 - w))) for w in w_g16]          # the CFOUR mode of each Gaussian mode
    best = None
    for r in signed_permutations():
        signs = []
        for m in range(n):
            # a mode with no first derivative at all (inactive in IR and Raman) would get +1; formaldehyde has none
            mine = np.concatenate([g16['dipgrad'][m], g16['polgrad'][m].ravel()])
            theirs = np.concatenate([r @ c4['dipgrad'][perm[m]], (r @ c4['polgrad'][perm[m]] @ r.T).ravel()])
            signs.append(1 if mine @ theirs >= 0 else -1)
        matched = to_gaussian(c4, perm, signs, r)
        score = max(relative_gap(matched[k], g16[k]) for k in DERIVATIVES)
        if best is None or score < best[0]:
            best = (score, r, matched)
    _, r, matched = best
    return {'g16': g16, 'c4': c4, 'perm': perm, 'w_g16': w_g16, 'w_c4': w_c4, 'r': r, 'matched': matched}


def test_harmonic_frequencies(pair):
    assert sorted(pair['perm']) == list(range(len(pair['perm'])))
    assert np.abs(pair['w_c4'][pair['perm']] - pair['w_g16']).max() < 0.05


@pytest.mark.parametrize('name', DERIVATIVES)
def test_derivatives(pair, name):
    """
    dipgrad, diphess, polgrad, polhess in atomic units agree within 2e-4 of the tensor's largest value. The two
    readers convert by different routes (Gaussian: per mass-weighted coordinate in amu^(1/2) Bohr, the dipole in
    Debye; CFOUR: per dimensionless coordinate, times sqrt(w)), so this checks both conversions.
    """
    assert relative_gap(pair['matched'][name], pair['g16'][name]) <= 2e-4


def test_rotational_constants(pair):
    """B (cm-1) per axis, in Gaussian's axes."""
    b_c4 = np.abs(pair['r']) @ pair['c4']['B']
    np.testing.assert_allclose(b_c4, pair['g16']['B'], rtol=1e-5)


def mapped_states(pair, kind):
    """CFOUR's states, with the modes renamed to Gaussian's."""
    to_g16 = {c4_mode: m for m, c4_mode in enumerate(pair['perm'])}
    return {tuple(sorted(str(to_g16[int(i)]) for i in key)): value for key, value in pair['c4'][kind].items()}


def test_harmonic_states(pair):
    c4, g16 = mapped_states(pair, 'harmonic_states'), pair['g16']['harmonic_states']
    assert c4.keys() == g16.keys()
    assert max(abs(c4[k] - g16[k]) for k in g16) < 0.05


def test_anharmonic_states(pair):
    """
    The states of nu1, nu2 and nu3 only (Gaussian's A modes 1-3, renumbered by modes_mapping) agree within 1 cm-1.
    The others differ by the programs' resonance treatment, not by the readers: Gaussian finds a Fermi resonance
    nu5 = nu3 + nu6 (gap 130 cm-1) and a 2-2 Darling-Dennison resonance of nu4 and nu6, and treats them
    variationally; CFOUR removes resonant terms only below a gap of 50 cm-1 ("Delta_omega = 50 cm-1" in its out
    file). So nu5 differs by 29 cm-1, and states with nu4, nu5 or nu6 by up to 141 cm-1.
    """
    c4, g16 = mapped_states(pair, 'anharmonic_states'), pair['g16']['anharmonic_states']
    assert c4.keys() == g16.keys()
    nu1_to_nu3 = {str(pair['g16']['modes_mapping'][a] - 1) for a in (1, 2, 3)}
    plain = [k for k in g16 if set(k) <= nu1_to_nu3]
    assert len(plain) == 19
    assert max(abs(c4[k] - g16[k]) for k in plain) < 1
