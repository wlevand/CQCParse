"""
CFOUR vs Gaussian, formaldehyde HF/cc-pVTZ and HF/cc-pVQZ: the same molecule and method from two programs, read
by two different readers, so the two results must agree. It covers the dipole and polarizability derivatives, so
it checks their unit conversions, and the harmonic frequencies and states, B and the anharmonic states. The force
constants: test_cfour_matches_gaussian (tests/test_force_constants.py).

Matching the two results:
    - modes: by harmonic frequency (both readers number them by ascending frequency, but two close modes could
      swap between the programs);
    - the axes and the sign of each mode: each program orients the molecule its own way, and the sign of a normal
      coordinate is arbitrary. The axes: the signed permutation of x, y, z (one of 48) that fits best; the mode
      signs: fitted to the first derivatives. For a symmetric molecule several axes fit equally well (a mirror
      plane of the molecule flips an axis and the signs of some modes together); any of them will do.
Neither fit can hide a wrong unit factor, which changes the size of the values.
Measured: frequencies within 0.004 cm-1; each derivative tensor within 7e-5 of its largest value;
B within 8e-7 of the value; harmonic states within 0.01 cm-1; anharmonic states of nu1-nu3 within 0.4 cm-1.
"""
import itertools

import numpy as np
import pytest

from CQCParse.utils import state_key
from tests.test_baseline import parse

NAMES = ['nc_sqrt_eigval', 'harmonic_states', 'anharmonic_states', 'B', 'dipgrad', 'diphess', 'polgrad', 'polhess']
DERIVATIVES = ['dipgrad', 'diphess', 'polgrad', 'polhess']


def signed_permutations():
    """
    The 48 ways to relabel and flip the axes x, y, z: each a 3x3 matrix r with one +1 or -1 in each row and column,
    so that r @ v is the vector v in the new axes. E.g. order (1, 0, 2) and signs (1, -1, 1): new x = old y,
    new y = -old x, new z = old z.
    """
    for order in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            r = np.zeros((3, 3))
            r[range(3), order] = signs
            yield r


def to_gaussian(c4, perm, signs, r):
    """
    CFOUR's derivative tensors turned into Gaussian's mode order, mode signs and axes, to compare with Gaussian's.
        perm[m]   the CFOUR mode of Gaussian mode m
        signs[m]  +1 or -1: the sign that turns that CFOUR mode into Gaussian mode m
        r         CFOUR x, y, z -> Gaussian's (one of signed_permutations)
    Each mode index gets its mode's sign; each x, y, z index gets r: a vector v becomes r v, a 3x3 tensor a becomes
    r a r^T. E.g. diphess[m, n] = signs[m] signs[n] r (CFOUR diphess[perm[m], perm[n]]).
    """
    s = np.asarray(signs, dtype=float)
    ss = np.outer(s, s)
    return {'dipgrad': s[:, None] * (c4['dipgrad'][perm] @ r.T),
            'diphess': ss[:, :, None] * (c4['diphess'][np.ix_(perm, perm)] @ r.T),
            'polgrad': s[:, None, None] * np.einsum('ij,mjk,lk->mil', r, c4['polgrad'][perm], r),
            'polhess': ss[:, :, None, None] * np.einsum('ij,mnjk,lk->mnil', r, c4['polhess'][np.ix_(perm, perm)], r)}


def relative_gap(a, b):
    """The largest difference between a and b, divided by the largest absolute value in b (the reference)."""
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.fixture(scope='module', params=['cc-pVTZ', 'cc-pVQZ'])
def pair(request):
    """
    The Gaussian and CFOUR results of one basis set (request.param), matched as the module docstring says:
        1. modes: perm[m] is the CFOUR mode whose harmonic frequency is closest to Gaussian mode m's;
        2. for each of the 48 axis choices r (signed_permutations), the sign of each mode: +1 if Gaussian's first
           derivatives of mode m (dipgrad and polgrad together) point the same way as CFOUR's, turned by r, else -1;
        3. of the 48, the r whose matched tensors (to_gaussian) are closest to Gaussian's (relative_gap).
    Returns a dict: g16 and c4 (the two results), perm, w_g16 and w_c4 (harmonic frequencies, cm-1, in each
    program's mode order), r, and matched (CFOUR's derivative tensors in Gaussian's modes and axes).
    Module scope: the fit runs once per basis set, for all the tests here.
    """
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


def test_geometry_and_normal_modes():
    """
    equilibrium_geometry and normal_modes in the same units from both readers: formaldehyde HF/cc-pVTZ (the CFOUR
    HF/cc-pVQZ folder has no MOLDEN). Compared without fitting the axes or the mode signs: the interatomic distances
    (bohr), and each mode's displacement length on each atom (dx/dQ, m_e^(-1/2)), the modes paired in their order.
    A geometry in angstrom would be off by a factor 1.89; modes per amu^(1/2) bohr instead of m_e^(1/2) bohr by 42.7.
    Measured: distances within 2e-6 bohr; lengths within 6e-6 of the largest (Gaussian's 5-decimal modes).
    """
    g16, c4 = (parse(f'{program}_FORM_HF_cc-pVTZ', ['equilibrium_geometry', 'atoms', 'normal_modes'])
               for program in ('g16', 'c4'))
    assert g16['atoms'] == c4['atoms']

    def distances(geometry):
        return np.linalg.norm(geometry[:, None] - geometry[None], axis=2)

    def lengths(modes):
        return np.array([np.linalg.norm(modes[k], axis=1) for k in sorted(modes)])

    np.testing.assert_allclose(distances(g16['equilibrium_geometry']), distances(c4['equilibrium_geometry']),
                               atol=1e-5)
    assert relative_gap(lengths(c4['normal_modes']), lengths(g16['normal_modes'])) < 1e-4


def test_rotational_constants(pair):
    """B (cm-1) per axis, in Gaussian's axes."""
    b_c4 = np.abs(pair['r']) @ pair['c4']['B']
    np.testing.assert_allclose(b_c4, pair['g16']['B'], rtol=1e-5)


def mapped_states(pair, kind):
    """
    CFOUR's states (kind: 'harmonic_states' or 'anharmonic_states') with Gaussian's mode numbers in the keys, so
    that the two dicts have the same keys. E.g. if perm[4] == 2 (CFOUR mode 2 is Gaussian mode 4), CFOUR's key
    ('2', '2') becomes ('4', '4').
    """
    to_g16 = {c4_mode: m for m, c4_mode in enumerate(pair['perm'])}
    return {state_key(to_g16[int(i)] for i in key): value for key, value in pair['c4'][kind].items()}


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
