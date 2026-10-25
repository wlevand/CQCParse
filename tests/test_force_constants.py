"""
The conversion of cubic and quartic force constants from cm-1 (dimensionless normal coordinates) to atomic units,
CQCParse.utils.reduced_to_au, used by both readers for cff and qff (todo C3, R1).
"""
import numpy as np
import pytest
from scipy import constants

from CQCParse.parsing.parseGaussian_forWilson import (
    get_cubic_post,
    get_quartic_post,
    parse_cubic_constants,
    parse_quartic_constants,
)
from CQCParse.utils import reduced_to_au
from tests.test_baseline import BASE, EXAMPLES, SOURCES, parse

# m_u / m_e: Gaussian's column K is in Hartree amu^(-n/2) Bohr^(-n), cff and qff in Hartree m_e^(-n/2) a0^(-n)
AMU_IN_ME = constants.physical_constants['atomic mass constant'][0] / constants.physical_constants['atomic unit of mass'][0]
GAUSSIAN_SOURCES = [s for s, (kind, *_) in SOURCES.items() if kind == 'gaussian']


def test_reduced_to_au_example():
    """The worked example of reduced_to_au's docstring: formaldehyde HF/cc-pVTZ, Gaussian term (1,1,1)."""
    k = reduced_to_au(np.full((1, 1, 1), -1312.24856), [3084.240])[0, 0, 0]
    assert k == pytest.approx(-9.9604e-6, abs=1e-10)
    # Gaussian's own column K for this term: -0.77521, 5 decimals
    assert abs(k - (-0.77521) / AMU_IN_ME ** 1.5) <= 0.5e-5 / AMU_IN_ME ** 1.5


@pytest.mark.parametrize('source', GAUSSIAN_SOURCES)
@pytest.mark.parametrize('name, order', [('cff', 3), ('qff', 4)])
def test_matches_gaussian_column_k(source, name, order):
    """
    cff and qff, converted from column FI, agree with Gaussian's own column K on every term, within K's rounding:
    half its last decimal (0.5e-5 in amu units), plus a little from the rounding of FI and of the frequencies.
    This checks the formula independently of the readers, for cubic and quartic alike (todo R1, C3).
    """
    data = parse(source, BASE + [name])
    n = len(data['nc_sqrt_eigval'])
    lines = [line.strip() for line in open(EXAMPLES + '/' + SOURCES[source][1])]
    if order == 3:
        rows = parse_cubic_constants(lines)[0][['I', 'J', 'K', 'K(I,J,K)']].to_numpy()
        from_k = get_cubic_post(n, rows)
    else:
        rows = parse_quartic_constants(lines)[0][['I', 'J', 'K', 'L', 'K(I,J,K,L)']].to_numpy()
        from_k = get_quartic_post(n, rows)
    from_k = from_k / AMU_IN_ME ** (order / 2)
    rounding = 0.5e-5 / AMU_IN_ME ** (order / 2)
    assert np.abs(data[name] - from_k).max() <= 1.5 * rounding


@pytest.mark.parametrize('basis', ['cc-pVTZ', 'cc-pVQZ'])
@pytest.mark.parametrize('name', ['cff', 'cff_rc', 'qff', 'qff_rc'])
def test_cfour_matches_gaussian(basis, name):
    """
    Formaldehyde HF in both programs: after matching the modes by harmonic frequency, the force constants agree
    within 1e-3 of the largest term (todo C3: CFOUR qff was about 570x too large). Absolute values, because some
    modes have the opposite sign in the two programs. Measured: up to 2.1e-4, and 7.5e-4 for qff_rc HF/cc-pVQZ (one
    small term, (4,4,5,6) in Gaussian's numbering, differs already in the two programs' files).
    """
    names = ['nc_sqrt_eigval', name]
    g16, c4 = parse(f'g16_FORM_HF_{basis}', names), parse(f'c4_FORM_HF_{basis}', names)
    n = len(g16['nc_sqrt_eigval'])
    w_g16 = np.array([g16['nc_sqrt_eigval'][m] for m in range(n)])
    w_c4 = np.array([c4['nc_sqrt_eigval'][m] for m in range(n)])
    perm = [int(np.argmin(np.abs(w_c4 - w))) for w in w_g16]     # the CFOUR mode of each Gaussian mode
    assert sorted(perm) == list(range(n)) and np.abs(w_c4[perm] - w_g16).max() < 1
    g16_vals = np.abs(g16[name])
    c4_vals = np.abs(c4[name][np.ix_(*[perm] * g16[name].ndim)])
    assert np.abs(c4_vals - g16_vals).max() <= 1e-3 * g16_vals.max()
