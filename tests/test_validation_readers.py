"""
The readers kept for checking wilson's VPT2 against the programs' own (todo_cleanup Q10). parse_from_source does not
use them: Gaussian's X matrix (parse_anharmonic_x_matrix) and the resonances each program found
(get_detected_resonances_g16, get_detected_resonances_c4).
"""
import numpy as np
import pytest

from CQCParse.parsing.parseCFOUR_forWilson import get_detected_resonances_c4
from CQCParse.parsing.parseGaussian_forWilson import (
    get_detected_resonances_g16,
    parse_anharmonic_x_matrix,
)
from tests.test_baseline import BASE, EXAMPLES, SOURCES, parse

GAUSSIAN_SOURCES = [s for s, (kind, *_) in SOURCES.items() if kind == 'gaussian']
CFOUR_SOURCES = [s for s, (kind, *_) in SOURCES.items() if kind == 'cfour']


def lines_of(source):
    with open(EXAMPLES + '/' + SOURCES[source][1]) as file:
        return file.readlines()


@pytest.mark.parametrize('source', GAUSSIAN_SOURCES)
def test_x_matrix(source):
    """n x n (n modes), symmetric, and the total is the sum of its three parts within the file's print rounding
    (measured: at most 8e-3 cm-1, ACDM)."""
    x = parse_anharmonic_x_matrix(lines_of(source))
    n = len(parse(source, ['nc_sqrt_eigval'])['nc_sqrt_eigval'])
    total = x['total_anharmonic']
    assert total.shape == x['matrix_size'] == (n, n)
    np.testing.assert_array_equal(total, total.T)
    parts = x['coriolis_contributions'] + x['3rd_deriv_contributions'] + x['4th_deriv_contributions']
    assert np.abs(total - parts).max() < 0.01


@pytest.mark.parametrize('source', ['g16_FORM_HF_cc-pVTZ', 'g16_FORM_B3LYP_cc-pVQZ'])
def test_x_matrix_order_matches_states(source):
    """
    The X matrix is in Gaussian's A order, the order of anharmonic_states: E(2 nu_i) - 2 E(nu_i) = 2 X_ii and
    E(nu_i + nu_j) - E(nu_i) - E(nu_j) = X_ij, within the states' 3 decimals. Modes 0, 1, 2 only (nu1-nu3): nu4-nu6
    are in Gaussian's resonance treatment (see tests/test_cfour_vs_gaussian.py).
    """
    x = parse_anharmonic_x_matrix(lines_of(source))['total_anharmonic']
    states = parse(source, BASE)['anharmonic_states']

    def energy(*modes):
        return states[tuple(sorted(str(m) for m in modes))]
    for i in range(3):
        assert (energy(i, i) - 2 * energy(i)) / 2 == pytest.approx(x[i, i], abs=5e-3)
        for j in range(i):
            assert energy(i, j) - energy(i) - energy(j) == pytest.approx(x[i, j], abs=5e-3)


def test_x_matrix_missing_section():
    with pytest.raises(ValueError, match='Anharmonic X Matrix'):
        parse_anharmonic_x_matrix(['nothing here\n'])


@pytest.mark.parametrize('source, count, first', [('g16_FORM_HF_cc-pVTZ', 1, (4, 2, 5)),
                                                  ('g16_FORM_B3LYP_cc-pVQZ', 2, None),
                                                  ('g16_ACAC_B3LYP_cc-pVQZ', 8, (1, 3, 6)),
                                                  ('g16_ACDM_B3LYP_cc-pVQZ', 27, None)])
def test_fermi_resonances_g16(source, count, first):
    """One (I, J, K) per row of Gaussian's Fermi table, modes from 0 in A order. It returned None before (todo_cleanup
    Q10): it tested the list of lines for an element equal to "Resonance Analysis"."""
    found = get_detected_resonances_g16(lines_of(source))
    assert len(found) == count
    assert all(len(r) == 3 and all(0 <= m < 42 for m in r) for r in found)
    if first:
        assert found[0] == first


def test_fermi_resonances_g16_no_section():
    """CO has 1 mode, so no resonance analysis."""
    with pytest.raises(ValueError, match='Resonance Analysis'):
        get_detected_resonances_g16(lines_of('g16_CO_HF_STO-3G'))


@pytest.mark.parametrize('source', CFOUR_SOURCES)
def test_resonances_c4_none_in_examples(source):
    """CFOUR found no resonance in any example (formaldehyde's Fermi gap, 130 cm-1, is above its 50 cm-1)."""
    assert get_detected_resonances_c4(EXAMPLES + '/' + SOURCES[source][1] + '/out') == []


def test_resonances_c4_reads_lines(tmp_path):
    """It returned [] for every file: it looped over the characters of the file, not its lines (todo_cleanup Q10).
    A made-up out file: the format of a real resonance line is not known from the examples."""
    out = tmp_path / 'out'
    out.write_text(' Thresholds for removing resonance denominators:\n'
                   ' Resonance between mode 9 and combination 7 + 8\n'
                   ' some other line\n')
    assert get_detected_resonances_c4(str(out)) == ['Resonance between mode 9 and combination 7 + 8']
    out.write_text(' no thresholds line\n')
    with pytest.raises(ValueError, match='Thresholds'):
        get_detected_resonances_c4(str(out))
