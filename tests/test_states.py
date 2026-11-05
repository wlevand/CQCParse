"""
harmonic_states and anharmonic_states of the example files, at every level the files print: 1, 2 and 3 quanta.
The checks use counting and the physics of the states, not values CQCParse returned before.
    test_all_states_up_to_3_quanta  every state up to 3 quanta is there, with the same keys in both dicts. With n
                                    modes: n states of 1 quantum, n(n+1)/2 of 2 and n(n+1)(n+2)/6 of 3, e.g.
                                    formaldehyde (n = 6): 6 + 21 + 56 = 83
    test_cfour_keys_of_two_digit_modes  CFOUR keys in numeric order, as Gaussian's: ('2', '10')
    test_harmonic_sum_rule          a harmonic state is the sum of its modes' harmonic frequencies, e.g.
                                    ('0', '0', '3') = 2 w0 + w3
    test_anharmonic_3_quanta        each 3-quanta state is what VPT2 gives from the 1- and 2-quanta states
                                    (vpt2_prediction)
The tolerances are the print rounding: both programs print the energies with 3 decimals (CFOUR the harmonic ones
with 6), so each printed value is within 0.0005 cm-1 of the program's own.
"""
import itertools
from collections import Counter

import pytest

from CQCParse.parsing import parse_from_source
from tests.test_baseline import BASE, SOURCES, parse

C4_SOURCES = [s for s, (kind, *_) in SOURCES.items() if kind == 'cfour']
G16_FORM_SOURCES = [s for s in SOURCES if s.startswith('g16_FORM')]


def key_of(modes):
    """The key of the state with one quantum per mode in modes, as the readers write it: the mode numbers as
    strings, in numeric order. E.g. (3, 0, 0) -> ('0', '0', '3'), (10, 2) -> ('2', '10'). Written out here, not
    CQCParse.utils.state_key, so that these tests check the rule. ACAC and ACDM have two-digit modes."""
    return tuple(str(m) for m in sorted(modes))


@pytest.mark.parametrize('source', list(SOURCES))
def test_all_states_up_to_3_quanta(source):
    data = parse(source, BASE)
    n = len(data['nc_sqrt_eigval'])
    expected = {key_of(modes) for q in (1, 2, 3) for modes in itertools.combinations_with_replacement(range(n), q)}
    for states in ('harmonic_states', 'anharmonic_states'):
        keys = set(data[states])
        missing, extra = sorted(expected - keys), sorted(keys - expected)
        assert not missing and not extra, (f'{states}: {len(missing)} states missing (e.g. {missing[:3]}), '
                                           f'{len(extra)} not expected (e.g. {extra[:3]})')


def test_cfour_keys_of_two_digit_modes(tmp_path):
    """
    With 10 or more modes, CFOUR keys are in numeric order, as Gaussian's are: ('2', '10'), not ('10', '2'). The CFOUR
    examples have 6 modes, so a made-up out file: 6 atoms, CFOUR modes 7 .. 18 as vibrations 0 .. 11, and only the
    lines the reader needs for the states. Rows: I J K L M, NI NJ NK NL NM, anharmonic, intensity, harmonic.
    """
    rows = [f'{m} 0 0 0 0 1 0 0 0 0 {1000.0 + m:.3f} 1.000000 {2000.0 + m:.6f}' for m in range(7, 19)]
    rows += ['17 9 0 0 0 1 1 0 0 0 3000.000 1.000000 4000.000000',        # vibrations 10 and 2
             '17 9 0 0 0 2 1 0 0 0 5000.000 1.000000 6000.000000']        # vibrations 10, 10 and 2
    (tmp_path / 'out').write_text('\n'.join([f'Normal coordinate {k}' for k in range(1, 19)]
                                            + ['All levels with up to three quanta'] + rows
                                            + ['Electric dipole moment function in dimensionless normal coordinates']))
    data = parse_from_source(['harmonic_states', 'anharmonic_states'], 'cfour', str(tmp_path), '', '')
    assert data['harmonic_states'][('2', '10')] == 4000.0
    assert data['anharmonic_states'][('2', '10', '10')] == 5000.0
    expected = {key_of([m]) for m in range(12)} | {('2', '10'), ('2', '10', '10')}
    assert set(data['harmonic_states']) == set(data['anharmonic_states']) == expected


@pytest.mark.parametrize('source', list(SOURCES))
def test_harmonic_sum_rule(source):
    """
    Each harmonic state is the sum of the harmonic frequencies of its modes, and the fundamentals are
    nc_sqrt_eigval. This checks that every key is read with the right modes, two-digit modes (ACAC, ACDM) included.
    Tolerance: a state of 3 quanta and its up to 3 fundamentals, each rounded by up to 0.0005 cm-1.
    """
    data = parse(source, BASE)
    w, harmonic = data['nc_sqrt_eigval'], data['harmonic_states']
    assert all(harmonic[key_of([m])] == w[m] for m in w)
    gaps = {key: value - sum(w[int(m)] for m in key) for key, value in harmonic.items()}
    worst = max(gaps, key=lambda key: abs(gaps[key]))
    assert abs(gaps[worst]) <= 4 * 0.0005 + 1e-9, \
        f'{worst}: {harmonic[worst]} cm-1, the sum of its harmonic frequencies is {harmonic[worst] - gaps[worst]}'


def vpt2_prediction(states, quanta):
    """
    The energy (cm-1) of a state that VPT2 gives from the 1- and 2-quanta states, if no resonance is treated.
    VPT2, counted from the ground state, with w_i the harmonic frequencies and x_ij the anharmonic constants (the
    X matrix; non-degenerate modes, as in all the files tested here):
        E(n) = sum_i w_i n_i + sum_i x_ii (n_i^2 + n_i) + sum_{i<j} x_ij (n_i n_j + n_i / 2 + n_j / 2)
    The x terms are the anharmonic correction. E.g. one mode: E(nu) = w + 2x, E(2 nu) = 2w + 6x, E(3 nu) = 3w + 12x.
    Here w is replaced by the anharmonic fundamentals E(nu_i), which hold their own correction, and x by the
    2-quanta states, so that only anharmonic_states is needed:
        E(n) = sum_i n_i E(nu_i) + sum_i x_ii n_i (n_i - 1) + sum_{i<j} x_ij n_i n_j,
        x_ii = (E(2 nu_i) - 2 E(nu_i)) / 2,    x_ij = E(nu_i + nu_j) - E(nu_i) - E(nu_j)
    E.g. one mode: E(3 nu) = 3 E(nu) + 6x. Formaldehyde HF/cc-pVTZ (CFOUR), mode 0: E(nu) = 1320.210 and
    E(2 nu) = 2631.835 give x = -4.2925 and E(3 nu) = 3934.875; CFOUR prints 3934.874.
    states: anharmonic_states; quanta: {mode: number of quanta}, e.g. {0: 2, 3: 1} for ('0', '0', '3').
    """
    def energy(*modes):
        return states[key_of(modes)]

    e = sum(c * energy(i) for i, c in quanta.items())
    for i, c in quanta.items():
        e += (energy(i, i) - 2 * energy(i)) / 2 * c * (c - 1)
    for i, j in itertools.combinations(quanta, 2):
        e += (energy(i, j) - energy(i) - energy(j)) * quanta[i] * quanta[j]
    return e


@pytest.mark.parametrize('source', C4_SOURCES + ['g16_CO_HF_STO-3G'] + G16_FORM_SOURCES)
def test_anharmonic_3_quanta(source):
    """
    Each 3-quanta state is vpt2_prediction from the 1- and 2-quanta states. CFOUR treats no resonance in these
    files, so all its states; CO has 1 mode. Gaussian treats resonant states its own way (see
    tests/test_cfour_vs_gaussian.py), so in formaldehyde only the states of nu1-nu3 (Gaussian's A modes 1-3), and
    not ACAC or ACDM, which have too many resonances.
    Tolerance: the prediction is a sum of printed energies whose factors add up to 6 in size, e.g.
    E(3 nu_i) = 3 E(2 nu_i) - 3 E(nu_i), and the state itself: 7 values rounded by up to 0.0005 cm-1.
    """
    data = parse(source, BASE + (['modes_mapping'] if source in G16_FORM_SOURCES else []))
    states = data['anharmonic_states']
    if source in G16_FORM_SOURCES:
        modes = {data['modes_mapping'][a] - 1 for a in (1, 2, 3)}
    else:
        modes = set(range(len(data['nc_sqrt_eigval'])))
    three = [key for key in states if len(key) == 3 and {int(m) for m in key} <= modes]
    assert three
    gaps = {key: states[key] - vpt2_prediction(states, Counter(int(m) for m in key)) for key in three}
    worst = max(gaps, key=lambda key: abs(gaps[key]))
    assert abs(gaps[worst]) <= 7 * 0.0005 + 1e-9, \
        f'{worst}: {states[worst]} cm-1, VPT2 from the 1- and 2-quanta states gives {states[worst] - gaps[worst]}'
