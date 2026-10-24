import numpy as np
import pytest

from CQCParse.parsing.parse_wilson_obtainer import (parse_cfour_output, parse_gaussian16_output, parse_from_source,
                                                    is_linear)
from CQCParse.utils import PKG_ROOT
from tests.test_baseline import BASE, parse

FORM_PAIR = ['g16_FORM_HF_cc-pVTZ', 'c4_FORM_HF_cc-pVTZ']


@pytest.mark.parametrize('source', FORM_PAIR)
@pytest.mark.parametrize('name, partner', [('cff', 'cff_rc'), ('cff_rc', 'cff'), ('qff', 'qff_rc'), ('qff_rc', 'qff')])
def test_force_constants_by_name(source, name, partner):
    """A request for the force constants in one unit returns both units (todo A1), asked for as wilson asks."""
    result = parse(source, BASE + [name])
    assert name in result and partner in result, f'{name} asked for, returned {sorted(result)}'


@pytest.mark.parametrize('source', FORM_PAIR)
@pytest.mark.parametrize('name', ['B', 'coriolis', 'dipgrad', 'diphess', 'polgrad', 'polhess',
                                  'cff', 'cff_rc', 'qff', 'qff_rc'])
def test_name_alone(source, name):
    """A name asked for on its own gives the same value as asked for with BASE, as wilson asks (todo A2)."""
    np.testing.assert_array_equal(parse(source, [name])[name], parse(source, BASE + [name])[name])


@pytest.mark.parametrize('n_atoms, n_vib, linear', [(2, 1, True), (3, 4, True), (4, 7, True),
                                                    (3, 3, False), (4, 6, False)])
def test_is_linear(n_atoms, n_vib, linear):
    """Linear: 3N-5 vibrational modes (CO, CO2, C2H2); nonlinear: 3N-6 (H2O, formaldehyde) (todo C4)."""
    assert is_linear(n_atoms, n_vib) is linear


def test_is_linear_wrong_count():
    with pytest.raises(ValueError):
        is_linear(4, 8)


# a CFOUR out file for a linear triatomic (CO2-like: 9 normal coordinates, vibrations 6 .. 9), only the sections
# the reader needs for the states
LINEAR_CFOUR_OUT = """\
  Normal coordinates for reference point
{normal_coordinates}
                   All levels with up to three quanta
--------------------------------------------------------------------------------
  MODE MODE MODE MODE MODE                   Anharmonic    Anharm      Harmonic
   I    J    K    L    M   NI  NJ  NK  NL  NM Frequency   Intensity   Transition
--------------------------------------------------------------------------------
    6    0    0    0    0  1   0   0   0   0   660.000    10.000000    667.000000
    7    0    0    0    0  1   0   0   0   0   660.000    10.000000    667.000000
    8    0    0    0    0  1   0   0   0   0  1380.000     0.000000   1388.000000
    9    0    0    0    0  1   0   0   0   0  2340.000   500.000000   2349.000000
    6    0    0    0    0  2   0   0   0   0  1320.000     0.100000   1334.000000
    8    9    0    0    0  1   1   0   0   0  3710.000     5.000000   3737.000000

     Electric dipole moment function in dimensionless normal coordinates
""".format(normal_coordinates='\n'.join(f'  Normal coordinate {k:22d}\n' + '        0.0 0.0 0.0\n' * 3
                                        for k in range(1, 10)))


def test_cfour_linear_mode_numbers(tmp_path):
    """CFOUR numbers a linear molecule's vibrations from 6, not 7; the reader returns them as 0 .. n-1 (todo C4)."""
    (tmp_path / 'out').write_text(LINEAR_CFOUR_OUT)
    result = parse_from_source(dict.fromkeys(['nc_sqrt_eigval', 'harmonic_states', 'anharmonic_states']),
                               'cfour', str(tmp_path), lvl_theory='', basis_set='')
    assert result['nc_sqrt_eigval'] == {0: 667., 1: 667., 2: 1388., 3: 2349.}
    assert result['harmonic_states'][('0', '0')] == 1334.
    assert result['anharmonic_states'][('2', '3')] == 3710.


def test_linear_argument_ignored():
    with pytest.warns(DeprecationWarning):
        parse_from_source({'nc_sqrt_eigval': None}, 'cfour', PKG_ROOT + '/CQCParse/files_examples/refinedc4/HFcc_pVTZ',
                          lvl_theory='HF', basis_set='cc-pVTZ', linear=True)


def test_parse_from_source_cfour():
    print()
    complete_data = ['cff', 'anharmonic_states', 'nc_sqrt_eigval', 'dipgrad', 'B', 'polgrad', 'coriolis', 'polhess', 'qff', 'diphess']
    rq_data = dict.fromkeys(complete_data, None)
    base_file_loc = PKG_ROOT + '/CQCParse/files_examples/refinedc4/CCSDTcc_pVQZ'
    r1 = parse_from_source(source_type='cfour', requested_data=rq_data, base_file_loc=base_file_loc,
                           lvl_theory='CCSD(T)', basis_set='cc-pVQZ')

    # print(r1)

# def test_parse_from_source_g16():
    print()
    complete_data = ['cff', 'anharmonic_states', 'nc_sqrt_eigval', 'dipgrad', 'B', 'polgrad', 'coriolis', 'polhess', 'qff', 'diphess']
    rq_data = dict.fromkeys(complete_data, None)
    base_file_loc = PKG_ROOT + '/CQCParse/files_examples/dftGaussian/FORM/B3LYPcc_pVDZ/g16_inputFull_3q.out'
    r2 = parse_from_source(source_type='gaussian', requested_data=rq_data, base_file_loc=base_file_loc,
                           lvl_theory='B3LYP', basis_set='cc-pVDZ')

    # print(r2)

    import numpy as np
    for k in r1:
        assert type(r1[k]) is type(r2[k])
        if type(r1[k]) is np.ndarray:
            assert r1[k].shape == r2[k].shape
        if type(r1[k]) is dict:
            assert set(r1[k].keys()) == set(r2[k].keys())