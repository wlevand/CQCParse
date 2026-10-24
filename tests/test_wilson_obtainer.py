import pytest

from CQCParse.parsing.parse_wilson_obtainer import parse_cfour_output, parse_gaussian16_output, parse_from_source
from CQCParse.utils import PKG_ROOT
from tests.test_baseline import BASE, parse


@pytest.mark.parametrize('source', ['g16_FORM_HF_cc-pVTZ', 'c4_FORM_HF_cc-pVTZ'])
@pytest.mark.parametrize('name, partner', [('cff', 'cff_rc'), ('cff_rc', 'cff'), ('qff', 'qff_rc'), ('qff_rc', 'qff')])
def test_force_constants_by_name(source, name, partner):
    """A request for the force constants in one unit returns both units (todo A1), asked for as wilson asks."""
    result = parse(source, BASE + [name])
    assert name in result and partner in result, f'{name} asked for, returned {sorted(result)}'


def test_parse_from_source_cfour():
    print()
    complete_data = ['cff', 'anharmonic_states', 'nc_sqrt_eigval', 'dipgrad', 'B', 'polgrad', 'coriolis', 'polhess', 'qff', 'diphess']
    rq_data = dict.fromkeys(complete_data, None)
    base_file_loc = PKG_ROOT + '/CQCParse/files_examples/refinedc4/CCSDTcc_pVQZ'
    r1 = parse_from_source(source_type='cfour', requested_data=rq_data, base_file_loc=base_file_loc,
                           lvl_theory='CCSD(T)', basis_set='cc-pVQZ', linear=False)

    # print(r1)

# def test_parse_from_source_g16():
    print()
    complete_data = ['cff', 'anharmonic_states', 'nc_sqrt_eigval', 'dipgrad', 'B', 'polgrad', 'coriolis', 'polhess', 'qff', 'diphess']
    rq_data = dict.fromkeys(complete_data, None)
    base_file_loc = PKG_ROOT + '/CQCParse/files_examples/dftGaussian/FORM/B3LYPcc_pVDZ/g16_inputFull_3q.out'
    r2 = parse_from_source(source_type='gaussian', requested_data=rq_data, base_file_loc=base_file_loc,
                           lvl_theory='B3LYP', basis_set='cc-pVDZ', linear=False)

    # print(r2)

    import numpy as np
    for k in r1:
        assert type(r1[k]) is type(r2[k])
        if type(r1[k]) is np.ndarray:
            assert r1[k].shape == r2[k].shape
        if type(r1[k]) is dict:
            assert set(r1[k].keys()) == set(r2[k].keys())