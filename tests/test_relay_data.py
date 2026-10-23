import os

from CQCParse.relay.relay_data import DataVault
from CQCParse.utils import PKG_ROOT

EXAMPLES = PKG_ROOT + '/CQCParse/files_examples'

def test_DV_make_data_input_dict():
    """
    Test the make_data_input_dict method of the DataVault class.
    The CSV holds paths relative to its own folder; DataVault makes them absolute.
    """
    dv = DataVault(EXAMPLES + '/calculations.csv')

    source_program = "gaussian"
    mol_tuple = ("FORM", 'conf1', "B3LYP", "cc-pVQZ")

    result = dv.make_data_input_dict(source_program, mol_tuple)

    print("\nResult:", result)

    assert isinstance(result, dict), "Result should be a dictionary."
    assert "FORM" == result["files"]["mol_name"], "Molecule name should be in the result."
    assert result["files"]['log'] == EXAMPLES + '/dftGaussian/FORM/B3LYPcc_pVQZ/g16_inputFull_3q.out'
    assert os.path.isfile(result["files"]['log'])

    source_program = "cfour"
    mol_tuple = ("FORM", 'conf1', "CCSD(T)", "cc-pVQZ")

    result = dv.make_data_input_dict(source_program, mol_tuple)

    print("\nResult:", result)

    assert isinstance(result, dict), "Result should be a dictionary."
    assert "FORM" == result["files"]["mol_name"], "Molecule name should be in the result."
    assert result["files"]['out'] == EXAMPLES + '/refinedc4/CCSDTcc_pVQZ/out'
    for key in ['out', 'out_anharm_final', 'cubic', 'quartic', 'polar_pkl', 'molden']:
        assert os.path.isfile(result["files"][key]), key
    # 'dipolexyz' is the base name of the dipolex, dipoley, dipolez files
    for axis in 'xyz':
        assert os.path.isfile(result["files"]['dipolexyz'] + axis)
