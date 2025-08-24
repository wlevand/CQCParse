from CQCParse.relay.relay_data import DataVault

def test_DV_make_data_input_dict():
    """
    Test the make_data_input_dict method of the DataVault class.
    """
    dv = DataVault('/home/vlev/sprint/calculations/calculations.csv')
    
    source_program = "gaussian"
    mol_tuple = ("FORM", 'conf1', "B3LYP", "aug-cc-pVTZ")
    
    result = dv.make_data_input_dict(source_program, mol_tuple)
    
    print("\nResult:", result)

    assert isinstance(result, dict), "Result should be a dictionary."
    assert "FORM" == result["files"]["mol_name"], "Molecule name should be in the result."
    assert result["files"]['log'] == '/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/dftGaussian/FORM/B3LYPaug_cc_pVTZ/g16_inputFull_3q.out'

    source_program = "cfour"
    mol_tuple = ("FORM", 'conf1', "CCSD(T)", "cc-pVQZ")
    
    result = dv.make_data_input_dict(source_program, mol_tuple)
    
    print("\nResult:", result)

    assert isinstance(result, dict), "Result should be a dictionary."
    assert "FORM" == result["files"]["mol_name"], "Molecule name should be in the result."
    assert result["files"]['out'] == '/mnt/c/Users/vle014/OneDrive - UiT Office 365/Documents/files_fram/refinedc4/FORM/CCSDTcc_pVQZ/out'
