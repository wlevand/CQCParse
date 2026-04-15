from CQCParse.parsing.simple_parsed_data import parse_cfour_output, parse_gaussian16_output
from CQCParse.utils import PKG_ROOT
import numpy as np

def test_parse_gaussian16_output():
    molecule, level_of_theory, basis = 'FORM', 'B3LYP', 'cc-pVDZ'
    log_file = PKG_ROOT + '/CQCParse/files_examples/dftGaussian/FORM/B3LYPcc_pVDZ/g16_inputFull_3q.out'
    parsed_data = parse_gaussian16_output(molecule=molecule, 
                                          level_of_theory=level_of_theory, 
                                          basis=basis,
                                          log_file=log_file)
    
    assert parsed_data.atoms == ['O', 'C', 'H', 'H']
    assert len(parsed_data.normal_modes) == 6
    assert parsed_data.hess is None
    
    assert np.all(parsed_data.B == np.array([1.134063, 1.2903538, 9.3629432]))
    assert parsed_data.coriolis.shape == (3, 6, 6)

    assert parsed_data.dipgrad.shape == (6, 3)       # g f
    assert parsed_data.diphess.shape == (6, 6, 3)    # g g f
    assert parsed_data.polgrad.shape == (6, 3, 3)    # g f f
    assert parsed_data.polhess.shape == (6, 6, 3, 3) # g g f f
    assert parsed_data.cff.shape == (6, 6, 6)        # g g g
    assert parsed_data.qff.shape == (6, 6, 6, 6)     # g g g g
    assert parsed_data.units == {'harmonic_states': 'cm-1', 'anharmonic_states': 'cm-1', 
                                 'equilibrium_geometry': 'Angstrom', 'normal_modes': 'Angstrom', 
                                 'hess': '', 'B': 'cm-1', 'coriolis': 'dimensionless', 
                                 'dipgrad': 'a.u.', 'diphess': 'a.u.', 
                                 'polgrad': 'a.u.', 'polhess': 'a.u.', 
                                 'cff': 'cm-1', 'qff': 'cm-1', 
                                 'cff_au': 'Ha*m_e(-3/2)*a0(-3)', 'qff_au': 'Ha*m_e(-2)*a0(-4)'}

def test_parse_cfour_output():
    molecule, level_of_theory, basis = 'FORM', 'B3LYP', 'cc-pVDZ'
    files_dict = {'out_file' : PKG_ROOT + '/CQCParse/files_examples/refinedc4/CCSDTcc_pVQZ/out', 
                  'molden' : PKG_ROOT + '/CQCParse/files_examples/refinedc4/CCSDTcc_pVQZ/MOLDEN',
                  'cubic_file': PKG_ROOT + '/CQCParse/files_examples/refinedc4/CCSDTcc_pVQZ/cubic',
                  'quartic_file': PKG_ROOT + '/CQCParse/files_examples/refinedc4/CCSDTcc_pVQZ/quartic',
                  'dipole_file': PKG_ROOT + '/CQCParse/files_examples/refinedc4/CCSDTcc_pVQZ/dipole',
                  'polar_pkl': PKG_ROOT + '/CQCParse/files_examples/refinedc4/CCSDTcc_pVQZ/polar.pkl'}
    parsed_data = parse_cfour_output(molecule=molecule, 
                                     level_of_theory=level_of_theory, 
                                     basis=basis,
                                     files_dict=files_dict,
                                     linear_molecule=False)
    
    assert list(parsed_data.atoms) == ['O', 'C', 'H', 'H']
    assert len(parsed_data.normal_modes) == 6
    assert parsed_data.hess is None

    assert np.all(parsed_data.B == np.array([1.14505170600296, 1.30063729088682, 9.57220394121177]))
    assert parsed_data.coriolis.shape == (3, 6, 6)

    assert parsed_data.dipgrad.shape == (6, 3)       # g f
    assert parsed_data.diphess.shape == (6, 6, 3)    # g g f
    assert parsed_data.polgrad.shape == (6, 3, 3)    # g f f
    assert parsed_data.polhess.shape == (6, 6, 3, 3) # g g f f
    assert parsed_data.cff.shape == (6, 6, 6)        # g g g
    assert parsed_data.qff.shape == (6, 6, 6, 6)     # g g g g
    assert parsed_data.units == {'harmonic_states': 'cm-1', 'anharmonic_states': 'cm-1', 
                                 'equilibrium_geometry': 'Bohr', 'normal_modes': 'Bohr', 
                                 'hess': '', 'B': 'cm-1', 'coriolis': 'dimensionless', 
                                 'dipgrad': 'a.u.', 'diphess': 'a.u.', 
                                 'polgrad': 'a.u.', 'polhess': 'a.u.', 
                                 'cff': 'cm-1', 'qff': 'cm-1', 
                                 'cff_au': 'Ha*m_e(-3/2)*a0(-3)', 'qff_au': 'Ha*m_e(-2)*a0(-4)'}