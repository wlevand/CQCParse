import numpy as np
import pytest
from CQCParse.parsing.parseCFOUR_forWilson import pMOLDEN, pQUADRATURE, parse_output_file
from CQCParse.utils import PKG_ROOT

# formaldehyde, CFOUR VPT2 runs
C4_HF = PKG_ROOT + '/CQCParse/files_examples/refinedc4/HFcc_pVDZ'
C4_CCSDT = PKG_ROOT + '/CQCParse/files_examples/refinedc4/CCSDTcc_pVQZ'


def read_lines(path):
    with open(path, 'r') as file:
        return file.readlines()

def test_pMOLDEN_geometry_data():
    geometry_data, atoms, normal_modes_dict = pMOLDEN(C4_HF + '/MOLDEN')
    ref_geometry_data = np.array( [[-0.0000000000   ,    0.0000000000  ,     1.1177168336],
                                   [-0.0000000000   ,    0.0000000000  ,    -1.1160727840],
                                   [-0.0000000000   ,    1.7621743928  ,    -2.2250449092],
                                   [ 0.0000000000   ,   -1.7621743928  ,    -2.2250449092]])
    assert np.all(geometry_data == ref_geometry_data)

def test_pMOLDEN_atoms():
    geometry_data, atoms, normal_modes_dict = pMOLDEN(C4_HF + '/MOLDEN')
    ref_atoms = np.array(['O', 'C', 'H', 'H'])
    assert np.all(atoms == ref_atoms)

def test_pMOLDEN_normal_modes_dict():
    geometry_data, atoms, normal_modes_dict = pMOLDEN(C4_HF + '/MOLDEN')
    ref_normal_modes_dict10 = np.array([[ 0.0000000000 , -0.0000000000 ,  0.1554554154 ],
                                        [ 0.0000000000 ,  0.0000000000 , -0.2153090842 ],
                                        [ 0.0000000000 ,  0.1613517401 ,  0.0482290617 ],
                                        [-0.0000000000 , -0.1613517401 ,  0.0482290617 ]])
    ref_normal_modes_dict12 = np.array([[ -0.0000000000 ,  0.0005198582 , -0.0000000000 ],
                                        [  0.0000000000 , -0.0955724792 ,  0.0000000000 ],
                                        [ -0.0000000000 ,  0.5648573035 , -0.3502468672 ],
                                        [ -0.0000000000 ,  0.5648573035 ,  0.3502468672 ]])
    ref_normal_modes_dict7 = np.array([[  0.0371470723 , -0.0000000000 ,  0.0000000000 ],
                                       [ -0.1492485822 ,  0.0000000000 ,  0.0000000000 ],
                                       [  0.5937631518 , -0.0000000000 , -0.0000000000 ],
                                       [  0.5937631518 , -0.0000000000 ,  0.0000000000 ]])
    dictkeys = list(normal_modes_dict.keys())

    assert np.all(normal_modes_dict[10] == ref_normal_modes_dict10)
    assert np.all(normal_modes_dict[dictkeys[0]] == ref_normal_modes_dict7)
    assert np.all(normal_modes_dict[dictkeys[-1]] == ref_normal_modes_dict12)

def test_pQUADRATURE_geo():
    equilibrium_geometry, freqs, normal_coordinates = pQUADRATURE(read_lines(C4_CCSDT + '/QUADRATURE'))
    ref_equil = np.array([[ -0.0000000000 ,  0.0000000000 ,  1.1362646308],
                          [ -0.0000000000 ,  0.0000000000 , -1.1393542951],
                          [ -0.0000000000 ,  1.7663776970 , -2.2336239272],
                          [  0.0000000000 , -1.7663776970 , -2.2336239272],])
    assert np.all(equilibrium_geometry == ref_equil)

def test_pQUADRATURE_freqs():
    equilibrium_geometry, freqs, normal_coordinates = pQUADRATURE(read_lines(C4_CCSDT + '/QUADRATURE'))
    ref_freqs = np.array([ 1195.5150809715, 1278.4486784657, 1544.4781825563,
                           1791.3068455702, 2944.8223449746, 3014.5922182459])
    assert np.all(freqs == ref_freqs)

def test_pQUADRATURE_norm_coordinates():
    equilibrium_geometry, freqs, normal_coordinates = pQUADRATURE(read_lines(C4_CCSDT + '/QUADRATURE'))

    ref_normal_coordinates10 = np.array([[-0.0000000000  ,  0.0000000000 ,   0.0384798709],
                                         [ 0.0000000000  , -0.0000000000  , -0.0559724369],
                                         [-0.0000000000  ,  0.0465529529  ,  0.0278753707],
                                         [-0.0000000000  , -0.0465529529  ,  0.0278753707]])
    ref_normal_coordinates7 = np.array([[ 0.0114580254 , -0.0000000000 , -0.0000000000],
                                        [-0.0470329112 , -0.0000000000 , -0.0000000000],
                                        [ 0.1890828185 , -0.0000000000 , -0.0000000000],
                                        [ 0.1890828185 , -0.0000000000 ,  0.0000000000]])
    ref_normal_coordinates12 = np.array([[-0.0000000000 ,  0.0001147516 ,  0.0000000000],
                                         [ 0.0000000000 , -0.0191951344 , -0.0000000000],
                                         [-0.0000000000 ,  0.1133659922 , -0.0690570301],
                                         [-0.0000000000 ,  0.1133659922 ,  0.0690570301]])
    dictkeys = list(normal_coordinates.keys())

    assert np.all(normal_coordinates[10] == ref_normal_coordinates10)
    assert np.all(normal_coordinates[dictkeys[0]] == ref_normal_coordinates7)
    assert np.all(normal_coordinates[dictkeys[-1]] == ref_normal_coordinates12)

def test_parse_output_file():
    # outfile0.out has no levels table: an error (todo_cleanup Q12; it printed a line and returned None)
    with pytest.raises(ValueError, match='All levels with up to three quanta'):
        parse_output_file(C4_HF + '/outfile0.out')
    out = parse_output_file(C4_HF + '/out')
    assert isinstance(out, tuple)

@pytest.fixture
def mock_output_file(tmp_path):
    d = tmp_path / "sub"
    d.mkdir()
    p = d / "mock_output.txt"
    content = """
 ------------------------------------------------------------------------------
                   All levels with up to three quanta
--------------------------------------------------------------------------------
  MODE MODE MODE MODE MODE                   Anharmonic    Anharm      Harmonic
   I    J    K    L    M   NI  NJ  NK  NL  NM Frequency   Intensity   Transition
--------------------------------------------------------------------------------
7 0 0 0 0 1 0 0 0 0 1307.4 1.26 1325.3
7 0 0 0 0 2 0 0 0 0 2606.5 0.02 2650.6
8 7 0 0 0 1 1 0 0 0 2657.3 0.00 2685.0
--------------------------------------------------------------------------------
 ----------------------------------------------------------------------
     Electric dipole moment function in dimensionless normal coordinates
"""
    p.write_text(content)
    return p

def test_parse_output_file_normal_case_modes(mock_output_file):
    modes, labels, anharmonic_frequencies, anharmonic_intensities, harmonic_transitions = parse_output_file(
        str(mock_output_file))
    assert modes == [(7,), (7, 7), (7, 8)]

def test_parse_output_file_normal_case_anharmonic_frequencies(mock_output_file):
    modes, labels, anharmonic_frequencies, anharmonic_intensities, harmonic_transitions = parse_output_file(
        str(mock_output_file))
    assert np.allclose(anharmonic_frequencies, [1307.4, 2606.5, 2657.3])

def test_parse_output_file_normal_case_harmonic_transitions(mock_output_file):
    modes, labels, anharmonic_frequencies, anharmonic_intensities, harmonic_transitions = parse_output_file(
        str(mock_output_file))
    assert np.allclose(harmonic_transitions, np.array([1325.3, 2650.6, 2685.0]))