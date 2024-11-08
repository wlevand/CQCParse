from parsing.parseGaussian_forWilson import *

def test_GaussianDataParser():

    datadict = {'log': './test_files_gaussian/g16_inputFull_3q.out',
                '3quanta': './test_files_gaussian/g16_inputFull_3q.out'}
    parserGaussian = GaussianDataParser({'files': datadict})
    parserGaussian.getData()

    essential = [parserGaussian.harmonic_states,
                 parserGaussian.anharmonic_states,
                 parserGaussian.fundamentals_harmonic_str,
                 parserGaussian.fundamentals_anharmonic_str,
                 parserGaussian.dipole_first_derivatives,
                 parserGaussian.dipole_second_derivatives,
                 parserGaussian.polarizability_first_derivatives,
                 parserGaussian.polarizability_second_derivatives,
                 parserGaussian.cubic_force_constants
                 ]

    assert all(v is not None for v in essential) and len(essential) == 9