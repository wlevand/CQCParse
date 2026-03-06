from CQCParse.parsing.gaussian_parser import GaussianParser, GaussianOutput
from CQCParse.parsing.parseGaussian_forWilson import parse_coriolis
from CQCParse import debug

def test_GaussianDataParser():
    debug.level = 0

    log_file = '/home/vlev/Wilson/tests/test_database/dftGaussian/ACAC/B3LYPcc-pVQZ/g16_inputFull_3q.out'

    parserGaussian = GaussianParser(GaussianOutput('ACAC', 'B3LYP', 'cc_pVQZ', 'gaussian', log_file))
    print('\n', [i for i in dir(parserGaussian) if not i.startswith('_')])
    #  ['getDerivatives', 'getNormModes', 'getPreVPT2Data', 'getStructure', 'getVibStates',
    #  'load', 'natoms', 'nmodes', 'parse', 'relevant_files']

    parserGaussian.load()
    pdata = parserGaussian.parse()
    print(pdata)
    # <ParsedData: dict_keys(['molecule', 'program', 'basis', 'method', 'structure', 'nmodes', 'vib_states',
    # 'derivatives', 'normal_modes', 'anharm_correction_data', 'anharm_treatment', 'list2exclude'])

    assert pdata.check_if_have_data()

    # essential = [parserGaussian.harmonic_states,
    #              parserGaussian.anharmonic_states,
    #              parserGaussian.fundamentals_harmonic_str,
    #              parserGaussian.fundamentals_anharmonic_str,
    #              parserGaussian.dipgrad,
    #              parserGaussian.diphess,
    #              parserGaussian.polgrad,
    #              parserGaussian.polhess,
    #              parserGaussian.cff
    #              ]
    #
    # assert all(v is not None for v in essential)


def test_parse_coriolis():
    # fixme: copy this file to CQCParse repo
    log_file = '/home/vlev/Wilson/tests/test_database/dftGaussian/ACAC/B3LYPcc-pVQZ/g16_inputFull_3q.out'
    with open(log_file, 'r') as file:
        log_lines = [i.strip() for i in file.readlines()]
    nmodes = 18

    rotational_constant, coriolis_constant = parse_coriolis(log_lines, nmodes)

    assert len(rotational_constant) == 3

def test_parse_mode_mapping():
    from CQCParse.parsing.parseGaussian_forWilson import parse_mode_mapping
    log_file = '/home/vlev/wilson-suite/CQCParse/CQCParse/files_examples/dftGaussian/FORM/B3LYPcc_pVDZ/g16_inputFull_3q.out'
    with open(log_file, 'r') as file:
        log_lines = [i.strip() for i in file.readlines()]
    
    r = parse_mode_mapping(log_lines)
    print(r)
    assert r == {4: 1, 6: 2, 3: 3, 2: 4, 1: 5, 5: 6}

    lfile = '/home/vlev/wilson-suite/CQCParse/CQCParse/files_examples/dftGaussian/ACDM/B3LYPcc_pVQZ/g16_b3lyp_corr_inputFull_3q.out'

    with open(lfile, 'r') as file:
        log_lines_1 = [i.strip() for i in file.readlines()]
    
    r = parse_mode_mapping(log_lines_1)
    print(r)
    assert r == {42: 1, 41: 2, 40: 3, 39: 4, 38: 5, 37: 6, 36: 7, 
                 35: 8, 34: 9, 33: 10, 32: 11, 31: 12, 30: 13, 
                 29: 14, 28: 15, 27: 16, 26: 17, 25: 18, 24: 19, 
                 23: 20, 22: 21, 21: 22, 20: 23, 19: 24, 18: 25, 17: 26, 16: 27, 15: 28, 
                 14: 29, 13: 30, 12: 31, 11: 32, 10: 33, 9: 34, 
                 8: 35, 7: 36, 6: 37, 5: 38, 4: 39, 3: 40, 2: 41, 1: 42}