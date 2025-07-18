from parsing.parseGaussian_forWilson import parse_coriolis
from parsing.gaussian_parser import GaussianParser, GaussianOutput
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