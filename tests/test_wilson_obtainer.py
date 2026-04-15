from CQCParse.parsing.parse_wilson_obtainer import parse_cfour_output, parse_gaussian16_output, parse_from_source

def test_parse_from_source_cfour():
    print()
    complete_data = ['cff', 'anharmonic_states', 'nc_sqrt_eigval', 'dipgrad', 'B', 'polgrad', 'coriolis', 'polhess', 'qff', 'diphess']
    rq_data = dict.fromkeys(complete_data, None)
    base_file_loc = 'CQCParse/files_examples/refinedc4/CCSDTcc_pVQZ'
    r1 = parse_from_source(source_type='cfour', requested_data=rq_data, base_file_loc=base_file_loc, linear=False)

    # print(r1)

# def test_parse_from_source_g16():
    print()
    complete_data = ['cff', 'anharmonic_states', 'nc_sqrt_eigval', 'dipgrad', 'B', 'polgrad', 'coriolis', 'polhess', 'qff', 'diphess']
    rq_data = dict.fromkeys(complete_data, None)
    base_file_loc = 'CQCParse/files_examples/dftGaussian/FORM/B3LYPcc_pVDZ/g16_inputFull_3q.out'
    r2 = parse_from_source(source_type='gaussian', requested_data=rq_data, base_file_loc=base_file_loc, linear=False)

    # print(r2)

    import numpy as np
    for k in r1:
        assert type(r1[k]) is type(r2[k])
        if type(r1[k]) is np.ndarray:
            assert r1[k].shape == r2[k].shape
        if type(r1[k]) is dict:
            assert set(r1[k].keys()) == set(r2[k].keys())