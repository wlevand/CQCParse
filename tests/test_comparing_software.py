from parsing.parseGaussian_forWilson import *
from parsing.parseCFOUR_forWilson import *
from wilson.relay import DataVault

def test_anharmonicHF_QZ_freqs():

    # datadict_cfour = {'out_anharm_final': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVDZ/out',
    #                   'dipolexyz': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVDZ/dipole',
    #                   'polar_pkl': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVDZ/polar.pkl',
    #                   'cubic': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVDZ/cubic'}
    # parserC4 = CFOURdataParser({'files':datadict_cfour})
    data_vault = DataVault()
    datadict_cfourFull = data_vault.make_DatainputDict('cfour', ('FORM', 'HF', 'cc_pVQZ'))

    parserC4 = CFOURdataParser(datadict_cfourFull)

    parserC4.getData()
    essential_CFOUR = [parserC4.harmonic_states,
                       parserC4.anharmonic_states,
                       parserC4.fundamentals_harmonic_str,
                       parserC4.fundamentals_anharmonic_str,
                       ]
    c4_harmFreqs = sorted(list(essential_CFOUR[2].values()))
    c4_anharmFreqs = sorted(list(essential_CFOUR[3].values()))
    print('\nCFOUR\nharmonic_states', c4_harmFreqs)
    print('anharmonic_states', c4_anharmFreqs)
    resonances_strs = get_detected_resonances_c4(datadict_cfourFull['files']['out_anharm_final'])

    print(''.join(resonances_strs)) if len(resonances_strs)>0 else None

    # datadict_gaussian = {'log': '/mnt/c/Users/vle014/Downloads/files_fram/dftGaussian/FORM/HFcc_pVDZ/g16_inputFull_3q.out',
    #             '3quanta': '/mnt/c/Users/vle014/Downloads/files_fram/dftGaussian/FORM/HFcc_pVDZ/g16_inputFull_3q.out'}
    # parserGaussian = GaussianDataParser({'files': datadict_gaussian})
    data_vault = DataVault()
    datadict_gaussianFull = data_vault.make_DatainputDict('gaussian', ('FORM', 'HF', 'cc_pVQZ'))
    parserGaussian = GaussianDataParser(datadict_gaussianFull)
    # print(datadict_gaussianFull)
    parserGaussian.getData()

    essential_gaussian = [parserGaussian.harmonic_states,
                          parserGaussian.anharmonic_states,
                          parserGaussian.fundamentals_harmonic_str,
                          parserGaussian.fundamentals_anharmonic_str,
                          ]
    g16_harmFreqs = sorted(list(essential_gaussian[2].values()))
    g16_anharmFreqs = sorted(list(essential_gaussian[3].values()))
    print('\nGaussian cc_pVDZ\nharmonic_states', g16_harmFreqs)
    print('anharmonic_states', g16_anharmFreqs)
    with np.printoptions(suppress=True):
        print('CFOUR-Gaussian', np.array(c4_anharmFreqs) - np.array(g16_anharmFreqs))
    resonances_strs = get_detected_resonances_g16(datadict_gaussianFull['files']['3quanta'])
    # print(resonances_strs)
    print(''.join(resonances_strs)) if len(resonances_strs)>0 else None

    assert g16_anharmFreqs != c4_anharmFreqs
    assert np.allclose(g16_harmFreqs, c4_harmFreqs, atol=10**(-4))


# def test_anharmonicHF_sto3g_freqs_dvpt2():
#
#     # datadict_cfour = {'out_anharm_final': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVQZ/out',
#     #                   'dipolexyz': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVQZ/dipole',
#     #                   'polar_pkl': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVQZ/polar.pkl',
#     #                   'cubic': '/home/vlew/scriptsHPC/data/cfourdata/hcoh/HFcc_pVQZ/cubic'}
#     # parserC4 = CFOURdataParser({'files':datadict_cfour})
#
#     data_vault = DataVault()
#     datadict_cfourFull = data_vault.make_DatainputDict('cfour', ('FORM', 'HF', 'STO_3G'))
#
#     parserC4 = CFOURdataParser(datadict_cfourFull)
#
#     parserC4.getData()
#     essential_CFOUR = [parserC4.harmonic_states,
#                        parserC4.anharmonic_states,
#                        parserC4.fundamentals_harmonic_str,
#                        parserC4.fundamentals_anharmonic_str,
#                        ]
#
#     c4_harmFreqs = sorted(list(essential_CFOUR[2].values()))
#     c4_anharmFreqs = sorted(list(essential_CFOUR[3].values()))
#     print('\nCFOUR\nharmonic freqs', c4_harmFreqs)
#     print('anharmonic freqs', c4_anharmFreqs)
#     resonances_strs = get_detected_resonances_c4(datadict_cfourFull['files']['out_anharm_final'])
#     print(''.join(resonances_strs)) if len(resonances_strs)>0 else None
#
#     # datadict_gaussian = {'log': '/mnt/c/Users/vle014/Downloads/files_fram/dftGaussian/FORM/HFcc_pVQZ/g16_inputFull_3q.out',
#     #             '3quanta': '/mnt/c/Users/vle014/Downloads/files_fram/dftGaussian/FORM/HFcc_pVQZ/g16_inputFull_3q.out'}
#     # parserGaussian = GaussianDataParser({'files': datadict_gaussian})
#
#     datadict_gaussianFull = data_vault.make_DatainputDict('gaussian', ('FORM', 'HF', 'STO_3G'))
#     parserGaussian = GaussianDataParser(datadict_gaussianFull)
#     parserGaussian.getData()
#
#     essential_gaussian = [parserGaussian.harmonic_states,
#                           parserGaussian.anharmonic_states,
#                           parserGaussian.fundamentals_harmonic_str,
#                           parserGaussian.fundamentals_anharmonic_str,
#                           ]
#     g16_harmFreqs = sorted(list(essential_gaussian[2].values()))
#     g16_anharmFreqs = sorted(list(essential_gaussian[3].values()))
#     print('\nGaussian STO_3G\nharmonic freqs', g16_harmFreqs)
#     print('anharmonic freqs', g16_anharmFreqs)
#     with np.printoptions(suppress=True):
#         print('CFOUR-Gaussian', np.array(c4_anharmFreqs) - np.array(g16_anharmFreqs))
#     resonances_strs = get_detected_resonances_g16(datadict_gaussianFull['files']['3quanta'])
#     print(''.join(resonances_strs)) if len(resonances_strs)>0 else None
#
#     datadict_gaussianFull = data_vault.make_DatainputDict('gaussian', ('FORM', 'HF', 'STO_3G_GVPT2'))
#     parserGaussian = GaussianDataParser(datadict_gaussianFull)
#
#     parserGaussian.getData()
#
#     essential_gaussian = [parserGaussian.harmonic_states,
#                           parserGaussian.anharmonic_states,
#                           parserGaussian.fundamentals_harmonic_str,
#                           parserGaussian.fundamentals_anharmonic_str,
#                           ]
#     g16_harmFreqs = sorted(list(essential_gaussian[2].values()))
#     g16_anharmFreqs = sorted(list(essential_gaussian[3].values()))
#     print('\nGaussian STO_3G GVPT2\nharmonic freqs', g16_harmFreqs)
#     print('anharmonic freqs', g16_anharmFreqs)
#     with np.printoptions(suppress=True):
#         print('CFOUR-Gaussian', np.array(c4_anharmFreqs) - np.array(g16_anharmFreqs))
#     resonances_strs = get_detected_resonances_g16(datadict_gaussianFull['files']['3quanta'])
#     print(''.join(resonances_strs)) if len(resonances_strs)>0 else None
#
#     datadict_gaussianFull = data_vault.make_DatainputDict('gaussian', ('FORM', 'HF', 'STO_3G_DVPT2'))
#     parserGaussian = GaussianDataParser(datadict_gaussianFull)
#     parserGaussian.getData()
#
#     essential_gaussian = [parserGaussian.harmonic_states,
#                           parserGaussian.anharmonic_states,
#                           parserGaussian.fundamentals_harmonic_str,
#                           parserGaussian.fundamentals_anharmonic_str,
#                           ]
#     g16_harmFreqs = sorted(list(essential_gaussian[2].values()))
#     g16_anharmFreqs = sorted(list(essential_gaussian[3].values()))
#     print('\nGaussian STO_3G_DVPT2\nharmonic freqs', g16_harmFreqs)
#     print('anharmonic freqs', g16_anharmFreqs)
#     with np.printoptions(suppress=True):
#         print('CFOUR-Gaussian', np.array(c4_anharmFreqs) - np.array(g16_anharmFreqs))
#     resonances_strs = get_detected_resonances_g16(datadict_gaussianFull['files']['3quanta'])
#     print(''.join(resonances_strs)) if len(resonances_strs)>0 else None
#
#     datadict_gaussianFull = data_vault.make_DatainputDict('gaussian', ('FORM', 'HF', 'STO_3G_VPT2'))
#     parserGaussian = GaussianDataParser(datadict_gaussianFull)
#
#     parserGaussian.getData()
#
#     essential_gaussian = [parserGaussian.harmonic_states,
#                           parserGaussian.anharmonic_states,
#                           parserGaussian.fundamentals_harmonic_str,
#                           parserGaussian.fundamentals_anharmonic_str,
#                           ]
#     g16_harmFreqs = sorted(list(essential_gaussian[2].values()))
#     g16_anharmFreqs = sorted(list(essential_gaussian[3].values()))
#     print('\nGaussian STO_3G_VPT2\nharmonic freqs', g16_harmFreqs)
#     print('anharmonic freqs', g16_anharmFreqs)
#     with np.printoptions(suppress=True):
#         print('CFOUR-Gaussian', np.array(c4_anharmFreqs) - np.array(g16_anharmFreqs))
#     resonances_strs = get_detected_resonances_g16(datadict_gaussianFull['files']['3quanta'])
#     print(''.join(resonances_strs)) if len(resonances_strs)>0 else None
#
#     datadict_gaussianFull = data_vault.make_DatainputDict('gaussian', ('FORM', 'HF', 'STO_3G_GVPT2_Resonances'))
#     parserGaussian = GaussianDataParser(datadict_gaussianFull)
#     parserGaussian.getData()
#
#     essential_gaussian = [parserGaussian.harmonic_states,
#                           parserGaussian.anharmonic_states,
#                           parserGaussian.fundamentals_harmonic_str,
#                           parserGaussian.fundamentals_anharmonic_str,
#                           ]
#     g16_harmFreqs = sorted(list(essential_gaussian[2].values()))
#     g16_anharmFreqs = sorted(list(essential_gaussian[3].values()))
#     print('\nGaussian STO_3G_GVPT2_Resonances\nharmonic freqs', g16_harmFreqs)
#     print('anharmonic freqs', g16_anharmFreqs)
#     with np.printoptions(suppress=True):
#         print('CFOUR-Gaussian', np.array(c4_anharmFreqs) - np.array(g16_anharmFreqs))
#     resonances_strs = get_detected_resonances_g16(datadict_gaussianFull['files']['3quanta'])
#     print(''.join(resonances_strs)) if len(resonances_strs)>0 else None
#
#     datadict_gaussianFull = data_vault.make_DatainputDict('gaussian', ('FORM', 'HF', 'STO_3G_DVPT2_Resonances'))
#     parserGaussian = GaussianDataParser(datadict_gaussianFull)
#     parserGaussian.getData()
#
#     essential_gaussian = [parserGaussian.harmonic_states,
#                           parserGaussian.anharmonic_states,
#                           parserGaussian.fundamentals_harmonic_str,
#                           parserGaussian.fundamentals_anharmonic_str,
#                           ]
#     g16_harmFreqs = sorted(list(essential_gaussian[2].values()))
#     g16_anharmFreqs = sorted(list(essential_gaussian[3].values()))
#     print('\nGaussian STO_3G_DVPT2_Resonances\nharmonic freqs', g16_harmFreqs)
#     print('anharmonic freqs', g16_anharmFreqs)
#     with np.printoptions(suppress=True):
#         print('CFOUR-Gaussian', np.array(c4_anharmFreqs) - np.array(g16_anharmFreqs))
#     resonances_strs = get_detected_resonances_g16(datadict_gaussianFull['files']['3quanta'])
#     print(''.join(resonances_strs)) if len(resonances_strs)>0 else None
#
#     assert g16_anharmFreqs != c4_anharmFreqs
#     assert np.allclose(g16_harmFreqs, c4_harmFreqs, atol=10**(-4))


def test_anharmonicHF_CFF():
    pass

def test_anharmonicHF_dipoleF():
    pass

def test_anharmonicHF_dipoleS():
    pass

