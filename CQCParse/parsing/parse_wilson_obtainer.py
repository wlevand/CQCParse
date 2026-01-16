from typing import Any

def parse_from_source(requested_data: dict, 
                      source_type: str, 
                      base_file_loc: str,
                      lvl_theory: str = '',
                      basis_set: str = '') -> dict[str, Any]:
    """
    VALUES MUST BE IN ATOMIC UNITS
    """
    if source_type == 'gaussian':
        results_dict = parse_gaussian16_output(requested_data=requested_data, 
                                               log_file=base_file_loc)
        return results_dict
    
    elif source_type == 'cfour':
        files_dict = make_cfour_files_dict(base_file_loc)
        # todo: finish parse_cfour_output
        results_dict = parse_cfour_output(requested_data=requested_data,
                                          base_file_loc=files_dict)
        return results_dict



def parse_gaussian16_output(requested_data: list,
                            log_file: str):
    """
    
    Units:
        "cff_au": "Ha*m_e(-3/2)*a0(-3)", # a.u.
        "qff_au": "Ha*m_e(-2)*a0(-4)", # a.u.
    """
    results = {}
    
    with open(log_file, 'r') as file:
        log_lines = [i.strip() for i in file.readlines()]

    if 'equilibrium_geometry' or 'atoms' in requested_data:
        from .parseGaussian_forWilson import get_equil_geo
        try:
            atoms, equilibrium_geometry = get_equil_geo(log_lines)
            
            if 'equilibrium_geometry' in requested_data:
                results['equilibrium_geometry'] = equilibrium_geometry
            
            if 'atoms' in requested_data:
                results['atoms'] = atoms
        except Exception as e:
            print('Failed at "equilibrium_geometry" with:', e)
            raise ValueError("smth went wrong")

    if 'normal_modes' in requested_data:
        from .parseGaussian_forWilson import get_normal_modes
        try:
            nmodes = get_normal_modes(filename=log_file, Na=len(atoms))
            nmodes = {i: nm for i, nm in enumerate(nmodes)}
            results['normal_modes'] = nmodes
            len_nmodes = len(nmodes)
        except Exception as e:
            print('Failed at "normal_modes" with:', e)
            raise ValueError("smth went wrong")
    
    if 'anharmonic_states' or 'harmonic_states' or 'nc_sqrt_eigval' in requested_data:
        from .parseGaussian_forWilson import parse_frequencies
        try:
            # results_log = parse_frequencies(lines=log_lines)
            """
results {                       'Fundamental Bands': [['Fundamental', 'Bands'], ['Mode(n)', 'E(harm)', 'E(anharm)', 'I(harm)', 'I(anharm)'], 
                                ['1(1)', '2878.687', '2726.813', '70.98779730', '71.80014071'], 
                                ['2(1)', '1820.416', '1794.540', '110.62982992', '115.32220025'], 
                                ['3(1)', '1534.549', '1501.586', '10.90752988', '10.05151936'], 
                                ['4(1)', '1203.179', '1185.288', '3.71379143', '3.71224363'], 
                                ['5(1)', '2933.526', '2682.765', '129.45835494', '65.82735164'], 
                                ['6(1)', '1268.910', '1247.878', '12.36061629', '13.28335792'], []], 
                                'Overtones': [['Overtones'], ['---------'], ['Mode(n)', 'E(harm)', 'E(anharm)', 'I(anharm)'], 
                                ['1(2)', '5757.374', '5390.225', '0.04231731'], 
                                ['1(3)', '8636.061', '7990.237', '0.00517985'], 
                                ['2(2)', '3640.831', '3570.468', '4.43224602'], 
                                ['2(3)', '5461.247', '5327.784', '0.13867576'], 
                                ['3(2)', '3069.097', '3004.865', '3.19688971'], 
                                ['3(3)', '4603.646', '4509.837', '0.00593731'], 
                                ['4(2)', '2406.357', '2364.564', '0.06974080'], 
                                ['4(3)', '3609.536', '3537.828', '0.00200336'], 
                                ['5(2)', '5867.051', '5428.690', '1.34550308'], 
                                ['5(3)', '8800.577', '8032.189', '0.13211266'], 
                                ['6(2)', '2537.819', '2491.481', '0.56254170'], 
                                ['6(3)', '3806.729', '3730.809', '0.00769807'], []], 
                                'Combination Bands': [['Combination', 'Bands'], ['Mode(n)', 'E(harm)', 'E(anharm)', 'I(anharm)'], 
                                ['2(1)', '1(1)', '4699.103', '4522.216', '1.77770880'], 
                                ['2(2)', '1(1)', '6519.518', '6299.008', '0.01191159'], 
                                ['2(1)', '1(2)', '7577.790', '7186.492', '0.02698984'], 
                                ['3(1)', '1(1)', '4413.236', '4197.002', '0.00001461'], 
                                ['3(2)', '1(1)', '5947.784', '5668.884', '0.00346832'], 
                                ['3(1)', '1(2)', '7291.923', '6829.018', '0.00011976'], 
                                ['3(1)', '2(1)', '3354.964', '3290.380', '0.20557736'], 
                                ['3(2)', '2(1)', '4889.513', '4787.912', '0.03250538'], 
                                ['3(1)', '2(2)', '5175.380', '5060.561', '0.00085072'], 
                                ['3(1)', '2(1)', '1(1)', '6233.651', '5986.659', '0.00265168'], 
                                ['4(1)', '1(1)', '4081.866', '3905.476', '0.02613521'], 
                                ['4(2)', '1(1)', '5285.044', '5078.128', '0.00688096'], 
                                ['4(1)', '1(2)', '6960.553', '6562.264', '0.00039474'], 
                                ['4(1)', '2(1)', '3023.594', '2974.691', '0.51852681'], 
                                ['4(2)', '2(1)', '4226.773', '4148.830', '0.05744097'], 
                                ['4(1)', '2(2)', '4844.010', '4745.482', '0.01025787'], 
                                ['4(1)', '2(1)', '1(1)', '5902.281', '5695.743', '0.00053949'], 
                                ['4(1)', '3(1)', '2737.727', '2687.219', '0.00997136'], 
                                ['4(2)', '3(1)', '3940.906', '3866.839', '0.00177178'], 
                                ['4(1)', '3(2)', '4272.276', '4190.842', '0.00026269'], 
                                ['4(1)', '3(1)', '1(1)', '5616.414', '5376.010', '0.00012415'], 
                                ['4(1)', '3(1)', '2(1)', '4558.143', '4470.875', '0.00000044'], 
                                ['5(1)', '1(1)', '5812.213', '5345.290', '0.45308340'], 
                                ['5(2)', '1(1)', '8745.739', '7889.869', '0.14480704'], 
                                ['5(1)', '1(2)', '8690.900', '7875.885', '0.09644865'], 
                                ['5(1)', '2(1)', '4753.941', '4547.872', '5.08112938'], 
                                ['5(2)', '2(1)', '7687.467', '7227.306', '0.00157189'], 
                                ['5(1)', '2(2)', '6574.357', '6325.838', '0.04603463'], 
                                ['5(1)', '2(1)', '1(1)', '7632.628', '7142.731', '0.13818376'], 
                                ['5(1)', '3(1)', '4468.074', '4217.292', '0.52827583'], 
                                ['5(2)', '3(1)', '7401.600', '6859.101', '0.01746104'], 
                                ['5(1)', '3(2)', '6002.623', '5684.984', '0.00637531'], 
                                ['5(1)', '3(1)', '1(1)', '7346.761', '6779.891', '0.02177663'], 
                                ['5(1)', '3(1)', '2(1)', '6288.490', '6008.124', '0.02233402'], 
                                ['5(1)', '4(1)', '4136.704', '3916.796', '0.00000000'], 
                                ['5(2)', '4(1)', '7070.230', '6574.407', '0.00284106'], 
                                ['5(1)', '4(2)', '5339.883', '5076.287', '0.01821645'], 
                                ['5(1)', '4(1)', '1(1)', '7015.391', '6504.168', '0.00000000'], 
                                ['5(1)', '4(1)', '2(1)', '5957.120', '5708.237', '0.00000000'], 
                                ['5(1)', '4(1)', '3(1)', '5671.253', '5383.140', '0.00000000'], 
                                ['6(1)', '1(1)', '4147.597', '3967.717', '0.32491526'], 
                                ['6(2)', '1(1)', '5416.506', '5204.346', '0.00060032'], 
                                ['6(1)', '1(2)', '7026.284', '6624.156', '0.00113360'], 
                                ['6(1)', '2(1)', '3089.325', '3043.988', '4.22440293'], 
                                ['6(2)', '2(1)', '4358.235', '4272.935', '0.05263963'], 
                                ['6(1)', '2(2)', '4909.741', '4805.260', '0.00000044'], 
                                ['6(1)', '2(1)', '1(1)', '5968.012', '5756.577', '0.00000001'], 
                                ['6(1)', '3(1)', '2803.458', '2812.955', '80.68363077'], 
                                ['6(2)', '3(1)', '4072.368', '3999.219', '0.03501458'], 
                                ['6(1)', '3(2)', '4338.007', '4258.895', '0.00221844'], 
                                ['6(1)', '3(1)', '1(1)', '5682.145', '5440.983', '0.00060329'], 
                                ['6(1)', '3(1)', '2(1)', '4623.874', '4534.790', '0.00175475'], 
                                ['6(1)', '4(1)', '2472.088', '2440.610', '0.00000000'], 
                                ['6(2)', '4(1)', '3740.998', '3691.657', '0.00071089'], 
                                ['6(1)', '4(2)', '3675.267', '3627.330', '0.00814692'], 
                                ['6(1)', '4(1)', '1(1)', '5350.775', '5153.825', '0.00000000'], 
                                ['6(1)', '4(1)', '2(1)', '4292.504', '4223.470', '0.00000000'], 
                                ['6(1)', '4(1)', '3(1)', '4006.637', '3945.617', '0.00000000'], 
                                ['6(1)', '5(1)', '4202.435', '3968.654', '0.01871179'], 
                                ['6(2)', '5(1)', '5471.345', '5181.740', '0.00219606'], 
                                ['6(1)', '5(2)', '7135.961', '6615.533', '0.01389379'], 
                                ['6(1)', '5(1)', '1(1)', '7081.122', '6555.676', '0.00215602'], 
                                ['6(1)', '5(1)', '2(1)', '6022.851', '5758.689', '0.00076461'], 
                                ['6(1)', '5(1)', '3(1)', '5736.984', '5437.729', '0.00078543'], 
                                ['6(1)', '5(1)', '4(1)', '5405.614', '5141.601', '0.00031556'], []]}
            """
            # results_log = parse_frequencies_v2(lines=log_lines)
            from .parse_g16_freqs import parse_frequencies, get_allStates_from_parsed_freqs
            results_log = parse_frequencies(lines=log_lines)
            h_sts, ah_sts = get_allStates_from_parsed_freqs(results_log)
            """
{'Fundamental Bands': [((1,), 2878.687, 2726.813), ((2,), 1820.416, 1794.54), ((3,), 1534.549, 1501.586), ((4,), 1203.179, 1185.288), ((5,), 2933.526, 2682.765), ((6,), 1268.91, 1247.878)], 
'Overtones': [((1, 1), 5757.374, 5390.225), ((1, 1, 1), 8636.061, 7990.237), ((2, 2), 3640.831, 3570.468), ((2, 2, 2), 5461.247, 5327.784), ((3, 3), 3069.097, 3004.865), ((3, 3, 3), 4603.646, 4509.837), ((4, 4), 2406.357, 2364.564), ((4, 4, 4), 3609.536, 3537.828), ((5, 5), 5867.051, 5428.69), ((5, 5, 5), 8800.577, 8032.189), ((6, 6), 2537.819, 2491.481), ((6, 6, 6), 3806.729, 3730.809)], 
'Combination Bands': [((2, 1), 4699.103, 4522.216), ((2, 2, 1), 6519.518, 6299.008), ((2, 1, 1), 7577.79, 7186.492), ((3, 1), 4413.236, 4197.002), ((3, 3, 1), 5947.784, 5668.884), ((3, 1, 1), 7291.923, 6829.018), ((3, 2), 3354.964, 3290.38), ((3, 3, 2), 4889.513, 4787.912), ((3, 2, 2), 5175.38, 5060.561), ((3, 2, 1), 6233.651, 5986.659), ((4, 1), 4081.866, 3905.476), ((4, 4, 1), 5285.044, 5078.128), ((4, 1, 1), 6960.553, 6562.264), ((4, 2), 3023.594, 2974.691), ((4, 4, 2), 4226.773, 4148.83), ((4, 2, 2), 4844.01, 4745.482), ((4, 2, 1), 5902.281, 5695.743), ((4, 3), 2737.727, 2687.219), ((4, 4, 3), 3940.906, 3866.839), ((4, 3, 3), 4272.276, 4190.842), ((4, 3, 1), 5616.414, 5376.01), ((4, 3, 2), 4558.143, 4470.875), ((5, 1), 5812.213, 5345.29), ((5, 5, 1), 8745.739, 7889.869), ((5, 1, 1), 8690.9, 7875.885), ((5, 2), 4753.941, 4547.872), ((5, 5, 2), 7687.467, 7227.306), ((5, 2, 2), 6574.357, 6325.838), ((5, 2, 1), 7632.628, 7142.731), ((5, 3), 4468.074, 4217.292), ((5, 5, 3), 7401.6, 6859.101), ((5, 3, 3), 6002.623, 5684.984), ((5, 3, 1), 7346.761, 6779.891), ((5, 3, 2), 6288.49, 6008.124), ((5, 4), 4136.704, 3916.796), ((5, 5, 4), 7070.23, 6574.407), ((5, 4, 4), 5339.883, 5076.287), ((5, 4, 1), 7015.391, 6504.168), ((5, 4, 2), 5957.12, 5708.237), ((5, 4, 3), 5671.253, 5383.14), ((6, 1), 4147.597, 3967.717), ((6, 6, 1), 5416.506, 5204.346), ((6, 1, 1), 7026.284, 6624.156), ((6, 2), 3089.325, 3043.988), ((6, 6, 2), 4358.235, 4272.935), ((6, 2, 2), 4909.741, 4805.26), ((6, 2, 1), 5968.012, 5756.577), ((6, 3), 2803.458, 2812.955), ((6, 6, 3), 4072.368, 3999.219), ((6, 3, 3), 4338.007, 4258.895), ((6, 3, 1), 5682.145, 5440.983), ((6, 3, 2), 4623.874, 4534.79), ((6, 4), 2472.088, 2440.61), ((6, 6, 4), 3740.998, 3691.657), ((6, 4, 4), 3675.267, 3627.33), ((6, 4, 1), 5350.775, 5153.825), ((6, 4, 2), 4292.504, 4223.47), ((6, 4, 3), 4006.637, 3945.617), ((6, 5), 4202.435, 3968.654), ((6, 6, 5), 5471.345, 5181.74), ((6, 5, 5), 7135.961, 6615.533), ((6, 5, 1), 7081.122, 6555.676), ((6, 5, 2), 6022.851, 5758.689), ((6, 5, 3), 5736.984, 5437.729), ((6, 5, 4), 5405.614, 5141.601)]}
            """
            
            if not results_log:
                raise ValueError("result of parse_frequencies() is empty")
        except Exception as e:
            print('Failed at "parse_frequencies" with:', e)
            raise ValueError("smth went wrong in parse_frequencies() or get_allStates_from_parsed_freqs()")

        if 'anharmonic_states' in requested_data:
            try:
                # print('\nresults_log', results_log)
                # ah_sts = get_allStates_fromParsedResults(results_log, anharmonic=True)
                anharmonic_states = {tuple(sorted([str(i) for i in key])): value for key, value in ah_sts.items()}
                results['anharmonic_states'] = anharmonic_states
                len_nmodes = len([i for i in anharmonic_states if len(i)==1])
            except Exception as e:
                print('Failed at "anharmonic_states" with:', e)
                raise ValueError("smth went wrong")
            
        if 'harmonic_states' or 'nc_sqrt_eigval' in requested_data:
            try:
                # h_sts = get_allStates_fromParsedResults(results_log, anharmonic=False)
                # print('h_sts in parse_gaussian16_output()', h_sts)
                harmonic_states = {tuple(sorted([str(i) for i in key])): value for key, value in h_sts.items()}
                results['harmonic_states'] = harmonic_states
                fund_harm_dict = {k:v for k,v in harmonic_states.items() if len(k)==1}
                len_nmodes = len(fund_harm_dict)

                if 'nc_sqrt_eigval' in requested_data:
                    results['nc_sqrt_eigval'] = fund_harm_dict
            except Exception as e:
                print('Failed at "harmonic_states/nc_sqrt_eigval" with:', e)
                raise ValueError("smth went wrong")

    if 'B' or 'coriolis' in requested_data:
        linear = False
        if 'atoms' not in results:
            atoms, _ = get_equil_geo(log_lines)
        else:
            atoms = results['atoms']

        if len(atoms)==2:
            linear = True

        from .parseGaussian_forWilson import parse_coriolis
        rotational_constant, coriolis_constant = parse_coriolis(log_lines, len_nmodes, linear)
        
        if 'B' in requested_data:
            results['B'] = rotational_constant
        if 'coriolis' in requested_data:
            results['coriolis'] = coriolis_constant
    
    if 'dipgrad' or 'diphess' in requested_data:
        from .parseGaussian_forWilson import getDipDers_au
        mu = getDipDers_au(log_lines)

        if 'dipgrad' in requested_data:
            results['dipgrad'] = mu[0]
        if 'diphess' in requested_data:
            results['diphess'] = mu[1]
                
    if 'polgrad' or 'polhess' in requested_data:
        from .parseGaussian_forWilson import getPolarDers_au
        alpha = getPolarDers_au(log_lines)
        
        if 'polgrad' in requested_data:
            results['polgrad'] = alpha[0]
        if 'polhess' in requested_data:
            results['polhess'] = alpha[1]

    if 'cff' in requested_data:
        from .parseGaussian_forWilson import parse_cubic_constants, get_cubic_post
        cubic_df = parse_cubic_constants(log_lines)[0]
        selected_df1 = cubic_df[['I', 'J', 'K', 'K(I,J,K)']]
        cubic = selected_df1.to_numpy()
        
        cff_au = get_cubic_post(len_nmodes, cubic)
        results['cff'] = cff_au

    if 'qff' in requested_data:
        from .parseGaussian_forWilson import parse_quartic_constants, get_quartic_post
        quartic_df = parse_quartic_constants(log_lines)[0]
        selected_df2 = quartic_df[['I', 'J', 'K', 'L', 'K(I,J,K,L)']]
        quartic = selected_df2.to_numpy()

        qff_au = get_quartic_post(len_nmodes, quartic)
        results['qff'] = qff_au

    return results

    # units: dict[str, str] = field(default_factory=lambda: {
    #     "harmonic_states": "cm-1",
    #     "anharmonic_states": "cm-1",
    #     "equilibrium_geometry": "angstrom",
    #     "normal_modes": "angstrom",
    #     "hess": "",
    #     "B": "cm-1",
    #     "coriolis": "dimensionless",
    #     "dipgrad": "",
    #     "diphess": "",
    #     "polgrad": "",
    #     "polhess": "",
    #     "cff": "Ha*m_e(-3/2)*a0(-3)",
    #     "qff": "Ha*m_e(-2)*a0(-4)",
    #     "cff_au": "cm-1",
    #     "qff_au": "cm-1",
    #     })


def make_cfour_files_dict():
    return

def parse_cfour_output(requested_data: dict, files_dict: str, linear_molecule: bool) -> dict:

    nModesStart = 6 if linear_molecule else 7

    results = {}

    if 'equilibrium_geometry' or 'atoms' or 'normal_modes' in requested_data:
        from .parseCFOUR_forWilson import pMOLDEN
        coords, atoms, normal_modes_dict = pMOLDEN(files_dict['molden'])
        num_modes = len(normal_modes_dict)
    
        if 'equilibrium_geometry' in requested_data:
            results['equilibrium_geometry'] = coords

        if 'atoms' in requested_data:
            results['atoms'] = atoms

        if 'normal_modes' in requested_data:
            results['normal_modes'] = normal_modes_dict

    
    if 'anharmonic_states' or 'harmonic_states' or 'nc_sqrt_eigval' in requested_data:
        from .parseCFOUR_forWilson import parse_output_file
        vib_energy_levels_list, _, anharmonic_freqs, _, harmonic_freqs = parse_output_file(files_dict['out_file'])

        if 'anharmonic_states' in requested_data:
            anharm_states_dict = dict(zip(vib_energy_levels_list, anharmonic_freqs))
            anharmonic_states = {tuple(str(i-nModesStart) for i in k): v for k, v in anharm_states_dict.items()}
        
        if 'harmonic_states' or 'nc_sqrt_eigval' in requested_data:
            import numpy as np

            harm_states_dict = dict(zip(vib_energy_levels_list, harmonic_freqs))
            harmonic_states = {tuple(str(i-nModesStart) for i in k): v for k, v in harm_states_dict.items()}
            fundamentals_harmonic_int = {int(k[0]):v for k,v in harmonic_states.items() if len(k)==1}
            fund_harmonic_energies_array = np.array(list(fundamentals_harmonic_int.values()))
            
            labelsModes_original = [i + nModesStart for i in list(fundamentals_harmonic_int)]

    if 'B' or 'coriolis' in requested_data:
        from .parseCFOUR_forWilson import parse_coriolis
        rotational_constant, coriolis_constant = parse_coriolis(files_dict['out_file'],
                                                                num_modes,
                                                                startmode=nModesStart)
        if 'B' in requested_data:
            results['rotational_constant'] = rotational_constant
        if 'coriolis' in requested_data:
            results['coriolis'] = coriolis_constant

    if 'cff' in requested_data:
        from .parseCFOUR_forWilson import pCubicORQuartic, getCubicPost
        cubic = pCubicORQuartic(files_dict['cubic_file'])
        cff_au = getCubicPost(fundamentals_harmonic_int, cubic,
                            startmode=nModesStart, recipcm=False)
        cubic_cm_1 = getCubicPost(fundamentals_harmonic_int, cubic,
                                startmode=nModesStart, recipcm=True)
        
    if 'qff' in requested_data:
        from .parseCFOUR_forWilson import pCubicORQuartic, getQuarticPost
        quartic = pCubicORQuartic(files_dict['quartic_file'])
        qff_au = getQuarticPost(fundamentals_harmonic_int, quartic,
                                startmode=nModesStart, recipcm=False)
        quartic_cm_1 = getQuarticPost(fundamentals_harmonic_int, quartic,
                                    startmode=nModesStart, recipcm=True)
    
    if 'dipgrad' or 'diphess' in requested_data:
        from .parseCFOUR_forWilson import getDipoleDers_anharm_au_simple

        mu = getDipoleDers_anharm_au_simple(filenamebase=files_dict['dipole_file'], 
                                            labels=labelsModes_original, # why
                                            nModesStart=nModesStart,
                                            fund_harmonic_energies_array=fund_harmonic_energies_array)
    if 'polgrad' or 'polhess' in requested_data:
        from .parseCFOUR_forWilson import getPolarDers_pkl_au_simple

        alpha = getPolarDers_pkl_au_simple(polar_pkl_file=files_dict['polar_pkl'], 
                                        fund_harmonic_energies_array=fund_harmonic_energies_array)


        # units={
        #     "harmonic_states": "cm-1",
        #     "anharmonic_states": "cm-1",
        #     "equilibrium_geometry": "Bohr",
        #     "normal_modes": "Bohr",
        #     "hess": "",
        #     "B": "cm-1",
        #     "coriolis": "dimensionless", # ?
        #     "dipgrad": "a.u.",
        #     "diphess": "a.u.",
        #     "polgrad": "a.u.",
        #     "polhess": "a.u.",
        #     "cff": "cm-1",
        #     "qff": "cm-1",
        #     "cff_au": "Ha*m_e(-3/2)*a0(-3)", # a.u.
        #     "qff_au": "Ha*m_e(-2)*a0(-4)", # a.u.
        # })