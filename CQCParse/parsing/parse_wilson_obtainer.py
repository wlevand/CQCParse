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
        atoms, equilibrium_geometry = get_equil_geo(log_lines)
        
        if 'equilibrium_geometry' in requested_data:
            results['equilibrium_geometry'] = equilibrium_geometry
        
        if 'atoms' in requested_data:
            results['atoms'] = atoms

    if 'normal_modes' in requested_data:
        from .parseGaussian_forWilson import get_normal_modes
        nmodes = get_normal_modes(filename=log_file, Na=len(atoms))
        nmodes = {i: nm for i, nm in enumerate(nmodes)}
        results['normal_modes'] = nmodes
        len_nmodes = len(nmodes)
    
    if 'anharmonic_states' or 'harmonic_states' or 'nc_sqrt_eigval' in requested_data:
        from .parseGaussian_forWilson import parse_frequencies, get_allStates_fromParsedResults
        results_log = parse_frequencies(lines=log_lines)

        if 'anharmonic_states' in requested_data:
            ah_sts = get_allStates_fromParsedResults(results_log, anharmonic=True)
            anharmonic_states = {tuple([str(i) for i in key]): value for key, value in ah_sts.items()}
            results['anharmonic_states'] = anharmonic_states
            len_nmodes = len([i for i in anharmonic_states if len(i)==1])
            
        if 'harmonic_states' or 'nc_sqrt_eigval' in requested_data:
            h_sts = get_allStates_fromParsedResults(results_log, anharmonic=False)
            harmonic_states = {tuple([str(i) for i in key]): value for key, value in h_sts.items()}
            results['harmonic_states'] = harmonic_states
            fund_harm_dict = {k:v for k,v in harmonic_states.items() if len(k)==1}
            len_nmodes = len(fund_harm_dict)

            if 'nc_sqrt_eigval' in requested_data:
                results['nc_sqrt_eigval'] = fund_harm_dict
    
    if 'B' or 'coriolis' in requested_data:
        from .parseGaussian_forWilson import parse_coriolis
        rotational_constant, coriolis_constant = parse_coriolis(log_lines, len_nmodes)
        
        if 'B' in requested_data:
            results['rotational_constant'] = rotational_constant
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