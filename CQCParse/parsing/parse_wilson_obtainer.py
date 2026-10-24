from typing import Any


def is_linear(n_atoms: int, n_vib: int) -> bool:
    """A molecule of n_atoms atoms is linear if it has 3N-5 vibrational modes, nonlinear if 3N-6."""
    if n_vib == 3 * n_atoms - 5:
        return True
    if n_vib == 3 * n_atoms - 6:
        return False
    raise ValueError(f'{n_vib} vibrational modes for {n_atoms} atoms: expected {3 * n_atoms - 6} (nonlinear) '
                     f'or {3 * n_atoms - 5} (linear)')


def parse_from_source(requested_data: dict,
                      source_type: str,
                      base_file_loc: str,
                      lvl_theory: str,
                      basis_set: str,
                      reindex_modes: bool = False,
                      linear: bool | None = None) -> dict[str, Any]:
    """
    VALUES MUST BE IN ATOMIC UNITS

    returns a dict like requested_data dict, with values as data parsed from outputs

    lvl_theory, basis_set: not used; kept because wilson passes them.
    reindex_modes: Gaussian only, renumbers the modes from Gaussian's A order to its H order. The result's
        'reindex_modes' says whether that was done. CFOUR has one mode order: its data is never reindexed, and its
        'reindex_modes' is always False.
    linear: ignored, kept so that old calls still work (passing it gives a DeprecationWarning). Both readers decide
        from the file whether the molecule is linear (is_linear: 3N-5 vibrational modes).
    """
    if linear is not None:
        import warnings
        warnings.warn("parse_from_source: 'linear' is ignored, the readers decide it from the file",
                      DeprecationWarning, stacklevel=2)

    if source_type == 'gaussian':

        results_dict = parse_gaussian16_output(requested_data=requested_data, 
                                               log_file=base_file_loc)
        
        # reindexing if requested
        if reindex_modes:
            from CQCParse.relay.modes_reindexing import reindex_tensor, build_permutation, validate_mapping, reindex_dict
            """
            'normal_modes' -- already in H numbering
            'modes_mapping' -- contains mapping A to H
            Need to reindex:
             'anharmonic_states', 'harmonic_states', 'nc_sqrt_eigval'  -- dicts
             'coriolis', 'dipgrad', 'diphess', 'polgrad', 'polhess', 'cff', 'cff_rc', 'qff', 'qff_rc' -- tensors
            """
            # results_dict needs to have `modes_mapping` key
            validate_mapping(results_dict['modes_mapping'], len(results_dict['normal_modes']))

            perm = build_permutation(results_dict['modes_mapping'], nmodes=len(results_dict['normal_modes']))

            for key in ['coriolis', 'dipgrad', 'diphess', 'polgrad', 'polhess', 'cff', 'cff_rc', 'qff', 'qff_rc']:
                
                if key == 'coriolis':
                    results_dict[key] = reindex_tensor(results_dict[key], perm, [1,2])
                elif 'grad' in key:
                    results_dict[key] = reindex_tensor(results_dict[key], perm, [0])
                elif 'hess' in key:
                    results_dict[key] = reindex_tensor(results_dict[key], perm, [0,1])
                elif 'cff' in key:
                    results_dict[key] = reindex_tensor(results_dict[key], perm, [0,1,2])
                elif 'qff' in key:
                    results_dict[key] = reindex_tensor(results_dict[key], perm, [0,1,2,3])
            
            for key in ['anharmonic_states', 'harmonic_states', 'nc_sqrt_eigval']:
                if key in results_dict:
                    results_dict[key] = reindex_dict(results_dict[key], results_dict['modes_mapping'])

            results_dict['reindex_modes'] = True
        else:
            # to keep track of transformations
            results_dict['reindex_modes'] = False

        return results_dict
    
    elif source_type == 'cfour':
        if not isinstance(base_file_loc, dict):
            files_dict = make_cfour_files_dict(base_file_loc)
        else:
            files_dict = base_file_loc
        
        # todo: finish parse_cfour_output
        results_dict = parse_cfour_output(requested_data=requested_data,
                                          files_dict=files_dict)
        # as for Gaussian, to keep track of transformations (none for CFOUR)
        results_dict['reindex_modes'] = False
        return results_dict



def parse_gaussian16_output(requested_data: dict,
                            log_file: str):
    """
    
    Units:
        "harmonic_states": "cm-1",
        "anharmonic_states": "cm-1",
        "equilibrium_geometry": "angstrom",
        "normal_modes": "angstrom",
        "B": "cm-1",
        "coriolis": "dimensionless",
        "cff": "Ha*m_e(-3/2)*a0(-3)",
        "qff": "Ha*m_e(-2)*a0(-4)",
        "cff_rc": "cm-1",
        "qff_rc": "cm-1",

    cff and cff_rc come together: a request for either returns both. The same for qff and qff_rc.
    Gaussian prints both units (columns K and FI), so both are Gaussian's own numbers.

    Linear molecules (for B and coriolis): decided from the file, from the number of atoms and of vibrational
    modes (is_linear).
    """
    results = {}

    with open(log_file, 'r') as file:
        log_lines = [i.strip() for i in file.readlines()]

    if any(k in requested_data for k in ('equilibrium_geometry', 'atoms')): 
    
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
        from .parseGaussian_forWilson import get_normal_modes, get_equil_geo

        if 'atoms' not in results:
            atoms, _ = get_equil_geo(log_lines)
        else:
            atoms = results['atoms']

        try:
            # FIXME: `atoms` mayb not be defined at this point
            nmodes = get_normal_modes(filename=log_file, Na=len(atoms))
            nmodes = {i: nm for i, nm in enumerate(nmodes)}
            results['normal_modes'] = nmodes
        except Exception as e:
            print('Failed at "normal_modes" with:', e)
            raise ValueError("smth went wrong")

    # the harmonic fundamentals give the mode count (len_nmodes), the same as in nc_sqrt_eigval
    needs_freqs = ('harmonic_states', 'anharmonic_states', 'nc_sqrt_eigval', 'B', 'coriolis',
                   'cff', 'cff_rc', 'qff', 'qff_rc')
    if any(k in requested_data for k in needs_freqs):

        try:
            from .parse_g16_freqs import parse_frequencies, get_allStates_from_parsed_freqs
            results_log = parse_frequencies(lines=log_lines)
            h_sts, ah_sts = get_allStates_from_parsed_freqs(results_log)

            if not results_log:
                raise ValueError("result of parse_frequencies() is empty")

            harmonic_states = {tuple(sorted([str(i) for i in key])): value for key, value in h_sts.items()}
            fund_harm_dict = {k:v for k,v in harmonic_states.items() if len(k)==1}
            len_nmodes = len(fund_harm_dict)
        except Exception as e:
            print('Failed at "parse_frequencies" with:', e)
            raise ValueError("smth went wrong in parse_frequencies() or get_allStates_from_parsed_freqs()")

        if 'anharmonic_states' in requested_data:
            try:
                anharmonic_states = {tuple(sorted([str(i) for i in key])): value for key, value in ah_sts.items()}
                results['anharmonic_states'] = anharmonic_states
            except Exception as e:
                print('Failed at "anharmonic_states" with:', e)
                raise ValueError("smth went wrong")

        if any(k in requested_data for k in ('harmonic_states', 'nc_sqrt_eigval')):

            try:
                results['harmonic_states'] = harmonic_states

                if 'nc_sqrt_eigval' in requested_data:
                    results['nc_sqrt_eigval'] = {int(k[0]): v for k,v in fund_harm_dict.items()}
            except Exception as e:
                print('Failed at "harmonic_states/nc_sqrt_eigval" with:', e)
                raise ValueError("smth went wrong")

    if any(k in requested_data for k in ('B', 'coriolis')): 
        from .parseGaussian_forWilson import parse_Na, parse_coriolis
        linear = is_linear(parse_Na(log_lines), len_nmodes)

        rotational_constant, coriolis_constant = parse_coriolis(log_lines, len_nmodes, linear)
        
        if 'B' in requested_data:
            results['B'] = rotational_constant
        if 'coriolis' in requested_data:
            results['coriolis'] = coriolis_constant
    
    if any(k in requested_data for k in ('dipgrad', 'diphess')):

        from .parseGaussian_forWilson import getDipDers_au
        mu = getDipDers_au(log_lines)

        if 'dipgrad' in requested_data:
            results['dipgrad'] = mu[0]
        if 'diphess' in requested_data:
            results['diphess'] = mu[1]
                
    if any(k in requested_data for k in ('polgrad', 'polhess')):
    
        from .parseGaussian_forWilson import getPolarDers_au
        alpha = getPolarDers_au(log_lines)
        
        if 'polgrad' in requested_data:
            results['polgrad'] = alpha[0]
        if 'polhess' in requested_data:
            results['polhess'] = alpha[1]

    if any(k in requested_data for k in ('cff', 'cff_rc')):
        from .parseGaussian_forWilson import parse_cubic_constants, get_cubic_post
        cubic_df = parse_cubic_constants(log_lines)[0]
        selected_df1 = cubic_df[['I', 'J', 'K', 'K(I,J,K)']]
        cubic = selected_df1.to_numpy()
        
        cff_au = get_cubic_post(len_nmodes, cubic)
        results['cff'] = cff_au

        cubic_rcm = cubic_df[['I', 'J', 'K', 'FI(I,J,K)']].to_numpy() # cm-1
        results['cff_rc'] = get_cubic_post(len_nmodes, cubic_rcm, reduced=False)


    if any(k in requested_data for k in ('qff', 'qff_rc')):
        from .parseGaussian_forWilson import parse_quartic_constants, get_quartic_post
        quartic_df = parse_quartic_constants(log_lines)[0]
        selected_df2 = quartic_df[['I', 'J', 'K', 'L', 'K(I,J,K,L)']]
        quartic = selected_df2.to_numpy()

        qff_au = get_quartic_post(len_nmodes, quartic)
        results['qff'] = qff_au

        quartic_rcm = quartic_df[['I', 'J', 'K', 'L', 'FI(I,J,K,L)']].to_numpy() # cm-1
        results['qff_rc'] = get_quartic_post(len_nmodes, quartic_rcm, reduced=False)

    if 'modes_mapping' in requested_data:
        from .parseGaussian_forWilson import parse_mode_mapping
        results['modes_mapping'] = parse_mode_mapping(log_lines)

    return results


def make_cfour_files_dict(base_dir_loc):
    """
    files_dict_keys = ['molden', 'out_file', 'cubic_file', 'quartic_file', 'dipole_file', 'polar_pkl']
    """
    files_dict_keys = {'molden': 'MOLDEN', 
                       'out_file': 'out', 
                       'cubic_file': 'cubic', 
                       'quartic_file': 'quartic', 
                       'dipole_file': 'dipole', 
                       'polar_pkl': 'polar.pkl'}

    for k, v in files_dict_keys.items():
        files_dict_keys[k] = base_dir_loc + '/' + v

    return files_dict_keys


def parse_cfour_output(requested_data: dict, files_dict: str) -> dict:
    """
    files_dict_keys = ['molden', 'out_file', 'cubic_file', 'quartic_file', 'dipole_file', 'polar_pkl']

    Mode numbers: CFOUR numbers all 3N modes; the first 6 (5 for a linear molecule) are translations and
    rotations. The out file gives 3N and the vibrational modes, so the reader decides linear itself (is_linear)
    and returns the vibrations as modes 0 .. n-1.

    Units:
        "harmonic_states": "cm-1",
        "anharmonic_states": "cm-1",
        "equilibrium_geometry": "Bohr",
        "normal_modes": "Bohr",
        "hess": "",
        "B": "cm-1",
        "coriolis": "dimensionless", # ?
        "dipgrad": "a.u.",
        "diphess": "a.u.",
        "polgrad": "a.u.",
        "polhess": "a.u.",
        "cff": "Ha*m_e(-3/2)*a0(-3)", # a.u.
        "qff": "Ha*m_e(-2)*a0(-4)", # a.u.
        "cff_rc": "cm-1",
        "qff_rc": "cm-1",

    cff and cff_rc come together: a request for either returns both. The same for qff and qff_rc.
    CFOUR's cubic and quartic files are in cm-1, so cff_rc and qff_rc are CFOUR's own numbers; cff and qff are
    converted from them.

    AVERAGING: CFOUR lists some values twice, once per index order, from two numerical differentiations:
        diphess  every mode pair, e.g. (9,10) and (10,9) in the dipole(xyz) files
        qff, qff_rc  the (i,i,j,j) terms, e.g. (12,12,7,7) and (7,7,12,12) in the quartic file
    The two differ slightly (up to ~1e-3 of the value in the example files). The reader returns their mean, in
    every index order. See getDipoleDers_anharm and getQuarticPost.
    """
    results = {}

    # the out file gives the mode numbers (and linear), the mode count and the harmonic frequencies for the unit
    # conversion of the properties
    needs_out_file = ('anharmonic_states', 'harmonic_states', 'nc_sqrt_eigval', 'normal_modes', 'B', 'coriolis',
                      'cff', 'cff_rc', 'qff', 'qff_rc', 'dipgrad', 'diphess', 'polgrad', 'polhess')
    if any(k in requested_data for k in needs_out_file):
        import numpy as np
        from .parseCFOUR_forWilson import parse_output_file, count_normal_coordinates
        vib_energy_levels_list, _, anharmonic_freqs, _, harmonic_freqs = parse_output_file(files_dict['out_file'])

        # CFOUR mode numbers of the vibrations: 3N - n_vib + 1 .. 3N, i.e. from 7 (nonlinear) or 6 (linear)
        n_all_modes = count_normal_coordinates(files_dict['out_file'])
        fundamentals_original = sorted(k[0] for k in vib_energy_levels_list if len(k) == 1)
        linear = is_linear(n_all_modes // 3, len(fundamentals_original))
        nModesStart = n_all_modes - len(fundamentals_original) + 1
        if fundamentals_original != list(range(nModesStart, n_all_modes + 1)):
            raise ValueError(f'CFOUR fundamentals are modes {fundamentals_original}, '
                             f'expected {nModesStart} .. {n_all_modes} of {n_all_modes} normal coordinates')

        harm_states_dict = dict(zip(vib_energy_levels_list, harmonic_freqs))
        harmonic_states = {tuple(str(i-nModesStart) for i in k): v for k, v in harm_states_dict.items()}
        fundamentals_harmonic_int = {int(k[0]):v for k,v in harmonic_states.items() if len(k)==1}
        fund_harmonic_energies_array = np.array(list(fundamentals_harmonic_int.values()))
        num_modes = len(fundamentals_harmonic_int)

        labelsModes_original = [i + nModesStart for i in list(fundamentals_harmonic_int)]

        if 'anharmonic_states' in requested_data:
            anharm_states_dict = dict(zip(vib_energy_levels_list, anharmonic_freqs))
            anharmonic_states = {tuple(str(i-nModesStart) for i in k): v for k, v in anharm_states_dict.items()}
            results['anharmonic_states'] = anharmonic_states

        if any(k in requested_data for k in ('harmonic_states', 'nc_sqrt_eigval')):
            results['harmonic_states'] = harmonic_states

            if 'nc_sqrt_eigval' in requested_data:
                results['nc_sqrt_eigval'] = fundamentals_harmonic_int

    if any(k in requested_data for k in ('equilibrium_geometry', 'atoms', 'normal_modes')):
        from .parseCFOUR_forWilson import pMOLDEN
        # linear (from the out file, read above for normal_modes) changes only which MOLDEN vibrations are kept
        coords, atoms, normal_modes_dict = pMOLDEN(files_dict['molden'],
                                                   linear=linear if 'normal_modes' in requested_data else False)

        if 'equilibrium_geometry' in requested_data:
            results['equilibrium_geometry'] = coords

        if 'atoms' in requested_data:
            results['atoms'] = atoms

        if 'normal_modes' in requested_data:
            results['normal_modes'] = normal_modes_dict

    if any(k in requested_data for k in ('B', 'coriolis')):
        from .parseCFOUR_forWilson import parse_coriolis
        rotational_constant, coriolis_constant = parse_coriolis(files_dict['out_file'],
                                                                num_modes,
                                                                startmode=nModesStart)
        if 'B' in requested_data:
            results['B'] = rotational_constant
        if 'coriolis' in requested_data:
            results['coriolis'] = coriolis_constant

    if any(k in requested_data for k in ('cff', 'cff_rc')):
        from .parseCFOUR_forWilson import pCubicORQuartic, getCubicPost
        cubic = pCubicORQuartic(files_dict['cubic_file'])
        cff_au = getCubicPost(fundamentals_harmonic_int, cubic,
                            startmode=nModesStart, recipcm=False)
        cubic_cm_1 = getCubicPost(fundamentals_harmonic_int, cubic,
                                startmode=nModesStart, recipcm=True)
        results['cff'] = cff_au
        results['cff_rc'] = cubic_cm_1

    if any(k in requested_data for k in ('qff', 'qff_rc')):
        from .parseCFOUR_forWilson import pCubicORQuartic, getQuarticPost
        quartic = pCubicORQuartic(files_dict['quartic_file'])
        qff_au = getQuarticPost(fundamentals_harmonic_int, quartic,
                                startmode=nModesStart, recipcm=False)
        quartic_cm_1 = getQuarticPost(fundamentals_harmonic_int, quartic,
                                    startmode=nModesStart, recipcm=True)
        results['qff'] = qff_au
        results['qff_rc'] = quartic_cm_1

    if any(k in requested_data for k in ('dipgrad', 'diphess')):
        from .parseCFOUR_forWilson import getDipoleDers_anharm_au_simple

        mu = getDipoleDers_anharm_au_simple(filenamebase=files_dict['dipole_file'], 
                                            labels=labelsModes_original, # why
                                            nModesStart=nModesStart,
                                            fund_harmonic_energies_array=fund_harmonic_energies_array)
        if 'dipgrad' in requested_data:
            results['dipgrad'] = mu[0]
        if 'diphess' in requested_data:
            results['diphess'] = mu[1]

    if any(k in requested_data for k in ('polgrad', 'polhess')):
        from .parseCFOUR_forWilson import getPolarDers_pkl_au_simple

        alpha = getPolarDers_pkl_au_simple(polar_pkl_file=files_dict['polar_pkl'], 
                                        fund_harmonic_energies_array=fund_harmonic_energies_array)

        if 'polgrad' in requested_data:
            results['polgrad'] = alpha[0]
        if 'polhess' in requested_data:
            results['polhess'] = alpha[1]        
    
    return results
