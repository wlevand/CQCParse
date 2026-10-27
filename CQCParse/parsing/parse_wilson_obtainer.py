from collections.abc import Collection
from contextlib import contextmanager
from typing import Any

SOURCE_TYPES = ('gaussian', 'cfour')
# every name a reader knows; modes_mapping is Gaussian only (CFOUR has one mode order, so it returns no such key)
KNOWN_NAMES = ('equilibrium_geometry', 'atoms', 'normal_modes',
               'harmonic_states', 'anharmonic_states', 'nc_sqrt_eigval', 'B', 'coriolis',
               'dipgrad', 'diphess', 'polgrad', 'polhess', 'cff', 'cff_rc', 'qff', 'qff_rc', 'modes_mapping')
# what Gaussian numbers in its A order (its anharmonic section): the tensors, with their mode axes, and the dicts
REINDEXED_TENSORS = {'coriolis': (1, 2), 'dipgrad': (0,), 'diphess': (0, 1), 'polgrad': (0,), 'polhess': (0, 1),
                     'cff': (0, 1, 2), 'cff_rc': (0, 1, 2), 'qff': (0, 1, 2, 3), 'qff_rc': (0, 1, 2, 3)}
REINDEXED_DICTS = ('harmonic_states', 'anharmonic_states', 'nc_sqrt_eigval')


@contextmanager
def _source_note(source_type, location):
    """An error raised inside keeps its type and gets a note naming the source (the readers' errors do not)."""
    try:
        yield
    except Exception as e:
        e.add_note(f'while reading the {source_type} output {location}')
        raise


def is_linear(n_atoms: int, n_vib: int) -> bool:
    """A molecule of n_atoms atoms is linear if it has 3N-5 vibrational modes, nonlinear if 3N-6."""
    if n_vib == 3 * n_atoms - 5:
        return True
    if n_vib == 3 * n_atoms - 6:
        return False
    raise ValueError(f'{n_vib} vibrational modes for {n_atoms} atoms: expected {3 * n_atoms - 6} (nonlinear) '
                     f'or {3 * n_atoms - 5} (linear)')


def parse_from_source(requested_data: Collection[str],
                      source_type: str,
                      base_file_loc: str,
                      lvl_theory: str,
                      basis_set: str,
                      reindex_modes: bool = True,
                      linear: bool | None = None) -> dict[str, Any]:
    """
    VALUES MUST BE IN ATOMIC UNITS

    returns a dict like requested_data dict, with values as data parsed from outputs

    requested_data: the names to read, a list (as wilson passes) or a dict whose keys are the names. A name not in
        KNOWN_NAMES raises ValueError.
    source_type: 'gaussian' or 'cfour' (SOURCE_TYPES); anything else raises ValueError.
    Errors from the readers keep their type, with a note that names base_file_loc.
    lvl_theory, basis_set: not used; kept because wilson passes them.
    Mode order: by default, every mode index of both readers is in ascending harmonic frequency, so mode k is the
        same vibration in every key and in both programs' results. CFOUR has that order. Gaussian numbers the modes
        two ways: its harmonic section (normal_modes) in that order, H; its anharmonic section (every other key with
        mode indices) in its A order, in all the example files the Mulliken numbering (grouped by symmetry species,
        then descending frequency). The Gaussian reader renumbers A to H, with the table the file prints;
        modes_mapping returns that table, A to H, both counted from 1.
        Ascending frequency is not a fixed label: two close modes can swap between methods, so mode k of two
        results can be different vibrations.
        CFOUR's normal_modes keys are its own mode numbers (from 7 for a nonlinear molecule).
    reindex_modes: True (the default), the order above. False, Gaussian only: every key in Gaussian's A order,
        normal_modes included. The result's 'reindex_modes' is True when Gaussian data was renumbered to H order;
        for CFOUR, which is never renumbered, it is always False.
    linear: ignored, kept so that old calls still work (passing it gives a DeprecationWarning). Both readers decide
        from the file whether the molecule is linear (is_linear: 3N-5 vibrational modes).
    """
    if linear is not None:
        import warnings
        warnings.warn("parse_from_source: 'linear' is ignored, the readers decide it from the file",
                      DeprecationWarning, stacklevel=2)

    if source_type not in SOURCE_TYPES:
        raise ValueError(f'source_type {source_type!r}: expected one of {SOURCE_TYPES}')
    unknown = [k for k in requested_data if k not in KNOWN_NAMES]
    if unknown:
        raise ValueError(f'unknown names in requested_data: {unknown}; known: {KNOWN_NAMES}')

    if source_type == 'gaussian':

        with _source_note(source_type, base_file_loc):
            results_dict = parse_gaussian16_output(requested_data=requested_data,
                                                   log_file=base_file_loc,
                                                   reindex_modes=reindex_modes)
        # to keep track of transformations
        results_dict['reindex_modes'] = reindex_modes
        return results_dict
    
    elif source_type == 'cfour':
        if not isinstance(base_file_loc, dict):
            files_dict = make_cfour_files_dict(base_file_loc)
        else:
            files_dict = base_file_loc

        with _source_note(source_type, base_file_loc):
            results_dict = parse_cfour_output(requested_data=requested_data,
                                              files_dict=files_dict)
        # as for Gaussian, to keep track of transformations (none for CFOUR)
        results_dict['reindex_modes'] = False
        return results_dict



def parse_gaussian16_output(requested_data: dict,
                            log_file: str,
                            reindex_modes: bool = True):
    """
    Mode order: H, ascending harmonic frequency (reindex_modes, the default), or Gaussian's A order; see
    parse_from_source.

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
    cff_rc and qff_rc are Gaussian's column FI (cm-1, dimensionless normal coordinates). cff and qff are converted
    from them with the harmonic frequencies (CQCParse.utils.reduced_to_au, which explains the formula), the same
    way as in the CFOUR reader. They are not read from Gaussian's column K (atomic units, amu): K has only 5
    decimals, so a small term keeps 1 or 2 digits there (up to 11% off in the example files), while FI keeps them.

    Linear molecules (for B and coriolis): decided from the file, from the number of atoms and of vibrational
    modes (is_linear).
    """
    results = {}
    len_nmodes = None

    with open(log_file, 'r') as file:
        log_lines = [i.strip() for i in file]

    if any(k in requested_data for k in ('equilibrium_geometry', 'atoms')): 
    
        from .parseGaussian_forWilson import get_equil_geo
        atoms, equilibrium_geometry = get_equil_geo(log_lines)

        if 'equilibrium_geometry' in requested_data:
            results['equilibrium_geometry'] = equilibrium_geometry

        if 'atoms' in requested_data:
            results['atoms'] = atoms

    if 'normal_modes' in requested_data:
        from .parseGaussian_forWilson import get_normal_modes, parse_Na

        # only the atom count is needed, not the geometry (which get_equil_geo refuses for ACAC)
        nmodes = get_normal_modes(log_lines, parse_Na(log_lines))
        nmodes = {i: nm for i, nm in enumerate(nmodes)}
        results['normal_modes'] = nmodes

    # the harmonic fundamentals give the mode count (len_nmodes), the same as in nc_sqrt_eigval
    needs_freqs = ('harmonic_states', 'anharmonic_states', 'nc_sqrt_eigval', 'B', 'coriolis',
                   'cff', 'cff_rc', 'qff', 'qff_rc')
    if any(k in requested_data for k in needs_freqs):

        from .parse_g16_freqs import get_allStates_from_parsed_freqs, parse_frequencies
        results_log = parse_frequencies(lines=log_lines)
        if not results_log:
            raise ValueError('the anharmonic frequency section has no states')
        h_sts, ah_sts = get_allStates_from_parsed_freqs(results_log)

        harmonic_states = {tuple(sorted([str(i) for i in key])): value for key, value in h_sts.items()}
        fund_harm_dict = {k:v for k,v in harmonic_states.items() if len(k)==1}
        len_nmodes = len(fund_harm_dict)
        # harmonic wavenumbers in mode order 0 .. n-1, for the conversion of cff and qff
        harmonic_cm = [fund_harm_dict[(str(m),)] for m in range(len_nmodes)]

        if 'anharmonic_states' in requested_data:
            anharmonic_states = {tuple(sorted([str(i) for i in key])): value for key, value in ah_sts.items()}
            results['anharmonic_states'] = anharmonic_states

        if any(k in requested_data for k in ('harmonic_states', 'nc_sqrt_eigval')):
            results['harmonic_states'] = harmonic_states

            if 'nc_sqrt_eigval' in requested_data:
                results['nc_sqrt_eigval'] = {int(k[0]): v for k,v in fund_harm_dict.items()}

    if any(k in requested_data for k in ('B', 'coriolis')): 
        from .parseGaussian_forWilson import parse_coriolis, parse_Na
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
        from CQCParse.utils import reduced_to_au

        from .parseGaussian_forWilson import get_cubic_post, parse_cubic_constants
        cubic_df = parse_cubic_constants(log_lines)[0]
        cubic_rcm = cubic_df[['I', 'J', 'K', 'FI(I,J,K)']].to_numpy() # cm-1
        cff_rc = get_cubic_post(len_nmodes, cubic_rcm)

        # from FI, not from column K (see docstring)
        results['cff'] = reduced_to_au(cff_rc, harmonic_cm)
        results['cff_rc'] = cff_rc


    if any(k in requested_data for k in ('qff', 'qff_rc')):
        from CQCParse.utils import reduced_to_au

        from .parseGaussian_forWilson import get_quartic_post, parse_quartic_constants
        quartic_df = parse_quartic_constants(log_lines)[0]
        quartic_rcm = quartic_df[['I', 'J', 'K', 'L', 'FI(I,J,K,L)']].to_numpy() # cm-1
        qff_rc = get_quartic_post(len_nmodes, quartic_rcm)

        # from FI, not from column K (see docstring)
        results['qff'] = reduced_to_au(qff_rc, harmonic_cm)
        results['qff_rc'] = qff_rc

    if 'modes_mapping' in requested_data:
        from .parseGaussian_forWilson import parse_mode_mapping
        results['modes_mapping'] = parse_mode_mapping(log_lines)

    # one mode order for the whole result: every key read above is in A order, but normal_modes in H order
    a_ordered = [k for k in results if k in REINDEXED_TENSORS or k in REINDEXED_DICTS]
    if a_ordered if reindex_modes else 'normal_modes' in results:
        from CQCParse.relay.modes_reindexing import build_permutation, reindex_dict, reindex_tensor, validate_mapping

        from .parseGaussian_forWilson import parse_mode_mapping
        mapping = results.get('modes_mapping') or parse_mode_mapping(log_lines)   # A to H, both from 1
        n = len(mapping)
        validate_mapping(mapping, n)
        counts = [len_nmodes, len(results['normal_modes']) if 'normal_modes' in results else None]
        if any(count is not None and count != n for count in counts):
            raise ValueError(f'the A to H mode table has {n} modes, the file {[c for c in counts if c is not None]}')

        if reindex_modes:
            perm = build_permutation(mapping, n)
            for key in a_ordered:
                if key in REINDEXED_TENSORS:
                    results[key] = reindex_tensor(results[key], perm, REINDEXED_TENSORS[key])
                else:
                    results[key] = reindex_dict(results[key], mapping)
        else:
            results['normal_modes'] = {a - 1: results['normal_modes'][h - 1] for a, h in sorted(mapping.items())}

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
    converted from them with the harmonic frequencies (CQCParse.utils.reduced_to_au, which explains the formula),
    the same way as in the Gaussian reader.

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

        from .parseCFOUR_forWilson import count_normal_coordinates, parse_output_file
        vib_energy_levels_list, _, anharmonic_freqs, _, harmonic_freqs = parse_output_file(files_dict['out_file'])

        # CFOUR mode numbers of the vibrations: 3N - n_vib + 1 .. 3N, i.e. from 7 (nonlinear) or 6 (linear)
        n_all_modes = count_normal_coordinates(files_dict['out_file'])
        fundamentals_original = sorted(k[0] for k in vib_energy_levels_list if len(k) == 1)
        linear = is_linear(n_all_modes // 3, len(fundamentals_original))
        nModesStart = n_all_modes - len(fundamentals_original) + 1
        if fundamentals_original != list(range(nModesStart, n_all_modes + 1)):
            raise ValueError(f'CFOUR fundamentals are modes {fundamentals_original}, '
                             f'expected {nModesStart} .. {n_all_modes} of {n_all_modes} normal coordinates')

        harm_states_dict = dict(zip(vib_energy_levels_list, harmonic_freqs, strict=True))
        harmonic_states = {tuple(str(i-nModesStart) for i in k): v for k, v in harm_states_dict.items()}
        fundamentals_harmonic_int = {int(k[0]):v for k,v in harmonic_states.items() if len(k)==1}
        num_modes = len(fundamentals_harmonic_int)
        # harmonic wavenumbers in mode order 0 .. n-1, for the unit conversion of the properties
        fund_harmonic_energies_array = np.array([fundamentals_harmonic_int[m] for m in range(num_modes)])

        labelsModes_original = [i + nModesStart for i in list(fundamentals_harmonic_int)]

        if 'anharmonic_states' in requested_data:
            anharm_states_dict = dict(zip(vib_energy_levels_list, anharmonic_freqs, strict=True))
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
            if not normal_modes_dict:
                raise ValueError(f'no normal modes: no [FR-NORM-COORD] vibrations in {files_dict["molden"]}')
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
        from CQCParse.utils import reduced_to_au

        from .parseCFOUR_forWilson import getCubicPost, pCubicORQuartic
        cubic = pCubicORQuartic(files_dict['cubic_file'])
        cubic_cm_1 = getCubicPost(fundamentals_harmonic_int, cubic, startmode=nModesStart)
        results['cff'] = reduced_to_au(cubic_cm_1, fund_harmonic_energies_array)
        results['cff_rc'] = cubic_cm_1

    if any(k in requested_data for k in ('qff', 'qff_rc')):
        from CQCParse.utils import reduced_to_au

        from .parseCFOUR_forWilson import getQuarticPost, pCubicORQuartic
        quartic = pCubicORQuartic(files_dict['quartic_file'])
        quartic_cm_1 = getQuarticPost(fundamentals_harmonic_int, quartic, startmode=nModesStart)
        results['qff'] = reduced_to_au(quartic_cm_1, fund_harmonic_energies_array)
        results['qff_rc'] = quartic_cm_1

    if any(k in requested_data for k in ('dipgrad', 'diphess')):
        from .parseCFOUR_forWilson import getDipoleDers_anharm_au

        mu = getDipoleDers_anharm_au(filenamebase=files_dict['dipole_file'],
                                     labels=labelsModes_original,    # CFOUR's mode numbers of the vibrations
                                     nModesStart=nModesStart,
                                     fund_harmonic_energies_array=fund_harmonic_energies_array)
        if 'dipgrad' in requested_data:
            results['dipgrad'] = mu[0]
        if 'diphess' in requested_data:
            results['diphess'] = mu[1]

    if any(k in requested_data for k in ('polgrad', 'polhess')):
        from .parseCFOUR_forWilson import getPolarDers_pkl_au

        alpha = getPolarDers_pkl_au(polar_pkl_file=files_dict['polar_pkl'],
                                    fund_harmonic_energies_array=fund_harmonic_energies_array)

        if 'polgrad' in requested_data:
            results['polgrad'] = alpha[0]
        if 'polhess' in requested_data:
            results['polhess'] = alpha[1]        
    
    return results
