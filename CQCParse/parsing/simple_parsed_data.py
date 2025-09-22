from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class ParsedDataSimple:
    """
    Storage of parsed data - for a single calculation instance. 
    Flat structure
    """
    # identifiers
    molecule: str
    program: str
    level_of_theory: str
    basis: str

    atoms: np.ndarray[str] | list[str] = None # list of atoms as str
    equilibrium_geometry: np.ndarray = None
    
    harmonic_states: dict = None
    anharmonic_states: dict = None
    normal_modes: dict = None

    hess: np.ndarray = None
    B: np.ndarray = None # rotational constants, cm-1
    coriolis: np.ndarray = None # coriolis constants, cm-1
    fermi_resonances: list = None

    dipgrad: np.ndarray = None
    diphess: np.ndarray = None
    polgrad: np.ndarray = None
    polhess: np.ndarray = None
    cff: np.ndarray = None # cm-1
    qff: np.ndarray = None # cm-1
    cff_au: np.ndarray = None # [Hartree*m_e(-3/2)*a0(-3)]
    qff_au: np.ndarray = None # [Hartree*m_e(-2)*a0(-4)], unmassweighted?

    units: dict[str, str] = field(default_factory=lambda: {
        "harmonic_states": "cm-1",
        "anharmonic_states": "cm-1",
        "equilibrium_geometry": "angstrom",
        "normal_modes": "angstrom",
        "hess": "",
        "B": "cm-1",
        "coriolis": "dimensionless",
        "dipgrad": "",
        "diphess": "",
        "polgrad": "",
        "polhess": "",
        "cff": "Ha*m_e(-3/2)*a0(-3)",
        "qff": "Ha*m_e(-2)*a0(-4)",
        "cff_au": "cm-1",
        "qff_au": "cm-1",
        })



def parse_gaussian16_output(molecule: str, level_of_theory: str, basis: str, 
                            log_file: str) -> ParsedDataSimple:
    from .parseGaussian_forWilson import get_allStates_fromParsedResults, get_equil_geo, get_normal_modes, \
        parse_frequencies, getDipDers_au, getPolarDers_au, parse_cubic_constants, parse_quartic_constants, get_cubic_post, \
        get_quartic_post, parse_coriolis
    
    with open(log_file, 'r') as file:
        log_lines = [i.strip() for i in file.readlines()]
    
    atoms, equilibrium_geometry = get_equil_geo(log_lines)
    nmodes = get_normal_modes(filename=log_file, Na=len(atoms))
    nmodes = {i: nm for i, nm in enumerate(nmodes)}
    results_log = parse_frequencies(lines=log_lines)

    ah_sts = get_allStates_fromParsedResults(results_log, anharmonic=True)
    h_sts = get_allStates_fromParsedResults(results_log, anharmonic=False)

    anharmonic_states = {tuple(str(i) for i in key): value for key, value in ah_sts.items()}
    harmonic_states = {tuple(str(i) for i in key): value for key, value in h_sts.items()}
    
    rotational_constant, coriolis_constant = parse_coriolis(log_lines, len(nmodes))
    mu = getDipDers_au(log_lines)
    alpha = getPolarDers_au(log_lines)

    cubic_df = parse_cubic_constants(log_lines)[0]
    cubic_rcm = cubic_df[['I', 'J', 'K', 'FI(I,J,K)']].to_numpy()
    selected_df1 = cubic_df[['I', 'J', 'K', 'K(I,J,K)']]
    cubic = selected_df1.to_numpy()

    quartic_df = parse_quartic_constants(log_lines)[0]
    quartic_rcm = quartic_df[['I', 'J', 'K', 'L', 'FI(I,J,K,L)']].to_numpy()
    selected_df2 = quartic_df[['I', 'J', 'K', 'L', 'K(I,J,K,L)']]
    quartic = selected_df2.to_numpy()

    cff_au = get_cubic_post(len(nmodes), cubic)
    qff_au = get_quartic_post(len(nmodes), quartic)
    cff = get_cubic_post(len(nmodes), cubic_rcm, reduced=False)
    qff = get_quartic_post(len(nmodes), quartic_rcm, reduced=False)

    return ParsedDataSimple(molecule=molecule,
                            program='gaussian',
                            level_of_theory=level_of_theory,
                            basis=basis,
                            atoms=atoms,
                            equilibrium_geometry=equilibrium_geometry,
                            harmonic_states=harmonic_states,
                            anharmonic_states=anharmonic_states,
                            normal_modes=nmodes,
                            hess=None,
                            B=rotational_constant,
                            coriolis=coriolis_constant,
                            fermi_resonances=[],
                            dipgrad=mu[0],
                            diphess=mu[1],
                            polgrad=alpha[0],
                            polhess=alpha[1],
                            cff=cff,
                            qff=qff,
                            cff_au=cff_au,
                            qff_au=qff_au,
                            units={
                                "harmonic_states": "cm-1",
                                "anharmonic_states": "cm-1",
                                "equilibrium_geometry": "Angstrom",
                                "normal_modes": "Angstrom",
                                "hess": "",
                                "B": "cm-1",
                                "coriolis": "dimensionless", # ?
                                "dipgrad": "a.u.",
                                "diphess": "a.u.",
                                "polgrad": "a.u.",
                                "polhess": "a.u.",
                                "cff": "cm-1",
                                "qff": "cm-1",
                                "cff_au": "Ha*m_e(-3/2)*a0(-3)", # a.u.
                                "qff_au": "Ha*m_e(-2)*a0(-4)", # a.u.
                            })


def parse_cfour_output(molecule: str, level_of_theory: str, basis, 
                       files_dict: str, linear_molecule: bool) -> ParsedDataSimple:
    from .parseCFOUR_forWilson import (pMOLDEN, parse_output_file, parse_coriolis,
                                    getCubicPost, getQuarticPost, getDipoleDers_anharm_au_simple,
                                    getPolarDers_pkl_au_simple,
                                    pCubicORQuartic)
    
    coords, atoms, normal_modes_dict = pMOLDEN(files_dict['molden'])
    nModesStart = 6 if linear_molecule else 7
    num_modes = len(normal_modes_dict)

    rotational_constant, coriolis_constant = parse_coriolis(files_dict['out_file'],
                                                            num_modes,
                                                            startmode=nModesStart)
    vib_energy_levels_list, _, anharmonic_freqs, _, harmonic_freqs = parse_output_file(files_dict['out_file'])
    anharm_states_dict = dict(zip(vib_energy_levels_list, anharmonic_freqs))
    harm_states_dict = dict(zip(vib_energy_levels_list, harmonic_freqs))

    anharmonic_states = {tuple(str(i-nModesStart) for i in k): v for k, v in anharm_states_dict.items()}
    harmonic_states = {tuple(str(i-nModesStart) for i in k): v for k, v in harm_states_dict.items()}
    fundamentals_harmonic_int = {int(k[0]):v for k,v in harmonic_states.items() if len(k)==1}

    fund_harmonic_energies_array = np.array(list(fundamentals_harmonic_int.values()))

    cubic = pCubicORQuartic(files_dict['cubic_file'])
    quartic = pCubicORQuartic(files_dict['quartic_file'])

    cff_au = getCubicPost(fundamentals_harmonic_int, cubic,
                          startmode=nModesStart, recipcm=False)
    qff_au = getQuarticPost(fundamentals_harmonic_int, quartic,
                            startmode=nModesStart, recipcm=False)

    cubic_cm_1 = getCubicPost(fundamentals_harmonic_int, cubic,
                              startmode=nModesStart, recipcm=True)
    quartic_cm_1 = getQuarticPost(fundamentals_harmonic_int, quartic,
                                  startmode=nModesStart, recipcm=True)
    labelsModes_original = [i + nModesStart for i in list(fundamentals_harmonic_int)]

    mu = getDipoleDers_anharm_au_simple(filenamebase=files_dict['dipole_file'], 
                                        labels=labelsModes_original, 
                                        nModesStart=nModesStart,
                                        fund_harmonic_energies_array=fund_harmonic_energies_array)
    alpha = getPolarDers_pkl_au_simple(polar_pkl_file=files_dict['polar_pkl'], 
                                       fund_harmonic_energies_array=fund_harmonic_energies_array)

    return ParsedDataSimple(molecule=molecule,
                            program='cfour',
                            level_of_theory=level_of_theory,
                            basis=basis,
                            atoms=atoms,
                            equilibrium_geometry=coords,
                            harmonic_states=harmonic_states,
                            anharmonic_states=anharmonic_states,
                            normal_modes=normal_modes_dict,
                            hess=None,
                            B=rotational_constant,
                            coriolis=coriolis_constant,
                            fermi_resonances=[],
                            dipgrad=mu[0],
                            diphess=mu[1],
                            polgrad=alpha[0],
                            polhess=alpha[1],
                            cff=cubic_cm_1,
                            qff=quartic_cm_1,
                            cff_au=cff_au,
                            qff_au=qff_au,
                            units={
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
                                "cff": "cm-1",
                                "qff": "cm-1",
                                "cff_au": "Ha*m_e(-3/2)*a0(-3)", # a.u.
                                "qff_au": "Ha*m_e(-2)*a0(-4)", # a.u.
                            })