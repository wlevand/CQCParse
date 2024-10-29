"""
#################################################################################################
##                                                                                             ##
##                             Parsing Gaussian output files                                   ##
##                                                                                             ##
#################################################################################################
#
# Files:
#     - .log  --- the main full output file that contains all the relevant data
#     - .fchk --- formcheck (generated from checkpoint file)
"""

import numpy as np
# np.set_printoptions(linewidth=250, suppress=True, precision=3)
import sys
import pandas as pd
pd.set_option('display.max_rows', sys.maxsize)

class GaussianDataParser(object):

    def __init__(self, all_files_dict: dict):
        self.all_files_dict = all_files_dict
        # {'log', 'fchk', 'com'}

        self.nModesStart = None

        self.dipole_first_derivatives = None
        self.dipole_second_derivatives = None
        self.polarizability_first_derivatives = None
        self.polarizability_second_derivatives = None

        self.fundamentals_harmonic_str = None
        self.fundamentals_anharmonic_str = None

        self.fundamentals_harmonic_int = None
        self.fundamentals_anharmonic_int = None

        self.harmonic_states = None
        self.anharmonic_states = None
        self.cubic_force_constants = None
        self.quartic_constants = None

        self.equilibrium_geometry = None
        self.Q_normal_coordinates = None
        self.q_normal_coordinates_dimensionless = None

        self.atoms = None
        self.basis = None
        self.lot = None

    def getData(self, linear_molecule: bool = False):
        """Collect the data into the attributes.
        Uses methods:
            parse_output_file,
            pCubicORQuartic, getCubicPost,
            getDipoleDers_anharm,
            'polar_pkl' file <- getPolarDers(getDisplacementsPolarData,
                                getRotationMatrix, pTensor),
                                pklPolder
        """
        # {'log', 'fchk', 'com'}
        self.nModesStart = 6 if linear_molecule else 7

        results_log = parse_frequencies(self.all_files_dict['files']['3quanta'])
        self.fundamentals_anharmonic_int = {int(k)-1: float(v) for k, v in zip(results_log['Fundamental Bands']['mode_a'],
                                                                               results_log['Fundamental Bands'][2])}
        self.fundamentals_harmonic_int = {int(k)-1: float(v) for k, v in zip(results_log['Fundamental Bands']['mode_a'],
                                                                             results_log['Fundamental Bands'][1])}

        self.fundamentals_harmonic_str = {str(k):v for k,v in self.fundamentals_harmonic_int.items()}
        self.fundamentals_anharmonic_str = {str(k):v for k,v in self.fundamentals_anharmonic_int.items()}

        ah_sts = get_allStates_fromParsedResults(results_log, anharmonic=True)
        h_sts = get_allStates_fromParsedResults(results_log, anharmonic=False)

        self.anharmonic_states = {tuple(str(i) for i in key): value for key, value in ah_sts.items()}
        self.harmonic_states = {tuple(str(i) for i in key): value for key, value in h_sts.items()}

        mu = getDipDers_au(self.all_files_dict['files']['log'])
        self.dipole_first_derivatives = mu[0]
        self.dipole_second_derivatives = mu[1]

        alpha = getPolarDers_au(self.all_files_dict['files']['log'])
        self.polarizability_first_derivatives = alpha[0]
        self.polarizability_second_derivatives = alpha[1]

        cubic_df = parse_cubic_constants(self.all_files_dict['files']['log'])[0]
        selected_df = cubic_df[['I', 'J', 'K', 'K(I,J,K)']]
        cubic = selected_df.to_numpy()
        self.cubic_force_constants = get_cubic_post(len(self.fundamentals_harmonic_str), cubic)

# used in retrievedata.py
def parse_frequencies(file_path: str) -> dict[str: pd.DataFrame]:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    sections = ["Fundamental Bands", "Overtones", "Combination Bands"]
    results = {section: [] for section in sections}
    results_harm = {section: [] for section in sections}
    current_section = None

    start = False
    units_counter = 0
    for line in lines:
        if "Anharmonic Infrared Spectroscopy" in line:
            start = True
        elif "Units: Transition energies" in line:
            units_counter += 1
            if units_counter == 2:
                start = False
        elif start:
            if any(section in line for section in sections):
                current_section = next(section for section in sections if section in line)
            elif current_section:
                if '------------' not in line:
                    linelist = line.split()
                    # inserting None at the desired index 2
                    if len(linelist)==5 and current_section=='Combination Bands': linelist.insert(2, None)

                    results[current_section].append(linelist)

    results_dataframes = {}
    for section, data in results.items():
        if section != 'Overtones':
            results_dataframes[section] = pd.DataFrame(data[1:-1])
        else:
            results_dataframes[section] = pd.DataFrame(data[2:-1])

        main_numbers = [i.split('(')[0] for i in results_dataframes[section][0]]
        sub_numbers = [int(i[:-1].split('(')[1]) for i in results_dataframes[section][0]]
        # nserting columns at specific positions
        results_dataframes[section].insert(1, 'mode_a', main_numbers)
        results_dataframes[section].insert(2, 'n_a', sub_numbers)
        results_dataframes[section].drop(results_dataframes[section].columns[0], axis=1, inplace=True)

        if section=='Combination Bands':
            main_numbers = [int(i.split('(')[0]) for i in results_dataframes[section][1]]
            sub_numbers = [int(i[:-1].split('(')[1]) for i in results_dataframes[section][1]]

            results_dataframes[section].insert(3, 'mode_b', main_numbers)
            results_dataframes[section].insert(4, 'n_b', sub_numbers)
            results_dataframes[section].drop(results_dataframes[section].columns[2], axis=1, inplace=True)

            main_numbers = [int(i.split('(')[0]) if i is not None else i for i in results_dataframes[section][2]]
            sub_numbers = [int(i[:-1].split('(')[1]) if i is not None else i for i in results_dataframes[section][2]]

            results_dataframes[section].insert(5, 'mode_c', main_numbers)
            results_dataframes[section].insert(6, 'n_c', sub_numbers)
            results_dataframes[section].drop(results_dataframes[section].columns[4], axis=1, inplace=True)

    return results_dataframes

def get_allStates_fromParsedResults(results: pd.DataFrame, anharmonic: bool = False) -> dict:
    """results is a DataFrame from parse_frequencies()"""
    if anharmonic:
        results['Combination Bands']['mode_c'] = results['Combination Bands']['mode_c'].fillna(0)
        results['Combination Bands']['n_c'] = results['Combination Bands']['n_c'].fillna(0)
        funddict = {tuple([int(k) - 1]): float(v) for k, v in
                    zip(results['Fundamental Bands']['mode_a'], results['Fundamental Bands'][2])}

        states = {
            tuple(sorted([int(t)-1] * int(n))): float(v)
            for t, v, n in
            zip(results['Overtones']['mode_a'], results['Overtones'][2], results['Overtones']['n_a'])
        }
        combinationbands = {
            tuple(
                sorted([int(t1)-1] * int(n1) + [t2-1] * int(n2) + [
                    (int(t3) - 1)] * int(n3))
            ): float(v)
            for t1, t2, t3, v, n1, n2, n3 in
            zip(results['Combination Bands']['mode_a'], results['Combination Bands']['mode_b'],
                results['Combination Bands']['mode_c'],
                results['Combination Bands'][4], results['Combination Bands']['n_a'],
                results['Combination Bands']['n_b'], results['Combination Bands']['n_c'])
        }
        allstates_anharm = {**funddict, **states, **combinationbands}
        # allstates_anharm = {key: allstates_anharm[key] for key in sorted(allstates_anharm, key=allstates_anharm.get)}
        return allstates_anharm

    else:
        funddict1 = {tuple([int(k) - 1]): float(v) for k, v in
                     zip(results['Fundamental Bands']['mode_a'], results['Fundamental Bands'][1])}
        states1 = {tuple(sorted([int(t)-1] * int(n))): float(v) for t, v, n in
                   zip(results['Overtones']['mode_a'], results['Overtones'][1], results['Overtones']['n_a'])}
        combinationbands1 = {
            tuple(
                sorted([int(t1)-1] * int(n1) + [t2-1] * int(n2) + [
                    (int(t3) - 1)] * int(n3))
            ): float(v)
            for t1, t2, t3, v, n1, n2, n3 in
            zip(results['Combination Bands']['mode_a'], results['Combination Bands']['mode_b'],
                results['Combination Bands']['mode_c'],
                results['Combination Bands'][3], results['Combination Bands']['n_a'],
                results['Combination Bands']['n_b'], results['Combination Bands']['n_c'])
        }
        allstates_harm = {**funddict1, **states1, **combinationbands1}
        return allstates_harm

def get_detected_resonances_g16(filepath: str) -> list[str]:

    with open(filepath, 'r') as file:
        file_content = file.read()

    if "Resonance Analysis" in file_content:
        with open(filepath, 'r') as file:
            file_lines = file.readlines()
        found_resonances_str = []
        inFR = False
        for line in file_lines:
            if 'I      J  +   K' in line:
                inFR = True
                col_names = line.strip().split()
                found_resonances_str.append(line)
            if 'Active Fermi resonances' in line:
                number_of_FR = int(line.strip().split()[0])
                found_resonances_str.append(f'There are {number_of_FR} Fermi resonances')
                inFR = False

            if inFR:
                line_numbers = line.strip().split()
                if len(line_numbers)>0 and line not in found_resonances_str:
                    found_resonances_str.append(line)
        return found_resonances_str

def getDipDers_log(logfile: str) -> tuple:
    """
    Dipole derivatives: first order and second order
    Return: tuple[np.ndarray - shape(NM, 3), np.ndarray - shape(NM, NM, 3)]
    """
    dipl, units = parse_dipole_moment(logfile)
    a2d = dipl.loc[dipl['P'] == 'P1', ['X', 'Y', 'Z']].to_numpy()
    shpNM = a2d.shape[0]
    a2d2 = dipl.loc[dipl['P'] == 'P2', ['X', 'Y', 'Z']].to_numpy()
    a2d2_3d = np.zeros((shpNM, shpNM, 3))

    for i, j, xyz in zip(dipl.loc[dipl['P'] == 'P2', 'i'], dipl.loc[dipl['P'] == 'P2', 'j'], a2d2):
        a2d2_3d[int(i) - 1, int(j) - 1] = xyz
        a2d2_3d[int(j) - 1, int(i) - 1] = xyz

    return tuple([a2d, a2d2_3d])

def getDipDers_au(logfile: str) -> tuple:
    a2d, a2d2_3d = getDipDers_log(logfile)
    from scipy import constants
    # to go from amu to au mass unit (m_e)
    amc_au = constants.physical_constants['atomic mass constant'][0] / \
             constants.physical_constants['atomic unit of mass'][0]

    # transformation from Gaussian to Wilson units
    firstder = a2d / np.sqrt(amc_au)
    secder = a2d2_3d / amc_au
    return tuple([firstder, secder])

def getPolarDers_log(logfile: str) -> tuple:
    """
    Polarizability derivatives: first order and second order
    Return: tuple[np.ndarray - shape(NM, 3, 3), np.ndarray - shape(NM, NM, 3, 3)]
    """
    pol = parse_polarizability(logfile)
    shpNM = int(pol.loc[pol[0] == 'P1', 1].max())
    p1_3d = np.zeros((shpNM, 3, 3))
    for i in pol.loc[pol[0] == 'P1', 1].unique():
        # in the rows with the current 'i' value and 'P' is 'P1', select columns 5, 6, and 7
        xyz = pol.loc[(pol[0] == 'P1') & (pol[1] == i), [5, 6, 7]].values
        p1_3d[int(i) - 1] = xyz

    nm_i = int(pol.loc[pol[0] == 'P2', 1].max())
    nm_j = int(pol.loc[pol[0] == 'P2', 2].max())

    p2_4d = np.zeros((nm_i, nm_j, 3, 3))

    for i in pol.loc[pol[0] == 'P2', 1].unique():
        for j in pol.loc[pol[0] == 'P2', 2].unique():
            # in the rows with the current 'i' and 'j' values and 'P' is 'P2', select columns 5, 6, and 7
            xyz = pol.loc[(pol[0] == 'P2') & (pol[1] == i) & (pol[2] == j), [5, 6, 7]].values
            # if 'xyz' is not empty
            if xyz.shape[0] != 0:
                # xyz is a 2D array
                p2_4d[int(i) - 1, int(j) - 1] = xyz
                p2_4d[int(j) - 1, int(i) - 1] = xyz

    return tuple([p1_3d, p2_4d])

def getPolarDers_au(logfile: str) -> tuple:
    from scipy import constants
    # to go from amu to au mass unit (m_e)
    amc_au = constants.physical_constants['atomic mass constant'][0] / \
             constants.physical_constants['atomic unit of mass'][0]

    p1_3d, p2_4d = getPolarDers_log(logfile)
    fdpol = p1_3d / np.sqrt(amc_au)
    sdpol = p2_4d / amc_au

    return tuple([fdpol, sdpol])

# used in retrievedata.py
def parse_cubic_constants(file_path: str) -> [pd.DataFrame, list]:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    results = []
    start = False
    start2 = False
    units_lines = []

    for line in lines:
        if "CUBIC FORCE CONSTANTS IN NORMAL MODES" in line:
            start = True
        elif line.strip().startswith("Num. of 3rd derivatives"):
            break
        elif start:
            if line.strip().startswith("I"):
                start2 = True
            elif start2 and line.strip() and not line.isspace():
                parts = line.split()
                results.append(parts)
            elif (line.strip().startswith(': FI =') or line.strip().startswith(': k  =')
                  or line.strip().startswith(': K  =')):
                units_lines.append(line.strip())
    df = pd.DataFrame(results, columns=["I", "J", "K", "FI(I,J,K)", "k(I,J,K)", "K(I,J,K)"])

    return df, units_lines

def parse_quartic_constants(file_path: str) -> [pd.DataFrame, list]:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    results = []
    start = False
    start2 = False
    units_lines = []

    for line in lines:
        if "QUARTIC FORCE CONSTANTS IN NORMAL MODES" in line:
            start = True
        elif line.strip().startswith("Num. of 4th derivatives"):
            break
        elif start:
            if line.strip().startswith("I"):
                start2 = True
            elif start2 and line.strip() and not line.isspace():
                parts = line.split()
                results.append(parts)
            elif (line.strip().startswith(': FI =') or line.strip().startswith(': k  =')
                  or line.strip().startswith(': K  =')):
                units_lines.append(line.strip())
    df = pd.DataFrame(results, columns=["I", "J", "K", "L", "FI(I,J,K,L)", "k(I,J,K,L)", "K(I,J,K,L)"])

    return df, units_lines

# used in retrievedata.py
def get_cubic_post(len_freq: int, cubic: np.ndarray):
    K3 = np.zeros((len_freq, len_freq, len_freq), dtype=np.float64)

    for fijk in cubic:
        i = int(fijk[0]) - 7
        j = int(fijk[1]) - 7
        k = int(fijk[2]) - 7
        d = np.float64(fijk[3])

        K3[i, j, k] = d
        K3[i, k, j] = d
        K3[k, j, i] = d
        K3[k, i, j] = d
        K3[j, i, k] = d
        K3[j, k, i] = d

    from scipy import constants
    # to go from amu to au mass unit (m_e)
    amc_au = constants.physical_constants['atomic mass constant'][0] / \
             constants.physical_constants['atomic unit of mass'][0]

    K3 = K3 / amc_au**1.5

    return K3

def get_quartic_post(len_freq: int, quartic: np.ndarray):
    K4 = np.zeros((len_freq, len_freq, len_freq, len_freq), dtype=np.float64)

    for fijkl in quartic:
        i = int(fijkl[0]) - 7
        j = int(fijkl[1]) - 7
        k = int(fijkl[2]) - 7
        l = int(fijkl[3]) - 7
        d = np.float64(fijkl[4])

        indices = [(i, j, k, l), (i, j, l, k), (i, k, j, l), (i, k, l, j),
                   (i, l, j, k), (i, l, k, j), (j, i, k, l), (j, i, l, k),
                   (j, k, i, l), (j, k, l, i), (j, l, i, k), (j, l, k, i),
                   (k, i, j, l), (k, i, l, j), (k, j, i, l), (k, j, l, i),
                   (k, l, i, j), (k, l, j, i), (l, i, j, k), (l, i, k, j),
                   (l, j, i, k), (l, j, k, i), (l, k, i, j), (l, k, j, i)]

        for idx in indices:
            K4[idx] = d

    from scipy import constants
    # to go from amu to au mass unit (m_e)
    amc_au = constants.physical_constants['atomic mass constant'][0] / \
             constants.physical_constants['atomic unit of mass'][0]

    K4 = K4 / amc_au**2

    return K4

# used in retrievedata.py
def parse_dipole_moment(file_path: str) -> (pd.DataFrame, str):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    results = []
    start = False
    units_line = None
    column_names = ["P", "i", "j", "k", "X", "Y", "Z"]
    last_ijk = [np.nan, np.nan, np.nan]  # last seen "i", "j", "k" values

    from scipy import constants
    bohr_radius = constants.physical_constants['Bohr radius'][0]
    debye_to_SI = 10 ** -21 / constants.c
    au_to_SI = constants.e * bohr_radius
    debye_to_au = debye_to_SI / au_to_SI

    for line in lines:
        if line.strip().startswith('Electric Dipole'):
            start = True
        elif line.strip().startswith("Polarizability Tensor"):
            break
        elif start:
            if line.strip().startswith("Unit of the property"):
                units_line = line.strip()
            elif line.strip().startswith("P"):
                # parts = re.split("[| ]+", line.strip())
                parts = line.split('|')
                allparts = [parts[0].strip()]
                # if "i", "j", "k" values are missing, use last seen values
                if parts[1].strip() == '':
                    allparts.extend(last_ijk)
                else:
                    ijk = parts[1].strip().split()
                    ijk.extend([np.nan] * (3 - len(ijk)))
                    allparts.extend(ijk)
                allparts.extend([float(s.replace('D', 'e'))*debye_to_au for s in parts[2].split()])
                row_dict = {column_names[i]: value for i, value in enumerate(allparts)}
                # missing columns to None
                row = [row_dict.get(column_name, np.nan) for column_name in column_names]
                results.append(row)

    df = pd.DataFrame(results, columns=column_names)

    return df, units_line

# used in retrievedata.py
def parse_polarizability(file_path: str) -> pd.DataFrame:
    with open(file_path, 'r') as file:
        lines = file.readlines()

    results = []
    start = False
    units_line = None
    column_names = ["P", "i", "j", "k", "comp", "X", "Y", "Z"]
    last_ijk = [np.nan, np.nan, np.nan]  # last seen "i", "j", "k" values

    for line in lines:
        if line.strip().startswith('Polarizability Tensor'):
            start = True
        elif line.strip().startswith("============================================") and start:
            break
        elif start:
            if line.strip().startswith("Unit of the property"):
                units_line = line.strip()
            elif line.strip().startswith("P"):
                # parts = re.split("[| ]+", line.strip())
                parts = line.split('|')
                allparts = [parts[0].strip()]
                # use last seen values if missing
                if parts[1].strip() == '':
                    allparts.extend(last_ijk)
                else:
                    ijk = [int(i) for i in parts[1].strip().split()]
                    ijk.extend([np.nan] * (3 - len(ijk)))
                    allparts.extend(ijk)

                allparts.extend([parts[2].strip()])

                if len(parts[3].strip().split()) == 3:
                    allparts.extend([float(s.replace('D', 'e')) for s in parts[3].split()])
                else:
                    xyz = [float(s.replace('D', 'e')) for s in parts[3].strip().split()]
                    xyz.extend([np.nan] * (3 - len(xyz)))
                    allparts.extend(xyz)
                row_dict = {column_names[i]: value for i, value in enumerate(allparts)}
                row = [row_dict.get(column_name, np.nan) for column_name in column_names]
                results.append(row)

            elif ('|  X  |' in line or '|  Z  |') and len(line.split('|')) == 4 and not 'i' in line:
                parts = line.split('|')
                allparts = [np.nan]
                allparts.extend([np.nan, np.nan, np.nan])
                allparts.extend([parts[2].strip()])
                xyz = [float(s.replace('D', 'e')) for s in parts[3].strip().split()]
                xyz.extend([np.nan] * (3 - len(xyz)))
                allparts.extend(xyz)
                row_dict = {column_names[i]: value for i, value in enumerate(allparts)}
                # missing columns to None
                row = [row_dict.get(column_name, np.nan) for column_name in column_names]
                results.append(row)
    df = pd.DataFrame(results, columns=column_names)
    array = df.to_numpy()

    for i in range(0, len(array), 3):
        # current 3-row block
        block = array[i:i + 3]
        p_value = block[1, 0]
        ival = block[1, 1]
        kval = block[1, 2]
        jval = block[1, 3]
        # replacing nan values in the first column of the block with the P value
        for j in range(3):
            try:
                if np.isnan(block[j, 0]):
                    block[j, 0] = p_value
                if np.isnan(block[j, 1]):
                    block[j, 1] = ival if np.isnan(ival) else int(ival)
                else:
                    block[j, 1] = int(block[j, 1])
                if np.isnan(block[j, 2]):
                    block[j, 2] = kval if np.isnan(kval) else int(kval)
                else:
                    block[j, 2] = int(block[j, 2])
                if np.isnan(block[j, 3]):
                    block[j, 3] = jval if np.isnan(jval) else int(jval)
                else:
                    block[j, 3] = block[j, 3] if np.isnan(block[j, 3]) else int(block[j, 3])
            except TypeError:
                continue
        if np.isnan(block[0, 6]):
            block[0, 6] = block[1, 5]
        if np.isnan(block[0, 7]):
            block[0, 7] = block[2, 5]
        if np.isnan(block[1, 7]):
            block[1, 7] = block[2, 6]

    df = pd.DataFrame(array)
    return df#, units_line
