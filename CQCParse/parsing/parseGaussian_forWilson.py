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

from CQCParse.debug import debugfunc
from scipy import constants
import numpy as np
# np.set_printoptions(linewidth=250, suppress=True, precision=3)
import sys

import logging
logger = logging.getLogger("CQCParse")

import pandas as pd
pd.set_option('display.max_rows', sys.maxsize)


def GHz2Nu(ghz: float | np.ndarray) -> float | np.ndarray:
    """Conversion from GHz to cm-1"""
    return ghz*10**9/(constants.c*100)

class GaussianDataParser(object):

    def __init__(self, all_files_dict: dict = None):
        """
        all_files_dict = {"files": {"3quanta": '', "log": ''}}
        """

        if all_files_dict is None:
            self.all_files_dict = {}
        else:
            # will make self.all_files_dict and execute some logic
            self.addFilesDict(all_files_dict=all_files_dict)
        
        # # 'log', 'fchk', 'com' - files types
        # if 'log' in self.all_files_dict['files']:
        #     # saving filename in 'fname' because 'log' will contain list of lines instead of the filename
        #     self.all_files_dict['files']['fname'] = self.all_files_dict['files']['log']

        # for filetype in self.all_files_dict['files']:
        #     if filetype=='log':
        #         with open(self.all_files_dict['files']['log'], 'r') as file:
        #             self.all_files_dict['files']['log'] = [i.strip() for i in file.readlines()]

        self.nModesStart = None

        files = self.all_files_dict.get('files')
        if files is not None:
            self.molecule = files.get('mol_code')
        else:
            self.molecule = None
        
        self.program = self.all_files_dict.get('source')

        self.dipgrad = None
        self.diphess = None
        self.polgrad = None
        self.polhess = None

        self.fundamentals_harmonic_str = None
        self.fundamentals_anharmonic_str = None

        self.fundamentals_harmonic_int = None
        self.fundamentals_anharmonic_int = None

        self.harmonic_states = None
        self.anharmonic_states = None
        self.cff = None
        self.qff = None

        self.cubic_cm_1 = None
        self.quartic_cm_1 = None
        self.B, self.coriolis = None, None

        self.equilibrium_geometry = None
        self.Q_normal_coordinates = None
        self.q_normal_coordinates_dimensionless = None

        self.atoms = None
        self.basis = None
        self.lot = None
        self.nmodes = None

        self.Xmatrices = None

        self.dir_prefix = ''


    def __dir__(self):
        return['nModesStart',
               'dipgrad', 'diphess',
               'polgrad', 'polhess',
               'fundamentals_harmonic_str', 'fundamentals_anharmonic_str',
               'fundamentals_harmonic_int', 'fundamentals_anharmonic_int',
               'harmonic_states', 'anharmonic_states',
               'cff', 'quartic_force_constants',
               'equilibrium_geometry',
               'Q_normal_coordinates', 'q_normal_coordinates_dimensionless',
               'atoms', 'basis', 'lot']

    def addFilesDict(self, all_files_dict: dict):
        """
        Upd/attach a new dict as self.all_files_dict (source files for data)
        """
        self.all_files_dict = all_files_dict

        fname = self.dir_prefix + self.all_files_dict['files']['log']

        # 'log', 'fchk', 'com' - files types
        if 'log' in self.all_files_dict['files']:
            # saving filename in 'fname' because 'log' will contain list of lines instead of the filename
            self.all_files_dict['files']['fname'] = fname

        for filetype in self.all_files_dict['files']:
            if filetype=='log':
                with open(fname, 'r') as file:
                    self.all_files_dict['files']['log'] = [i.strip() for i in file.readlines()]


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

        results_log = parse_frequencies(self.all_files_dict['files']['log'])
        self.number_atoms = parse_Na(self.all_files_dict['files']['log'])

        self.elements, self.coords = get_equil_geo(self.all_files_dict['files']['log'])

        self.fundamentals_anharmonic_int = {int(k)-1: float(v) for k, v in zip(results_log['Fundamental Bands']['mode_a'],
                                                                               results_log['Fundamental Bands'][2])}
        self.fundamentals_harmonic_int = {int(k)-1: float(v) for k, v in zip(results_log['Fundamental Bands']['mode_a'],
                                                                             results_log['Fundamental Bands'][1])}
        # logger.warning(f"self.fundamentals_anharmonic_int: {self.fundamentals_anharmonic_int}")

        self.nmodes = len(self.fundamentals_harmonic_int)

        self.fundamentals_harmonic_str = {str(k):v for k,v in self.fundamentals_harmonic_int.items()}
        self.fundamentals_anharmonic_str = {str(k):v for k,v in self.fundamentals_anharmonic_int.items()}
        self.hess = np.zeros((self.nmodes, self.nmodes))
        for i in self.fundamentals_anharmonic_int:
            self.hess[i,i] = self.fundamentals_anharmonic_int[i]

        ah_sts = get_allStates_fromParsedResults(results_log, anharmonic=True)
        h_sts = get_allStates_fromParsedResults(results_log, anharmonic=False)

        self.anharmonic_states = {tuple(str(i) for i in key): value for key, value in ah_sts.items()}
        self.harmonic_states = {tuple(str(i) for i in key): value for key, value in h_sts.items()}

        mu = getDipDers_au(self.all_files_dict['files']['log'])
        self.dipgrad = mu[0]
        self.diphess = mu[1]

        alpha = getPolarDers_au(self.all_files_dict['files']['log'])
        self.polgrad = alpha[0]
        self.polhess = alpha[1]

        cubic_df = parse_cubic_constants(self.all_files_dict['files']['log'])[0]
        cubic_rcm = cubic_df[['I', 'J', 'K', 'FI(I,J,K)']].to_numpy()
        selected_df1 = cubic_df[['I', 'J', 'K', 'K(I,J,K)']]
        cubic = selected_df1.to_numpy()

        quartic_df = parse_quartic_constants(self.all_files_dict['files']['log'])[0]
        quartic_rcm = quartic_df[['I', 'J', 'K', 'L', 'FI(I,J,K,L)']].to_numpy()
        selected_df2 = quartic_df[['I', 'J', 'K', 'L', 'K(I,J,K,L)']]
        quartic = selected_df2.to_numpy()

        self.cff_au = get_cubic_post(len(self.fundamentals_harmonic_str), cubic)
        self.qff_au = get_quartic_post(len(self.fundamentals_harmonic_str), quartic)
        self.cff = get_cubic_post(len(self.fundamentals_harmonic_int), cubic_rcm, reduced=False)
        self.qff = get_quartic_post(len(self.fundamentals_harmonic_int), quartic_rcm, reduced=False)

        self.B, self.coriolis = parse_coriolis(self.all_files_dict['files']['log'],
                                                                          len(self.fundamentals_harmonic_int))
        self.DD11 = ('No 1-1 Darling-Dennison resonance found' not in self.all_files_dict['files']['log']
                    and 'Search for 1-1 Darling-Dennison resonances deactivated' not in self.all_files_dict['files']['log'])
        self.DD13 = ('No 1-3 Darling-Dennison resonance found' not in self.all_files_dict['files']['log']
                    and 'Search for 1-3 Darling-Dennison resonances deactivated' not in self.all_files_dict['files']['log'])
        self.DD22 = ('No 2-2 Darling-Dennison resonance found' not in self.all_files_dict['files']['log']
                    and 'Search for 2-2 Darling-Dennison resonances deactivated' not in self.all_files_dict['files']['log'])

        self.Xmatrices = parse_anharmonic_x_matrix(self.all_files_dict['files']['log'])

        # modes = get_normal_modes(self.all_files_dict['files']['fname'], self.number_atoms)
        # rmodes = reordered_modes(self.all_files_dict['files']['fname'])
        modes = get_normal_modes(self.all_files_dict['files']['fname'], self.number_atoms)
        rmodes = reordered_modes(self.all_files_dict['files']['fname'])
        self.normal_modes = {}
        for i in rmodes:
            self.normal_modes[i-1] = modes[rmodes[i]-1]


def parse_coriolis(lines: list[str], nModes: int, linear_molecule: bool = False)-> tuple[np.ndarray, np.ndarray]:
    """
    returns:
        rotational_constant - shape (3,); coriolis_constant - shape (3, nmodes, nmodes)
    """
    corXtuples, corYtuples, corZtuples = [], [], []
    rotational_constant = []

    start1 = False
    start_rotcont = False

    for i, line in enumerate(lines):

        if 'CORIOLIS COUPLINGS' in line:
            start1 = True
        elif 'Num. of Coriolis couplings larger than' in line:
            start1 = False

        if start1 and len(line.strip().split())==4 and line.strip().split()[0]=='x':
            l1 = [int(line.strip().split()[1]), int(line.strip().split()[2]), float(line.strip().split()[3])]
            corXtuples.append(tuple(l1))

        elif start1 and len(line.strip().split()) == 4 and line.strip().split()[0] == 'y':
            l2 = [int(line.strip().split()[1]), int(line.strip().split()[2]), float(line.strip().split()[3])]
            corYtuples.append(tuple(l2))

        elif start1 and len(line.strip().split()) == 4 and line.strip().split()[0] == 'z':
            l3 = [int(line.strip().split()[1]), int(line.strip().split()[2]), float(line.strip().split()[3])]
            corZtuples.append(tuple(l3))

        if start_rotcont:
            if len(rotational_constant)<3:
                rotational_constant.append(line.strip().split()[1])
                if len(rotational_constant)==3:
                    start_rotcont = False
            else:
                continue

        if "Rotational Constants (in cm^-1)" in line:
            # The Ae line is 3 lines after the header
            ae_line = lines[i + 3]
            be_line = lines[i + 4]
            ce_line = lines[i + 5]
            
            # Extract the equilibrium values (first column after the label)
            ae = float(ae_line.strip().split('=')[1].strip().split()[0])
            be = float(be_line.strip().split('=')[1].strip().split()[0])
            ce = float(ce_line.strip().split('=')[1].strip().split()[0])

            rotational_constant = [ae, be, ce]

        rotconst_str = 'equilibrium (e), ground vibr.state (00), and 00 + centr. dist.(0)'

        if rotconst_str in line and len(rotational_constant)<3:
            start_rotcont = True

        if linear_molecule:
            if 'E(harm)  E(anharm)' in line:
                print('linear')
                rot_order = [line.strip().split()[-2][-2], line.strip().split()[-1][-2]]
                cart_lbls = ['x', 'y', 'z']
                lbls_here = [i for i in cart_lbls if i in rot_order]
                reorder = [rot_order.index(i) for i in lbls_here]

                next_line = lines[i+1]
                if 'Equilibrium Geometry' not in next_line:
                    raise ValueError("Cannot get rotational constant at Equilibrium Geometry from file")
                rotational_constant = [float(i) for i in next_line.split()[2:]]

                rotational_constant = np.array([float(rotational_constant[i]) for i in reorder])


                break
        else:
            if 'E(harm)  E(anharm)' in line:
                rot_order = [line.strip().split()[-3][-2], line.strip().split()[-2][-2], line.strip().split()[-1][-2]]
                reorder = [rot_order.index('x'), rot_order.index('y'), rot_order.index('z')]
                rotational_constant = np.array([float(rotational_constant[i]) for i in reorder])

                next_line = lines[i+1]
                if 'Equilibrium Geometry' not in next_line:
                    raise ValueError("Cannot get rotational constant at Equilibrium Geometry from file")
                # rot_vals = [float(i) for i in next_line.split()[2:]]
                break

    corXtuples, corYtuples, corZtuples = (tuple(item for item in corXtuples if item[0] !=0. ),
                                          tuple(item for item in corYtuples if item[0] !=0. ),
                                          tuple(item for item in corZtuples if item[0] !=0. ))

    corX, corY, corZ = np.zeros((nModes, nModes)), np.zeros((nModes, nModes)), np.zeros((nModes, nModes))

    for i, j, val in corXtuples:
        corX[i - 1, j - 1] = val
        corX[j - 1, i - 1] = val

    for i, j, val in corYtuples:
        corY[i - 1, j - 1] = val
        corY[j - 1, i - 1] = val

    for i, j, val in corZtuples:
        corZ[i - 1, j - 1] = val
        corZ[j - 1, i - 1] = val
    coriolis_constant = np.array([corX, corY, corZ])

    return rotational_constant, coriolis_constant

def parse_rot_const(lines, linear_molecule: bool = False):
    return

def parse_Na(lines: list[str]):

    ind = 0

    Na = 0
    for line in lines:
        ind += 1
        if 'Distance matrix (angstroms):' in line:
            Na = int(lines[ind-3].split()[0])
            break
    if Na == 0:
        debugfunc('Warning: Number of atoms not found in output', 'parseGaussian_forWilson.parse_Na')
    return Na


# used in retrievedata.py
def parse_frequencies(lines: list[str]) -> dict[str: pd.DataFrame]:
    if 'Combination Bands' in lines:
        sections = ["Fundamental Bands", "Overtones", "Combination Bands"]
    else:
        sections = ["Fundamental Bands", "Overtones"]

    results = {section: [] for section in sections}
    current_section = None

    primary_line = "Anharmonic Infrared Spectroscopy"
    alternative_line = "Vibrational Energies at Anharmonic Level"

    if any(primary_line in line for line in lines):
        target_line_anhram = primary_line
    else:
        target_line_anhram = alternative_line

    start = False
    units_counter = 0
    break_count = 0

    for line in lines:
        if target_line_anhram in line:
            start = True

        elif "Units: Transition energies" in line:
            units_counter += 1
            if units_counter == 2:
                start = False
        elif start:
            if any(section in line for section in sections):
                current_section = next(section for section in sections if section in line)
                print('\ncurrent_section', current_section)
            if current_section:
                if '------------' not in line:
                    linelist = [i for i in line.strip('LH').split() if i != 'active']
                    # inserting None at the desired index 2
                    # if len(linelist)==5 and current_section=='Combination Bands':
                    #     linelist.insert(2, None)
                    # if linelist and len(linelist) > 0:
                    results[current_section].append(linelist)

                if (current_section == 'Combination Bands' and '==========================' in line) or (current_section == 'Overtones' and 'Dipole strengths (DS) in' in line):
                    break
    
    # print(results['Combination Bands'])
    print('\nresults', results)

    results_dataframes = {}

    for section, data in results.items():
        if section != 'Overtones':
            results_dataframes[section] = pd.DataFrame(data[1:-1])
        else:
            results_dataframes[section] = pd.DataFrame(data[2:-1])
        results_dataframes[section].dropna(axis = 0, how = 'all', inplace = True)
        
        # logger.warning(f"results_dataframes[section]: \n{results_dataframes[section]}")

        if target_line_anhram == primary_line:
            index_num = 0
        
        elif target_line_anhram == alternative_line:
            index_num = 0
            # if len(results_dataframes[section].columns) == 7:
            #     index_num = 0
            # elif len(results_dataframes[section].columns) == 8:
            #     index_num = 1

        # logger.warning(f"section: {section}")
        # logger.warning(f"results_dataframes: {results_dataframes}")
        # logger.warning(f"results_dataframes[section]: \n{results_dataframes[section]}")
        try:
            print(results_dataframes[section])
            print('section', section)
            main_numbers = [i.split('(')[0] for i in results_dataframes[section][index_num]]
        except KeyError as e:
            print('section', section)
            logger.error(f"KeyError: {index_num} not found in results_dataframes[section] for section {section}")
            print(f"Available columns in results_dataframes[section]: {results_dataframes[section].columns}")
            print(results_dataframes[section])
            raise e
        
        print('\nsmth', [i[:-1].split('(')[1] for i in results_dataframes[section][index_num]])
        sub_numbers = [int(i[:-1].split('(')[1]) for i in results_dataframes[section][index_num]]

        # nserting columns at specific positions
        results_dataframes[section].insert(1, 'mode_a', main_numbers)
        results_dataframes[section].insert(2, 'n_a', sub_numbers)
        results_dataframes[section].drop(results_dataframes[section].columns[0], axis=1, inplace=True)



        if section=='Combination Bands':

            main_numbers_c = [int(i.split('(')[0]) if '(' in i else None for i in results_dataframes[section][2]]
            sub_numbers_c = [int(i[:-1].split('(')[1]) if '(' in i else None for i in results_dataframes[section][2]]
            for i, mn in enumerate(main_numbers_c):
                if mn is None:
                    # Shift values to the right and put None in column 2
                    row = results_dataframes[section].loc[i]
                    # Shift columns 2,3,4 to 3,4,5 
                    results_dataframes[section].loc[i, [5,4,3]] = row[[4,3,2]].values
                    # Set column 2 to None
                    results_dataframes[section].loc[i, 2] = None

            # logger.warning(f"Processing Combination Bands section: {section}")
            # logger.warning(f"results_dataframes[section]: \n{results_dataframes[section]}")

            main_numbers = [int(i.split('(')[0]) for i in results_dataframes[section][1]]
            sub_numbers = [int(i[:-1].split('(')[1]) for i in results_dataframes[section][1]]

            results_dataframes[section].insert(3, 'mode_b', main_numbers)
            results_dataframes[section].insert(4, 'n_b', sub_numbers)
            results_dataframes[section].drop(results_dataframes[section].columns[2], axis=1, inplace=True)

            # logger.warning(f"results_dataframes[section]: \n{results_dataframes[section]}")


            # logger.warning(f"main_numbers: {main_numbers}")
            # logger.warning(f"sub_numbers: {sub_numbers}")
            results_dataframes[section].insert(5, 'mode_c', main_numbers_c)
            results_dataframes[section].insert(6, 'n_c', sub_numbers_c)
            
            pd.set_option('display.max_rows', 500)
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)
         
            results_dataframes[section].drop(results_dataframes[section].columns[4], axis=1, inplace=True)
            # logger.warning(f"results_dataframes[section]: \n{results_dataframes[section]}")

    # logger.warning(f"results_dataframes: \n{results_dataframes}")

    return results_dataframes


def parse_frequencies_v2(lines: list[str]) -> dict:
    """A safer, non-destructive alternative to `parse_frequencies`.

    - Finds the anharmonic frequencies table and parses `Fundamental Bands`,
      `Overtones`, and (if present) `Combination Bands`.
    - Tolerant to varying spacing and scientific notation with D/E exponents.
    - Returns a dict of pandas.DataFrame objects keyed by section name.
    """
    import re
    sections = ["Fundamental Bands", "Overtones", "Combination Bands"]
    results = {s: [] for s in sections}

    primary_line = "Anharmonic Infrared Spectroscopy"
    alternative_line = "Vibrational Energies at Anharmonic Level"
    header_idx = None
    for i, L in enumerate(lines):
        if primary_line in L or alternative_line in L:
            header_idx = i
            break
    if header_idx is None:
        logger.debug('parse_frequencies_v2: header not found')
        return {}

    current_section = None
    stop_tokens = ('Dipole strengths (DS) in', '==========================')

    # regex for tokens like 1(1) or 12(3)
    mode_token = re.compile(r"(\d+)\((\d+)\)")
    # regex for numeric values (handles D/E notation)
    num_token = re.compile(r"[+-]?[0-9]*\.?[0-9]+(?:[DEde][+-]?\d+)?")

    for L in lines[header_idx:]:
        if any(tok in L for tok in stop_tokens):
            break
        # detect section headers
        for s in sections:
            if s in L:
                current_section = s
                break
        if current_section is None:
            continue

        # strip and skip separator lines
        ls = L.strip()
        if not ls or ls.startswith('-') or ls.startswith('Units:'):
            continue

        # find mode tokens (one or more)
        mt = mode_token.findall(ls)
        if not mt:
            continue

        # collect numeric values after last mode token
        # split at the point after the last ')' to get numbers part
        try:
            after = ls.split(')')[-1]
        except Exception:
            after = ls
        nums = num_token.findall(after)

        # Build row depending on section type
        if current_section == 'Fundamental Bands':
            # expect one mode token and up to 4 numbers: E(harm), E(anharm), I(harm), I(anharm)
            a_mode, a_n = mt[0]
            row = {
                'mode_a': a_mode,
                'n_a': a_n,
                'E_harm': float(nums[0]) if len(nums) > 0 else np.nan,
                'E_anharm': float(nums[1]) if len(nums) > 1 else np.nan,
                'I_harm': float(nums[2]) if len(nums) > 2 else np.nan,
                'I_anharm': float(nums[3]) if len(nums) > 3 else np.nan,
            }
            results[current_section].append(row)

        elif current_section == 'Overtones':
            # overtones often have mode like 1(2) then E(harm), E(anharm), I(anharm)
            a_mode, a_n = mt[0]
            row = {
                'mode_a': a_mode,
                'n_a': a_n,
                'E_harm': float(nums[0]) if len(nums) > 0 else np.nan,
                'E_anharm': float(nums[1]) if len(nums) > 1 else np.nan,
                'I_anharm': float(nums[2]) if len(nums) > 2 else np.nan,
            }
            results[current_section].append(row)

        else:  # Combination Bands
            # combination lines contain multiple mode tokens followed by numeric columns
            modes = [m[0] for m in mt]
            ns = [m[1] for m in mt]
            # Attempt to map remaining numbers to intensities/frequencies
            vals = [float(x.replace('D', 'E')) for x in nums] if nums else []
            row = {'mode_a': modes[0] if len(modes) > 0 else None,
                   'n_a': ns[0] if len(ns) > 0 else None,
                   'mode_b': modes[1] if len(modes) > 1 else None,
                   'n_b': ns[1] if len(ns) > 1 else None,
                   'mode_c': modes[2] if len(modes) > 2 else None,
                   'n_c': ns[2] if len(ns) > 2 else None,
                   'values': vals}
            results[current_section].append(row)

    # Convert lists to DataFrames with consistent columns
    dfs = {}
    for s, rows in results.items():
        if not rows:
            continue
        dfs[s] = pd.DataFrame(rows)

    return dfs

def get_allStates_fromParsedResults(results: pd.DataFrame, anharmonic: bool = False) -> dict:
    """results is a DataFrame from parse_frequencies()"""
    results['Combination Bands']['mode_c'] = results['Combination Bands']['mode_c'].fillna(0)
    results['Combination Bands']['n_c'] = results['Combination Bands']['n_c'].fillna(0)
    # print(results['Combination Bands'])

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
        
        # print(results['Combination Bands']['mode_a'], results['Combination Bands']['mode_b'],
        #         results['Combination Bands']['mode_c'])
        
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

def get_detected_resonances_g16(file_content: list[str]) -> list[str]:

    # with open(filepath, 'r') as file:
    #     file_content = file.read()

    if "Resonance Analysis" in file_content:
        # with open(filepath, 'r') as file:
        #     file_lines = file.readlines()
        file_lines = file_content
        found_resonances_str = []
        inFR = False
        for line in file_lines:
            if 'I      J  +   K' in line:
                inFR = True
                # col_names = line.strip().split()
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

def getDipDers_log(logfile: list[str]) -> tuple:
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

def getDipDers_au(logfile: list[str]) -> tuple:
    """
    1a.u. = 2.541746Debye
    """
    a2d, a2d2_3d = getDipDers_log(logfile)
    from scipy import constants
    # to go from amu to au mass unit (m_e)
    amc_au = constants.physical_constants['atomic mass constant'][0] / \
             constants.physical_constants['atomic unit of mass'][0]

    # transformation from Gaussian to Wilson units
    firstder = a2d / np.sqrt(amc_au)
    secder = a2d2_3d / amc_au
    return tuple([firstder, secder])

def getPolarDers_log(logfile: list[str]) -> tuple:
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

def getPolarDers_au(logfile: list[str]) -> tuple:
    from scipy import constants
    # to go from amu to au mass unit (m_e)
    amc_au = constants.physical_constants['atomic mass constant'][0] / \
             constants.physical_constants['atomic unit of mass'][0]

    p1_3d, p2_4d = getPolarDers_log(logfile)
    fdpol = p1_3d / np.sqrt(amc_au)
    sdpol = p2_4d / amc_au

    return tuple([fdpol, sdpol])

# used in retrievedata.py
def parse_cubic_constants(lines: list[str]) -> tuple[pd.DataFrame, list]:
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()

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
                # if "Sym.Forb." in line:
                #     parts.append("Sym.Forb.")
                if len(parts) == 6:
                    parts.append(None)
                results.append(parts)
            elif (line.strip().startswith(': FI =') or line.strip().startswith(': k  =')
                  or line.strip().startswith(': K  =')):
                units_lines.append(line.strip())
    df = pd.DataFrame(results, columns=["I", "J", "K", "FI(I,J,K)", "k(I,J,K)", "K(I,J,K)", "SymmetryForbidden"])

    return df, units_lines

def parse_quartic_constants(lines: list[str]) -> tuple[pd.DataFrame, list]:

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
                if len(parts) == 7:
                    parts.append(None)
                results.append(parts)
            elif (line.strip().startswith(': FI =') or line.strip().startswith(': k  =')
                  or line.strip().startswith(': K  =')):
                units_lines.append(line.strip())
    df = pd.DataFrame(results, columns=["I", "J", "K", "L", "FI(I,J,K,L)", "k(I,J,K,L)", "K(I,J,K,L)", "SymmetryForbidden"])

    return df, units_lines

# used in retrievedata.py
def get_cubic_post(len_freq: int, cubic: np.ndarray, reduced: bool = True):
    K3 = np.zeros((len_freq, len_freq, len_freq), dtype=np.float64)

    for fijk in cubic:
        i = int(fijk[0]) -1
        j = int(fijk[1]) -1
        k = int(fijk[2]) -1
        d = np.float64(fijk[3])

        # logger.warning(f"K3: {K3}")
        # logger.warning(f"fijk: {fijk}, i: {i}, j: {j}, k: {k}, d: {d}")
        # logger.warning(f"K3 shape: {K3.shape}")

        K3[i, j, k] = d
        K3[i, k, j] = d
        K3[k, j, i] = d
        K3[k, i, j] = d
        K3[j, i, k] = d
        K3[j, k, i] = d

    # from [Hartree*amu(-3/2)*Bohr(-3)] to [Hartree*m_e(-3/2)*Bohr(-3)]
    if reduced:
        from scipy import constants
        # to go from amu to au mass unit (m_e)
        amc_au = constants.physical_constants['atomic mass constant'][0] / \
                 constants.physical_constants['atomic unit of mass'][0]

        K3 = K3 / amc_au**1.5

    return K3

def get_quartic_post(len_freq: int, quartic: np.ndarray, reduced: bool = True):
    K4 = np.zeros((len_freq, len_freq, len_freq, len_freq), dtype=np.float64)

    for fijkl in quartic:
        i = int(fijkl[0]) - 1
        j = int(fijkl[1]) - 1
        k = int(fijkl[2]) - 1
        L = int(fijkl[3]) - 1

        d = np.float64(fijkl[4])

        indices = [(i, j, k, L), (i, j, L, k), (i, k, j, L), (i, k, L, j),
                   (i, L, j, k), (i, L, k, j), (j, i, k, L), (j, i, L, k),
                   (j, k, i, L), (j, k, L, i), (j, L, i, k), (j, L, k, i),
                   (k, i, j, L), (k, i, L, j), (k, j, i, L), (k, j, L, i),
                   (k, L, i, j), (k, L, j, i), (L, i, j, k), (L, i, k, j),
                   (L, j, i, k), (L, j, k, i), (L, k, i, j), (L, k, j, i)]

        for idx in indices:
            K4[idx] = d

    if reduced:
        from scipy import constants
        # to go from amu to au mass unit (m_e)
        amc_au = constants.physical_constants['atomic mass constant'][0] / \
                 constants.physical_constants['atomic unit of mass'][0]

        K4 = K4 / amc_au**2

    return K4

# used in retrievedata.py
def parse_dipole_moment(lines: list[str]) -> tuple[pd.DataFrame, str]:

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
        elif line.strip().startswith("Polarizability Tensor") or line.strip().startswith("Input for POLYMODE"):
            break
        elif start:
            if line.strip().startswith("Unit of the property"):
                units_line = line.strip()
            elif line.strip().startswith("P"):
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
def parse_polarizability(lines: list[str]) -> pd.DataFrame:

    results = []
    start = False
    # units_line = None
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

            elif ('|  X  |' in line or '|  Z  |') and len(line.split('|')) == 4 and 'i' not in line:
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


def remove_extra_g16():
    """

    """

# ------------------ Normal modes ----------------------------------
def nm_parse_lines(filename):
    """
    from freq output file returns lines with NM displacements
    """
    allines = []
    with open(filename) as infile:
        copy = False
        for line in infile:
            if line.strip() == "Atom  AN      X      Y      Z        X      Y      Z        X      Y      Z":
                copy = True
                continue
            elif line.startswith(" Frequencies --  "):
                copy = False
                continue

            elif line.startswith(" - Thermochemistry"):
                return allines

            if copy:
                allines.append(line)

    return allines


def nm_floats(filename):
    """
    from lines with NM displacements returns numbers (floats)
    """
    import re
    nums = []
    allines = nm_parse_lines(filename)

    for line in allines:
        x = re.findall(r"[-+]?\d*\.\d+|\d+", line)
        for i in x:
            try:
                int(i)
            except ValueError:
                nums.append(float(i))
    return nums


def get_normal_modes(filename, Na):
    """
    from freq output file returns 2D array (array of vectorised 3Na-6 NMs as rows of 3Na)
    """

    # Ntot = (3 * Na - 6) * Na
    m1 = np.reshape(np.array(nm_floats(filename)), (-1, Na, 3, 3))
    result = np.transpose(m1, (0, 2, 1, 3)).reshape(-1, 3, Na, 3)

    return result.reshape(-1, 3 * Na)


def normal_modes_prec(lines, Na, linear):

    collect = False
    count = 0
    nmodes = 3*Na-5 if linear else 3*Na-6

    lines_relevant = {i:[] for i in range(nmodes)}

    for line in lines:

        if 'Coord Atom Element:' in line:
            collect = True
        if ' Harmonic frequencies (cm**-1), IR intensities (KM/Mole), Raman scattering' in line:
            count+=1
            if count==2:
                # collect = False
                return
        if collect and len(line.strip().split())==1:
            collect = False

        if collect:
            for i, e in enumerate(line.strip().split()[3:]):
                if len(lines_relevant[i])<Na*3:
                    lines_relevant[i].append(e)
                else:
                    lines_relevant[i].append(e)



def nm_matrix_check(referenceFile, currentFile, Na, phase_change=False):
    # Normal modes lists from the files
    referenceNMs = get_normal_modes(referenceFile, Na)     # all normal modes from ref_file
    currentNMs = get_normal_modes(currentFile, Na)         # all normal modes from curr_file

    ####  Setting up the identity matrix
    mtx = np.zeros((3 * Na - 6, 3 * Na - 6))

    ###  Phase check for all NMs based oon the 1st one
    # for the 1st non-zero element (coordinate), check if the sign is the same
    curr0 = np.array(currentNMs[0]).reshape(-1, 3)
    ref0 = np.array(referenceNMs[0]).reshape(-1, 3)
    non_zero_ref = np.transpose(np.nonzero(ref0))
    non_zero_curr = np.transpose(np.nonzero(curr0))

    first_non_zero = None
    for i in non_zero_ref:
        for j in non_zero_curr:
            comparison = i == j
            if comparison.all():
                first_non_zero = i
                break
        else:
            continue
        break

    if first_non_zero is not None:
        s1 = ref0[first_non_zero[0], first_non_zero[1]]
        s2 = curr0[first_non_zero[0], first_non_zero[1]]
        if np.sign(s1) != np.sign(s2):
            phase_change = True
        else:
            phase_change = False

    for i in range(3 * Na - 6):
        for j in range(3 * Na - 6):
            # curr = np.array(currentNMs[j]).reshape(-1, 3)   # a column of the "identity" matrix
            ref = np.array(referenceNMs[i]).reshape(-1, 3)   # a row of the "identity" matrix

            ## Phase change switch
            phase_change = False
            # For coordinates swapping for normal mode (x->z, z->x), a manual inspection is needed
            if phase_change:
                ref *= -1

            #if i==14 and j==12:  # i from ref ; j from curr
                #print(ref)
                #print(curr)


            mtx[i][j] = np.dot(ref.flatten(), currentNMs[j]) / \
                        np.linalg.norm(ref.flatten())/np.linalg.norm(currentNMs[j])

    new_order = []
    for i in range(len(mtx[0])):
        match = np.where(abs(mtx[i]) > 0.9)
        if len(match[0]) == 0:
            match = np.where(abs(mtx[i]) > 0.8)
            if len(match[0]) == 0:
                match = np.where(abs(mtx[i]) > 0.6)
                if len(match[0]) == 0:
                    match = np.where(abs(mtx[i]) > 0.6)
                    new_order.append(match)
            else:
                new_order.append(match)
        else:
            new_order.append(match)
    new_order = [y for x in new_order for y in x]
    new_new = np.concatenate(new_order).ravel()
    if len(new_new) == len(mtx[0]):
        return mtx, new_new, True
    else:
        print("Incomplete")   # if the order number was not identified for every normal mode
        print()
        # df = pd.DataFrame(mtx)
        # print(df)
        return mtx, new_new, False

def reordered_modes(filepath):
    """
    {A:H}
    """
    with open(filepath, 'r') as file:
        file_content = file.readlines()

    collect = False
    H = []
    A = []
    for L in file_content:
        if L.strip() == '(H) is reported in the present equivalency table:':
            collect = True
        if L.strip() == 'Normal modes will be READ in ASCENDING order (imag. freq. first)':
            collect = False

        if collect and ('(H)' in L or '(A)' in L) and 'reported' not in L:
            L = L.strip().replace('|', '').split()
            if '(H)' in L:
                for e in L[1:]:
                    H.append(int(e))
            elif '(A)' in L:
                for e in L[1:]:
                    A.append(int(e))

    return dict(sorted(dict(zip(A,H)).items()))


def get_equil_geo(allines):

    opt = False
    collect = False

    elements = []
    dictelem = {'6': 'C', '7': 'N', '8': 'O', '1': 'H'}
    coords = []

    countlines = 0

    for line in allines:
        if 'Optimization completed.' in line.strip():
            opt = True
            continue

        if opt and len(line.strip().split())==1 and collect:
            countlines+=1
            continue

        if 'Standard orientation:' in line.strip() and opt:
            collect = True
            continue

        if 'Rotational constants (GHZ):' in line.strip() and collect:
            break

        if collect and len(line.strip().split())>2 and countlines==2:
            elements.append(dictelem[line.strip().split()[1]])
            coords.append([float(line.strip().split()[-3]),
                           float(line.strip().split()[-2]),
                           float(line.strip().split()[-1])])

    coords = np.array(coords)

    return elements, coords


def lower_triangular_to_full_matrix(lower_triangular_list):
    """
    Convert a lower triangular matrix stored as list of lists to a full symmetric matrix.
    
    Args:
        lower_triangular_list (list): List of lists where each inner list represents
                                    a row of the lower triangular matrix
    
    Returns:
        numpy.ndarray: Full symmetric matrix as 2D numpy array
    """
    if not lower_triangular_list:
        return np.array([])
    
    # Determine matrix size
    n = len(lower_triangular_list)
    
    # Initialize full matrix with zeros
    full_matrix = np.zeros((n, n))
    
    # Fill in the lower triangular part
    for i in range(n):
        row = lower_triangular_list[i]
        for j in range(len(row)):
            if j <= i:  # Only fill lower triangular part
                full_matrix[i, j] = row[j]
                # Mirror to upper triangular part (symmetric matrix)
                full_matrix[j, i] = row[j]
    
    return full_matrix

# Function to parse a matrix section
def parse_matrix_section(lines, start_search, end_search, section_title_pattern="contributions to X Matrix"):
    matrix_data = []
    section_start = None
    
    # Find the section start
    for i in range(start_search, len(lines)):
        if end_search in lines[i]:
            break
        if section_title_pattern in lines[i]:
            section_start = i
            break
    
    if section_start is None:
        return matrix_data
    
    # Skip header lines and find the actual matrix data
    i = section_start + 1
    while i < len(lines):
        line = lines[i].strip()
        
        # Stop if we hit another section or end with equals signs
        if ("contributions to X Matrix" in line or 
            "Anharmonic X Matrix" in line or
            "=" in line):
            break
        
        # Skip empty lines and lines that are just column headers
        if not line:
            i += 1
            continue
        
        parts = line.split()
        
        # Skip column header lines (lines with only numbers like "1 2 3 4 5")
        if len(parts) >= 2 and all(part.isdigit() for part in parts):
            i += 1
            continue
        
        # Skip lines that are just a single number (continuation column headers)
        if len(parts) == 1 and parts[0].isdigit():
            i += 1
            continue
        
        # Parse matrix row - first element should be row number
        if len(parts) >= 2:
            try:
                row_num = int(parts[0])
                values = []
                
                for j in range(1, len(parts)):
                    val_str = parts[j]
                    # Handle scientific notation format like 0.000000D+00
                    if 'D' in val_str:
                        val_str = val_str.replace('D', 'E')
                    values.append(float(val_str))
                
                # Ensure we have enough rows in matrix_data
                while len(matrix_data) < row_num:
                    matrix_data.append([])
                
                # Add values to the appropriate row
                matrix_data[row_num-1].extend(values)
            
            except (ValueError, IndexError):
                # Skip lines that can't be parsed as matrix data
                pass
        
        i += 1
    
    return matrix_data


def parse_anharmonic_x_matrix(file_lines):
    """
    Parse the Anharmonic X Matrix section from Gaussian output file lines.
    
    Args:
        file_lines (list): List of strings, each representing a line from the file
    
    Returns:
        dict: Dictionary containing parsed anharmonic X matrix data with keys:
            - 'coriolis_contributions': 2D list of coriolis contribution values
            - '3rd_deriv_contributions': 2D list of 3rd derivative contribution values  
            - '4th_deriv_contributions': 2D list of 4th derivative contribution values
            - 'total_anharmonic': 2D list of total anharmonic X matrix values
            - 'matrix_size': integer indicating the size of the matrices
    """
    
    result = {
        'coriolis_contributions': [],
        '3rd_deriv_contributions': [],
        '4th_deriv_contributions': [],
        'total_anharmonic': [],
        'matrix_size': 0
    }
    
    # Find the start of the Anharmonic X Matrix section
    start_idx = None
    for i, line in enumerate(file_lines):
        if "Anharmonic X Matrix" in line:
            start_idx = i+2
            break
    
    if start_idx is None:
        return result
    
    
    # Parse each matrix section
    search_start = start_idx
    
    # Parse Coriolis contributions
    coriolis_start = None
    for i in range(search_start, len(file_lines)):
        if "Coriolis contributions to X Matrix" in file_lines[i]:
            coriolis_start = i
            break
    
    if coriolis_start:
        k = parse_matrix_section(
            file_lines, coriolis_start, "3rd Deriv. contributions"
        )
        result['coriolis_contributions'] = lower_triangular_to_full_matrix(k)
    
    # Parse 3rd derivative contributions
    third_deriv_start = None
    for i in range(search_start, len(file_lines)):
        if "3rd Deriv. contributions to X Matrix" in file_lines[i]:
            third_deriv_start = i
            break
    
    if third_deriv_start:
        m = parse_matrix_section(
            file_lines, third_deriv_start, "4th Deriv. contributions"
        )
        result['3rd_deriv_contributions'] = lower_triangular_to_full_matrix(m)
    
    # Parse 4th derivative contributions  
    fourth_deriv_start = None
    for i in range(search_start, len(file_lines)):
        if "4th Deriv. contributions to X Matrix" in file_lines[i]:
            fourth_deriv_start = i
            break
    
    if fourth_deriv_start:
        n = parse_matrix_section(
            file_lines, fourth_deriv_start, "Total Anharmonic X Matrix"
        )
        result['4th_deriv_contributions'] = lower_triangular_to_full_matrix(n)
    
    # Parse total anharmonic matrix
    total_start = None  
    for i in range(search_start, len(file_lines)):
        if "Total Anharmonic X Matrix" in file_lines[i]:
            total_start = i
            break
    
    if total_start:
        p = parse_matrix_section(
            file_lines, total_start, "end_of_section", "Total Anharmonic X Matrix"
        )
        result['total_anharmonic'] = lower_triangular_to_full_matrix(p)
    
    result['matrix_size'] = result['total_anharmonic'].shape
    
    return result

def parse_mode_mapping(file_lines: list[str]) -> dict[int, int]:
    """
    Parse the A->H normal mode equivalency table from a full output file.

    CFOUR numbering most likely matches H numbering
    """
    h_vals = []
    a_vals = []
    inside = False
    for line in file_lines:
        stripped = line.strip()
        if "The connection between this new numbering" in stripped:
            inside = True
            continue
        if inside:
            if stripped.startswith("(H)"):
                h_vals.extend(int(x) for x in stripped.split("|")[1:] if x.strip())
            elif stripped.startswith("(A)"):
                a_vals.extend(int(x) for x in stripped.split("|")[1:] if x.strip())
            elif stripped.startswith("Normal modes will be READ"):
                break
    return dict(zip(a_vals, h_vals))
