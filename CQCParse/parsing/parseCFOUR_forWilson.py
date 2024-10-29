"""
########################################################################################################################
##                                                                                                                    ##
##                                             Parsing CFOUR output files                                             ##
##                                                                                                                    ##
########################################################################################################################
#
# Files:
#     - outfile0.out/outfile(n).out  --- the main full output file:
#                            1)
#     - MOLDEN                       --- from starting anharmonic parallel run:
#                            1) equilibrium geometry (a.u.) ; 2) normal coordinates, non-mass-weighted (a.u.)
#     - NORMCO                       --- mass-weighted coordinates: equilibrium and normal coordinates, and frequencies
#     - FCMFINAL                     --- non-mass-weighted Hessian matrix in columns (xyz)
#     - DIPOL                        --- dipole moment (a.u.)
#     - DIPDER                       --- dipole moment first order derivatives (cartesian)
#     - POLAR                        --- static polarizability (a.u.)
#     - out                          --- the final output file in anharmonic parallel procedure:
#                            1) All levels with up to three quanta frequencies; 2) equilibrium geometry;
#                            3) normal coordinates, non-mass-weighted (a.u.); 4) F(IJKK)/a.u ; 5) F(IJKK)/cm-1 ;
#                            6) harmonic and fundamental frequencies and intensities
#     - dipolex(yz)                  --- dipole moment (1st-2nd-3rd) order derivatives (normal coordinates)
#     - cubic                        --- cubic force constants in cm-1 in dimensionless normal modes
#     - quartic                      --- quartic force constants in cm-1 in dimensionless normal modes
"""

import numpy as np
import os
import pickle


class CFOURdataParser:
    """A class that contains parsed CFOUR output data"""
    def __init__(self, all_files_dict):
        self.all_files_dict = all_files_dict
        # {'outfile_anharm_start', 'out_anharm_end', 'molden', 'dipolexyz',
        #  'normco', 'quadrature', 'polar', 'dipder', 'dipol', 'cubic', 'fcmfinal'
        #  ''}
        self.nModesStart = None

        self.dipole_first_derivatives = None
        self.dipole_second_derivatives = None
        self.polarizability_first_derivatives = None
        self.polarizability_second_derivatives = None

        self.fundamentals_harmonic_str = None
        self.fundamentals_anharmonic_str = None
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
        # {'outfile_anharm_start', 'out_anharm_final', 'molden', 'dipolexyz',
        #  'normco', 'quadrature', 'polar', 'dipder', 'dipol', 'cubic', 'fcmfinal'
        #  ''}
        self.nModesStart = 6 if linear_molecule else 7

        parsed_data = parse_output_file(self.all_files_dict['files']['out_anharm_final'])
        vib_energy_levels_list, labelsTable, anharmonic_freqs, anharmonic_ints, harmonic_freqs = parsed_data

        anharm_states_dict = dict(zip(vib_energy_levels_list, anharmonic_freqs))
        anharm_tuple_dict = {tuple([o - 7 for o in k]): v for k, v in anharm_states_dict.items()}
        harm_states_dict = dict(zip(vib_energy_levels_list, harmonic_freqs))
        harm_tuple_dict = {tuple([o - 7 for o in k]): v for k, v in harm_states_dict.items()}

        self.fundamentals_harmonic_str = {str(k[0]):v for k,v in harm_tuple_dict.items() if len(k)==1}
        self.fundamentals_anharmonic_str = {str(k[0]):v for k,v in anharm_tuple_dict.items() if len(k)==1}

        self.anharmonic_states = {tuple(str(i -7) for i in k): v for k, v in anharm_states_dict.items()}
        self.harmonic_states = {tuple(str(i-7) for i in k): v for k, v in harm_states_dict.items()}

        cubic = pCubicORQuartic(self.all_files_dict['files']['cubic'])
        self.funds_harm_ints = {int(k): v for k, v in self.fundamentals_harmonic_str.items()}
        # transformed to Wilson units in getCubicPost
        self.cubic_force_constants = getCubicPost(self.funds_harm_ints, cubic)

        labelsModes_original = [i+self.nModesStart for i in list(self.funds_harm_ints)]
        mu = getDipoleDers_anharm_au(self.all_files_dict['files']['dipolexyz'], labelsModes_original, self.nModesStart,
                                     self.fundamentals_harmonic_str)
        self.dipole_first_derivatives = mu[0]
        self.dipole_second_derivatives = mu[1]

        alpha = getPolarDers_pkl_au(self.all_files_dict['files']['polar_pkl'], self.fundamentals_harmonic_str)
        self.polarizability_first_derivatives = alpha[0]
        self.polarizability_second_derivatives = alpha[1]


# used for things
def parse_output_file(filepath: str):
    """
    Parsing the out file - output of the anharmonic parallel procedure
    :param filepath:
    :return: modes - 1, 2, 3 quanta states - combinations of states
             anharmonic_frequencies
             anharmonic_intensities
             harmonic_transitions
    """
    modes = []
    anharmonic_frequencies = []
    anharmonic_intensities = []
    harmonic_transitions = []

    with open(filepath, 'r') as file:
        file_content = file.read()

    if "All levels with up to three quanta" in file_content:
        start_index = file_content.find("All levels with up to three quanta")

        end_index = file_content.find("Electric dipole moment function in dimensionless normal coordinates",
                                              start_index)

        section = file_content[start_index:end_index]
        section = section.split('\n')
        for line in section:
            if ('-------------' not in line and 'Dipole' not in line
                    and 'quanta' not in line and 'MODE' not in line
                    and 'Intensity' not in line and line.strip()!=''):
                nus = line.strip().split()
                m = tuple([int(f) for f in nus[:-3]])
                modes.append(m)
                anharmonic_frequencies.append(float(nus[-3]))
                anharmonic_intensities.append(float(nus[-2]))
                harmonic_transitions.append(float(nus[-1]))

        modes = np.array(modes)
        anharmonic_frequencies = np.array(anharmonic_frequencies)
        anharmonic_intensities = np.array(anharmonic_intensities)
        harmonic_transitions = np.array(harmonic_transitions)

        labels = np.array('   I    J    K    L    M   NI  NJ  NK  NL  NM '.strip().split())
        modes = [modes_array2tuple(arr) for arr in modes]
        return modes, labels, anharmonic_frequencies, anharmonic_intensities, harmonic_transitions

    else:
        print('\nNo anharmonic levels information in this file')

# used for polarizability derivatives calculations
def pTensor(filepath: str):
    """
    Parsing DIPOL or POLAR or FCMFINAL files

    :return: np.ndarray tensor of dipole moment or polarizability
    """
    alines = []

    with open(filepath, 'r') as file:
        lines = file.readlines()
        start = 1 if len(lines[0].strip().split())==2 else 0
        alines.extend([np.array([float(k) for k in i.strip().split()]) for i in lines[start:]])

    return np.array(alines) #.reshape((-1, 3 * na))

def getRotationMatrix(filepath: str) -> np.array:
    """
    Parsing outfile0.out to get the rotation matrix for xyz axes
    :param filepath:
    :return: OMAT transformation (rotation) matrix from outfile which is printed when PRINT_LEVEL=1
    """
    with open(filepath, 'r') as file:
        file_content = file.read()
    if 'Transformation matrix between QCOM and QCOMP (OMAT)' in file_content:
        start_index = file_content.find("(QCOM = OMAT * QCOMP,")
        end_index = file_content.find("test:  OMAT * QCOMP = ", start_index)
        omat_section = file_content[start_index:end_index]
        newlinebreak = omat_section.find("\n")
        coords_section = omat_section[newlinebreak:][1:-1].split('\n')

        omat_array = []
        for l in coords_section[:-1]:
            ll = np.array([float(i) for i in l.strip().split() if '.' in i])
            omat_array.append(ll)
        omat_array = np.array(omat_array)

        return omat_array
    else:
        print('No rotation matrix found in this outfile')

def get_detected_resonances_c4(filepath: str):

    with open(filepath, 'r') as file:
        file_content = file.read()

    if "Thresholds for removing resonance denominators:" in file_content:
        with open(filepath, 'r') as file:
            file_lines = file.readlines()
        found_resonances_str = []
        for line in file_lines:
            if 'Resonance between' in line and 'combination' in line:
                found_resonances_str.append(line)
        if found_resonances_str:
            found_resonances_str.append(f'There are {len(found_resonances_str)} Fermi resonances\n')
        return found_resonances_str

# used
def pCubicORQuartic(filepath: str):
    """
    Parising cubic and quartic files with CFFs and QFFs
    :param filepath:
    :return: np.ndarray - parsed lines of files
    """
    alllines = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
    for l in lines:
        elements = l.split()
        alllines.append(np.array([int(k) if i < len(elements) - 1 else float(k) for i, k in enumerate(elements)]))

    return np.array(alllines)

# used
def getCubicPost(freq: dict, cubic: np.ndarray, recipcm: bool = False):
    """ Derives cubic and quartic anharmonic constants.
        It takes reduced values [cm-1] from gaussian output
        and transforms it to :
          * cubic   force constants : [Hartree*amu(-3/2)*Bohr(-3)]
          * quartic force constants : [Hartree*amu(-2  )*Bohr(-4)]
    """
    if recipcm:

        n = len(freq)
        K3 = np.zeros((n, n, n), dtype=np.float64)

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

        return K3

    else:
        n = len(freq)
        K3 = np.zeros((n, n, n), dtype=np.float64)

        for fijk in cubic:

            i = int(fijk[0]) - 7
            j = int(fijk[1]) - 7
            k = int(fijk[2]) - 7
            d = np.float64(fijk[3])

            d *= np.sqrt(freq[i] * freq[j] * freq[k])

            from scipy import constants
            a = np.sqrt(constants.h / constants.c / constants.physical_constants['unified atomic mass unit'][0] / 100)
            b = 10 ** 10 / 2 / np.pi / constants.physical_constants['Bohr radius'][0] / 10 ** 10
            Fact3R = (constants.physical_constants['hartree-joule relationship'][
                          0] / constants.h / constants.c / 100) * (a * b) ** 3

            d /= Fact3R

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
        # to Wilson units
        K3 /= amc_au ** 1.5
        return K3

# used in getDipoleDers_anharm
def pDipole(filenamebase: str):
    """
    Parsing dipole(xyz) files dipole(xyz)
    :param filenamebase: basename for dipole(xyz) files, e.g, 'dipole'
    :return: dictionary from dipole(xyz) data
    """
    dct = {}
    with open(filenamebase, 'r') as file:
        lines = file.readlines()
    for l in lines:
        nu = np.array([float(k) for k in l.split()])
        indx = tuple([int(i) for i in nu[:-1] if i!=0.0])
        if len(indx) == 1:
            dct[indx[0]] = nu[-1]
        else:
            dct[indx] = nu[-1]

    return dct

def getDipoleDers_anharm(filenamebase: str, labels: list, nModesStart: int):
    """
    Getting dipole moment derivatives tensors
    :param nModesStart:
    :param labels:
    :param filenamebase: basename for dipole(xyz) files, e.g, 'dipole'
    :return:  dmudqarray - first order derivatives of dipole moment, (3N-6, 3)
              dmudqdarray - second order derivatives of dipole moment, (3N-6, 3N-6, 3)
    """
    dipx = pDipole(filenamebase+'x')
    dipy = pDipole(filenamebase+'y')
    dipz = pDipole(filenamebase+'z')
    # dipx
    # {7: 0.0097191434, (7, 9): -0.0024929592, (7, 10): 0.0068563719,
    # (7, 11): 0.0006583914, (9, 7): -0.0024929355,
    # (11, 11, 7): -0.0004000706, (12, 12, 7): -0.0007561331}

    dq = len(labels)
    dmudq_array = np.zeros((dq, 3))
    dmudqdq_array = np.zeros((dq, dq, 3))

    for l in labels:
        if type(l) ==int:
            if l in dipx:
                dmudq_array[l-nModesStart, 0] = dipx[l]
            if l in dipy:
                dmudq_array[l-nModesStart, 1] = dipy[l]
            if l in dipz:
                dmudq_array[l-nModesStart, 2] = dipz[l]

    for l in dipx:
        if type(l) != int:
            if len(l) == 2:
                dmudqdq_array[(l[0] - nModesStart, l[1] - nModesStart, 0)] = dipx[l]

    for l in dipy:
        if type(l) != int:
            if len(l) == 2:
                dmudqdq_array[(l[0] - nModesStart, l[1] - nModesStart, 1)] = dipy[l]

    for l in dipz:
        if type(l) != int:
            if len(l) == 2:
                dmudqdq_array[(l[0] - nModesStart, l[1] - nModesStart, 2)] = dipz[l]

    return dmudq_array, dmudqdq_array

def getDipoleDers_anharm_au(filenamebase: str, labels: list, nModesStart: int, fundamentals_harmonic: dict) -> tuple:
    firstder, secder = getDipoleDers_anharm(filenamebase, labels, nModesStart)
    from wilson.spectrum2d.spectrum import rec_cm2rec_s
    w_h = rec_cm2rec_s(np.array([v for k, v in fundamentals_harmonic.items()]))
    matrix_2d = np.outer(w_h, w_h)
    # prefac_3d = w_h[:, np.newaxis, np.newaxis] * w_h[np.newaxis, :, np.newaxis] * w_h[np.newaxis,
    #                                                                                    np.newaxis, :]
    sqrtvec = 1. / np.sqrt(w_h)
    sqrtmat = 1. / np.sqrt(matrix_2d.T)
    # sqrt3d = 1. / np.sqrt(prefac_3d.T)

    firstder_mat = np.zeros_like(firstder)
    for i in range(len(sqrtvec)):
        for j in range(3):
            firstder_mat[i, j] = firstder[i, j] / sqrtvec[i]

    secder_mat = np.zeros_like(secder)
    for i in range(len(sqrtvec)):
        for j in range(len(sqrtvec)):
            for k in range(3):
                secder_mat[i, j, k] = secder[i, j, k] / sqrtmat[i, j]

    return tuple([firstder_mat, secder_mat])


def getPolarDers_pkl_au(polar_pkl_file: str, fundamentals_harmonic: dict):
    from wilson.spectrum2d.spectrum import rec_cm2rec_s
    w_h = rec_cm2rec_s(np.array([v for k, v in fundamentals_harmonic.items()]))
    matrix_2d = np.outer(w_h, w_h)
    sqrtvec = 1. / np.sqrt(w_h)
    sqrtmat = 1. / np.sqrt(matrix_2d.T)

    with open(polar_pkl_file, 'rb') as file:
        alpha = pickle.load(file)
    polarizability_first_derivatives = alpha[0]
    polarizability_second_derivatives = alpha[1]

    fdpol = np.zeros_like(polarizability_first_derivatives)
    for i in range(len(sqrtvec)):
        for j in range(3):
            for k in range(3):
                fdpol[i, j, k] = polarizability_first_derivatives[i, j, k] / sqrtvec[i]

    sdpol = np.zeros_like(polarizability_second_derivatives)
    for i in range(len(sqrtvec)):
        for j in range(len(sqrtvec)):
            # with open('./secPolder', 'a') as file1:
            #     file1.write(f'\n=============================={i} {j}\n{sqrtmat[i, j]}\n')
            #     file1.writelines(str(polarizability_second_derivatives[i, j, :, :]))

            for k in range(3):
                for l in range(3):
                    sdpol[i, j, k, l] = polarizability_second_derivatives[i, j, k, l] / sqrtmat[i, j]

    return tuple([fdpol, sdpol])


# used
def getDisplacementsPolarData(polar_dir: str, raw: bool = False):
    """
    Collecting the POLAR files data and also rotates is according to the matrix from the corresponding outfile0.out
    :param raw:
    :param polar_dir: where polar calcs were done

    :return:
    """
    import os
    directories = [d for d in os.listdir(polar_dir) if os.path.isdir(os.path.join(polar_dir, d))]
    allpolardata = {}
    allpolardataRaw = {}

    for directory in directories:
        dir_path = os.path.join(polar_dir, directory)

        polar_file_path = dir_path + '/POLAR'
        poldata = pTensor(polar_file_path)
        R = getRotationMatrix(dir_path + '/outfile0.out')

        allpolardataRaw[directory] = (poldata, R)
        # First, we transform one rank: temp = R α
        temp = np.einsum('ij,jk->ik', R.T, poldata)
        # Then, we transform the other rank: α' = temp R^T
        alpha_prime = np.einsum('ij,jk->ik', temp, R)

        # 'ij, j -> i'
        # 'ij, jk -> ik', 'nk, ik -> in'
        allpolardata[directory] = alpha_prime

    if raw:
        return allpolardataRaw
    else:
        return allpolardata

# used
def getPolarDers(polar_dir: str):
    """
    Computing polarizability derivatives
    :param polar_dir: where optimization was done
    :return:
    """
    # base_dir = 'equil/displacements'

    data = getDisplacementsPolarData(polar_dir)
    data['equil'] = pTensor(polar_dir + '/../anharm/POLAR')
    print(f'Equilibrium polarizability in {polar_dir + "/../anharm/POLAR"}')
    print(sorted(list(data.keys())))

    import re
    # Extract all numbers from directory names
    numbers = set()
    pattern = re.compile(r'\d+')
    for name in list(data.keys()):
        numbers.update(map(int, pattern.findall(name)))

    min_num = min(numbers)
    max_num = max(numbers)
    size = max_num - min_num + 1

    import os
    directories = [d for d in os.listdir(polar_dir) if os.path.isdir(os.path.join(polar_dir, d))]
    nums = [h.strip('np').split('_') for h in directories]
    flattened_list = [item for sublist in nums for item in sublist]
    flattened_list = set([int(g) for g in flattened_list])

    firstder = {}

    for f in flattened_list:
        firstder[f] = (data[f'{f}p'] - data[f'{f}n']) / 0.02

    secondders = {}

    # (∂²α_ij/∂Q_k∂Q_l) ≈ [α_ij(Q_k + ΔQ_k, Q_l + ΔQ_l) - α_ij(Q_k + ΔQ_k, Q_l - ΔQ_l)
    #   - α_ij(Q_k - ΔQ_k, Q_l + ΔQ_l) + α_ij(Q_k - ΔQ_k, Q_l - ΔQ_l)] / (4 * ΔQ_k * ΔQ_l)
    for k in flattened_list:
        for m in flattened_list:
            if k < m:
                val = (data[f'{k}_{m}pp'] - data[f'{k}_{m}pn'] - data[f'{k}_{m}np'] + data[f'{k}_{m}nn']) / (
                            4 * 0.01 * 0.01)
                secondders[(k, m)] = val
                secondders[(m, k)] = val

    # (∂²α_ij/∂Q_k²) ≈ [α_ij(Q_k + ΔQ_k) - 2α_ij(Q_k) + α_ij(Q_k - ΔQ_k)] / (ΔQ_k)²
    for b in flattened_list:
        secondders[(b, b)] = (data[f'{b}p'] - 2 * data['equil'] + data[f'{b}n']) / 0.01 ** 2

    polder = []
    for p in firstder:
        polder.append(firstder[p])
    first = np.array(polder)

    second = np.zeros((size, size, 3, 3))
    indices_to_insert = list(secondders.keys())
    # Insert the matrices at the specified indices
    for index, mat in zip(indices_to_insert, list(secondders.values())):
        i, j = index
        second[i - 7, j - 7] = mat

    return first, second

# not used now
def pklPolder(polar_dir):
    first, second = getPolarDers(polar_dir)
    filenamec = 'polar.pkl'

    with open(filenamec, 'wb') as file:
        pickle.dump((first, second), file)

    print(f'\n --> {filenamec} file created\n')

    return filenamec

# not used now
def pklPoldata(polar_dir):
    allpolardata = getDisplacementsPolarData(polar_dir)
    rawdata = getDisplacementsPolarData(polar_dir, raw=True)

    poldataeq = pTensor(polar_dir + '/../anharm/POLAR')
    R = getRotationMatrix(polar_dir + '/../anharm/outfile0.out')
    temp = np.einsum('ij,jk->ik', R.T, poldataeq)
    alpha_prime = np.einsum('ij,jk->ik', temp, R)
    allpolardata['equil'] = alpha_prime

    rawdata['equil'] = (alpha_prime, R)
    filenamec = 'polarData.pkl'

    with open(filenamec, 'wb') as file:
        pickle.dump(allpolardata, file)

    print(f'\n --> {filenamec} file created\n')

    filenamec2 = 'polarData_raw.pkl'

    with open(filenamec2, 'wb') as file:
        pickle.dump(rawdata, file)

    print(f'\n --> {filenamec2} file created\n')

    return filenamec

# used for polarizability calculations (in pklDimless_normal_modes,)
def pQUADRATURE(filepath) -> tuple[np.ndarray, np.array, dict[int: np.ndarray]]:
    """Dimensionless normal coordinates are here, in QUADRATURE file"""
    with open(filepath, 'r') as file:
        lines = file.readlines()

    current_frequency = None
    current_matrix = []
    undisplaced_matrix = []
    reading_matrix = False
    dqMat = []
    freqs = []

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('% frequency'):
            # If there's a current frequency, it means we've reached a new block
            if current_frequency is not None:
                freqs.append(current_frequency)
                dqMat.append(np.concatenate(current_matrix).reshape(-1, 3))
                current_matrix = []
            # Extract the frequency value
            current_frequency = float(lines[i + 1].strip())
        elif line.startswith('% back-transformed dimensionless normal coordinates'):
            reading_matrix = True
        elif line.startswith('% Reference (undisplaced) coordinates are:'):
            # Save the last frequency block before moving to undisplaced coordinates
            if current_frequency is not None:
                freqs.append(current_frequency)
                dqMat.append(np.concatenate(current_matrix).reshape(-1, 3))
                current_frequency = None
                current_matrix = []
            reading_matrix = True
        elif reading_matrix and line:
            # Extract the coordinates
            coords = [float(x) for x in line.split()]
            if len(tuple(coords))==3:
                current_matrix.append(np.array(coords))
        elif not line:
            reading_matrix = False

    # Add the last frequency block if it wasn't added
    if current_frequency is not None:
        freqs.append(current_frequency)
        dqMat.append(np.concatenate(current_matrix).reshape(-1, 3))

    # The last matrix read is the undisplaced matrix
    if current_matrix:
        undisplaced_matrix = current_matrix
    # normal_coordinates = np.vstack(dqMat).T

    normal_coordinates = dict(zip(np.arange(7, len(freqs)+7), dqMat))
    return np.array(undisplaced_matrix), freqs, normal_coordinates

# not used now
def pklDimless_normal_modes(quadratureFile):
    equilibrium_geometry, freqs, normal_modes = pQUADRATURE(quadratureFile)
    filenamec = 'dimensionless.pkl'

    with open(filenamec, 'wb') as file:
        pickle.dump((equilibrium_geometry, freqs, normal_modes), file)

    print(f'\n --> {filenamec} file created\n')

    return filenamec


"""
1-parse_output_file,
2-pCubicORQuartic, 3-getCubicPost,
4-getDipoleDers_anharm,
'polar_pkl' file <- 5-getPolarDers(6-getDisplacementsPolarData, 
                    7-getRotationMatrix, 8-pTensor), 
                    9-pklPolder"""
# -----------------------------------------------------------------------------
# not used now

# used in computeRedMass4nm which is not used now; also used for making displacements (polar)
def pMOLDEN(filepath: str, linear: bool = False) -> tuple[np.ndarray, np.array, dict[int: np.ndarray]]:
    """
    Parsing MOLDEN file (VIB job output, harmonic or anharmonic)
    :param linear:
    :param filepath:
    :return:  geometry_data, atoms, selected_dict
    """
    with open(filepath, 'r') as file:
        lines = file.readlines()

    atoms = []
    geometry_data = []
    vibrations_data = {}

    in_geometry_section = False
    in_vibration_section = False

    for line in lines:
        if '[FR-COORD]' in line:
            in_geometry_section = True
            in_vibration_section = False
            continue

        elif '[FR-NORM-COORD]' in line:
            in_vibration_section = True
            in_geometry_section = False
            continue

        if in_geometry_section:
            data = line.strip().split()
            atom_label = data[0]
            atoms.append(atom_label)
            geometry_data.append(data[1:])  # Exclude the atom label

        # Capture vibration data
        elif in_vibration_section:
            stripped_line = line.strip()
            line_parts = stripped_line.split()

            if stripped_line.startswith('vibration'):
                current_vibration_index = int(line_parts[-1])
                vibrations_data[current_vibration_index] = []
            else:
                # to check if it was initialized
                if 'current_vibration_index' in locals():
                    vibrations_data[current_vibration_index].append(list(map(float, line_parts)))
                else:
                    print('Error: Vibration number not initialized before data lines.')

    atoms = np.array(atoms)
    geometry_data = np.array(geometry_data, dtype=float)
    vibrations_data = {key: np.array(value) for key, value in vibrations_data.items()}
    # only normal modes, ignore first 6 or 5
    ndiscard = 5 if linear else 6
    normal_modes_dict = {key: value for idx, (key, value) in enumerate(vibrations_data.items()) if idx >= ndiscard}

    return geometry_data, atoms, normal_modes_dict


def pklOutFile(outfile):
    import pickle

    data = parse_output_file(outfile)
    filenamedata = 'vibdata.pkl'

    with open(filenamedata, 'wb') as file:
        pickle.dump(data, file)

    print(f'\n --> {filenamedata} file created\n')
    return os.getcwd()+'/'+filenamedata


# used in pklDipole
def getDipoleDers(filenamebase: str, outfile: str):
    """
    Getting dipole moment derivatives tensors
    :param filenamebase: basename for dipole(xyz) files, e.g, 'dipole'
    :param outfile: out file - output of the anharmonic parallel procedure
    :return:  dmudqarray - first order derivatives of dipole moment, (3N-6, 3)
              dmudqdarray - second order derivatives of dipole moment, (3N-6, 3N-6, 3)
    """
    dipx = pDipole(filenamebase + 'x')
    dipy = pDipole(filenamebase + 'y')
    dipz = pDipole(filenamebase + 'z')

    things = parse_output_file(outfile)

    labels = sorted(list(set([t[0] for t in things[0]])))
    labels = [int(k) for k in labels]

    dmudqdict = {}
    dmudqdqdict = {}
    dq = len(labels)
    dmudqdqdict['x'] = np.zeros((dq, dq))
    dmudqdqdict['y'] = np.zeros((dq, dq))
    dmudqdqdict['z'] = np.zeros((dq, dq))

    for l in labels:
        if type(l) == int:
            dmudqdict[l] = np.zeros(3)
            if l in dipx:
                dmudqdict[l][0] = dipx[l]
            if l in dipy:
                dmudqdict[l][1] = dipy[l]
            if l in dipz:
                dmudqdict[l][2] = dipz[l]

    for l in dipx:
        if type(l) != int:
            if len(l) == 2:
                dmudqdqdict['x'][(l[0] - 7, l[1] - 7)] = dipx[l]

    for l in dipy:
        if type(l) != int:
            if len(l) == 2:
                dmudqdqdict['y'][(l[0] - 7, l[1] - 7)] = dipy[l]

    for l in dipz:
        if type(l) != int:
            if len(l) == 2:
                dmudqdqdict['z'][(l[0] - 7, l[1] - 7)] = dipz[l]

    dmudqarray = np.array(list(dmudqdict.values()))
    dmudqdarray = np.array(list(dmudqdqdict.values())).T

    return dmudqarray, dmudqdarray

# not used
def pklDipole(filenamebase, outfile):
    import pickle

    print('\nPickling in pklDipole in', os.getcwd())

    dmudqarray, dmudqdarray = getDipoleDers(filenamebase, outfile)
    filename = 'dipolexyz.pkl'

    with open(filename, 'wb') as file:
        pickle.dump((dmudqarray, dmudqdarray), file)

    print(f'\n --> {filename} file created\n')

    return os.getcwd()+'/'+filename

# used
def pklCubic(cubicfile):

    cubic = pCubicORQuartic(cubicfile)
    filenamec = 'cubic.pkl'

    with open(filenamec, 'wb') as file:
        pickle.dump(cubic, file)

    print(f'\n --> {filenamec} file created\n')

    return filenamec

# -----------------------------------------------------------------------------

# used in parse_output_file
def modes_array2tuple(arr: np.array):
    """I J K L M NI NJ NK NL NM information array
    turned into a tuple, where I mode will be mentioned NI times and so on"""
    # Extract the non-zero elements from the array (ignoring the zeros after the first non-zero element)
    non_zero_elements = [x for x in arr[:3] if x != 0]  # We only care about the first two non-zero elements
    tuple_elements = []
    tuple_elements.extend([non_zero_elements[0]] * arr[5])

    if len(non_zero_elements) == 2:
        tuple_elements.extend([non_zero_elements[1]] * arr[6])

    if len(non_zero_elements) == 3:
        tuple_elements.extend([non_zero_elements[1]] * arr[6])
        tuple_elements.extend([non_zero_elements[2]] * arr[7])

    result_tuple = tuple(sorted(tuple_elements))

    return result_tuple