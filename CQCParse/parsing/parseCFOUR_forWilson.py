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
from scipy import constants
from scipy.constants import physical_constants

import logging
logger = logging.getLogger("CQCParse")

bohr_in_angstroms = physical_constants['Bohr radius'][0]/10**(-10)

def GHz2Nu(ghz: float | np.ndarray) -> float | np.ndarray:
    """Conversion from GHz to cm-1"""
    return ghz*10**9/(constants.c*100)

def convNu2Ene(reciprocal_cm: float | np.ndarray) -> float | np.ndarray:
    """Convert wavenumber (cm-1) to energy (Hartree)"""
    hartree2J = constants.physical_constants['hartree-joule relationship'][0]
    return reciprocal_cm * (100 * constants.h * constants.c / hartree2J)


def parse_coriolis(file_path: str, nModes: int, startmode: int)-> tuple[np.ndarray, np.ndarray]:
    """
    lines: list[str]

    returns:
        rotational_constant - shape (3,); coriolis_constant - shape (3, nmodes, nmodes)
        startmode = starting index of normal modes (when removed 5 or 6 rotational,tranlslational)
            when 5 are removed, first index is 6
            when 6 are removed, first index is 7
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    corXtuples, corYtuples, corZtuples = [], [], []
    rotational_constant = []

    start1, start2, start3 = False, False, False
    for line in lines:
        if 'Coriolis Zeta matrix for IXYZ=                     1 :' in line:
            start1 = True
        elif 'CHECKSUM' in line:
            start1 = False
        elif start1:
            l1 = [int(line.split()[0]), int(line.split()[1]), float(line.split()[2])]
            corXtuples.append(tuple(l1))

        if 'Coriolis Zeta matrix for IXYZ=                     2 :' in line:
            start2 = True
        elif 'CHECKSUM' in line:
            start2 = False
        elif start2:
            l2 = [int(line.split()[0]), int(line.split()[1]), float(line.split()[2])]
            corYtuples.append(tuple(l2))

        if 'Coriolis Zeta matrix for IXYZ=                     3 :' in line:
            start3 = True
        elif 'CHECKSUM' in line:
            start3 = False
        elif start3:
            l3 = [int(line.split()[0]), int(line.split()[1]), float(line.split()[2])]
            corZtuples.append(tuple(l3))

        if 'B in cm-1:' in line:
            rotational_constant.append(float(line.split()[-1]))
    rotational_constant = np.array(rotational_constant)
    if len(rotational_constant)>3:
        rotational_constant = rotational_constant[-3:]

    # vibrations only: a translation or rotation (mode < startmode) would land at a negative index
    corXtuples, corYtuples, corZtuples = (tuple(item for item in corXtuples if min(item[:2]) >= startmode),
                                          tuple(item for item in corYtuples if min(item[:2]) >= startmode),
                                          tuple(item for item in corZtuples if min(item[:2]) >= startmode))

    corX, corY, corZ = np.zeros((nModes, nModes)), np.zeros((nModes, nModes)), np.zeros((nModes, nModes))

    for i, j, val in corXtuples:
        corX[i - startmode, j - startmode] = val
        corX[j - startmode, i - startmode] = val

    for i, j, val in corYtuples:
        corY[i - startmode, j - startmode] = val
        corY[j - startmode, i - startmode] = val

    for i, j, val  in corZtuples:
        corZ[i - startmode, j - startmode] = val
        corZ[j - startmode, i - startmode] = val
    coriolis_constant = np.array([corX, corY, corZ])

    return rotational_constant, coriolis_constant


def count_normal_coordinates(filepath: str) -> int:
    """
    Number of normal coordinates in the out file: 3N for N atoms, translations and rotations included
    (section "Normal coordinates for reference point", one "Normal coordinate <k>" header per mode)
    """
    import re
    with open(filepath, 'r') as file:
        labels = {int(m.group(1)) for m in re.finditer(r'^\s*Normal coordinate\s+(\d+)\s*$', file.read(), re.M)}
    if not labels:
        raise ValueError(f'No "Normal coordinate" sections in {filepath}')
    return len(labels)


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
        for L in coords_section[:-1]:
            ll = np.array([float(i) for i in L.strip().split() if '.' in i])
            omat_array.append(ll)
        omat_array = np.array(omat_array)

        return omat_array
    else:
        print('No rotation matrix found in this outfile')

def get_detected_resonances_c4(filepath: str):

    with open(filepath, 'r') as file:
        file_content = file.read()

    if "Thresholds for removing resonance denominators:" in file_content:
        found_resonances_str = []
        for line in file_content:
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
    for L in lines:
        elements = L.split()
        alllines.append(np.array([int(k) if i < len(elements) - 1 else float(k) for i, k in enumerate(elements)]))

    return np.array(alllines)

# used
def getCubicPost(freq: dict, cubic: np.ndarray, startmode, recipcm: bool = False):
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
            i = int(fijk[0]) - startmode
            j = int(fijk[1]) - startmode
            k = int(fijk[2]) - startmode
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

            i = int(fijk[0]) - startmode
            j = int(fijk[1]) - startmode
            k = int(fijk[2]) - startmode
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

def getQuarticPost(freq: dict, quartic: np.ndarray, startmode, recipcm: bool = False):
    """ Derives cubic and quartic anharmonic constants.
        It takes reduced values [cm-1] from gaussian output
        and transforms it to :
          * cubic   force constants : [Hartree*amu(-3/2)*Bohr(-3)]
          * quartic force constants : [Hartree*amu(-2  )*Bohr(-4)]

        AVERAGING: the CFOUR quartic file lists some index sets more than once. The (i,i,j,j) terms come from two
        numerical differentiations, e.g. (12,12,7,7) -331.4924 and (7,7,12,12) -331.5419 cm-1 (CCSDTcc_pVQZ), and
        differ by up to ~1e-3 of the value. Every order of an index set gets the mean of all its lines.
    """
    n = len(freq)
    K4 = np.zeros((n, n, n, n), dtype=np.float64)

    # mean of the lines of one index set (see docstring)
    values = {}
    for fijkl in quartic:
        ijkl = tuple(sorted(int(m) - startmode for m in fijkl[:4]))
        values.setdefault(ijkl, []).append(np.float64(fijkl[4]))

    for (i, j, k, L), vals in values.items():
        d = np.mean(vals)

        indices = [(i, j, k, L), (i, j, L, k), (i, k, j, L), (i, k, L, j),
                   (i, L, j, k), (i, L, k, j), (j, i, k, L), (j, i, L, k),
                   (j, k, i, L), (j, k, L, i), (j, L, i, k), (j, L, k, i),
                   (k, i, j, L), (k, i, L, j), (k, j, i, L), (k, j, L, i),
                   (k, L, i, j), (k, L, j, i), (L, i, j, k), (L, i, k, j),
                   (L, j, i, k), (L, j, k, i), (L, k, i, j), (L, k, j, i)]

        for idx in indices:
            K4[idx] = d

    if not recipcm:
        from scipy import constants
        # to go from amu to au mass unit (m_e)
        amc_au = constants.physical_constants['atomic mass constant'][0] / \
                 constants.physical_constants['atomic unit of mass'][0]
        K4 = K4 / amc_au ** 2

    return K4


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
    for L in lines:
        nu = np.array([float(k) for k in L.split()])
        indx = tuple([int(i) for i in nu[:-1] if i!=0.0])
        if len(indx) == 1:
            dct[indx[0]] = nu[-1]
        else:
            dct[indx] = nu[-1]

    return dct

def getDipoleDers_anharm(filenamebase: str, labels: list, nModesStart: int):
    """
    Getting dipole moment derivatives tensors

    AVERAGING: the dipole(xyz) files list the second derivative d2mu/dQi dQj of each mode pair twice, in both
    orders, e.g. (9,10) and (10,9). The two come from two numerical differentiations and differ slightly (up to
    ~1e-3 of the value in the example files). Both slots [i, j] and [j, i] get the mean of the two. If only one
    order is listed, it goes to both slots.

    :param nModesStart:
    :param labels:
    :param filenamebase: basename for dipole(xyz) files, e.g, 'dipole'
    :return:  first derivatives of the dipole moment, dmu/dQi, shape (3N-6, 3)
              second derivatives of the dipole moment, d2mu/dQi dQj, shape (3N-6, 3N-6, 3), symmetric in i, j
    """
    last_part = filenamebase.split('/')[-1]

    if last_part in ['dipolex', 'dipoley', 'dipolez']:
        filenamebase = filenamebase[:-1]
    
    dipx = pDipole(filenamebase+'x')
    dipy = pDipole(filenamebase+'y')
    dipz = pDipole(filenamebase+'z')

    dq = len(labels)
    dmudq_array = np.zeros((dq, 3))
    dmudqdq_array = np.zeros((dq, dq, 3))

    for L in labels:
        if isinstance(L, int):
            if L in dipx:
                dmudq_array[L-nModesStart, 0] = dipx[L]
            if L in dipy:
                dmudq_array[L-nModesStart, 1] = dipy[L]
            if L in dipz:
                dmudq_array[L-nModesStart, 2] = dipz[L]

    # mean of the two orders of a pair (see docstring)
    for xyz, dip in enumerate((dipx, dipy, dipz)):
        for L in dip:
            if not isinstance(L, int):
                if len(L) == 2:
                    i, j = L[0] - nModesStart, L[1] - nModesStart
                    val = (dip[L] + dip.get((L[1], L[0]), dip[L])) / 2
                    dmudqdq_array[i, j, xyz] = val
                    dmudqdq_array[j, i, xyz] = val

    return dmudq_array, dmudqdq_array

def getDipoleDers_anharm_au(filenamebase: str, labels: list, nModesStart: int, fundamentals_harmonic: dict) -> tuple:
    firstder, secder = getDipoleDers_anharm(filenamebase, labels, nModesStart)
    w_h = convNu2Ene(np.array([v for k, v in fundamentals_harmonic.items()]))
    matrix_2d = np.outer(w_h, w_h)

    sqrtvec = 1. / np.sqrt(w_h)
    sqrtmat = 1. / np.sqrt(matrix_2d.T)

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


def getDipoleDers_anharm_au_simple(filenamebase: str, labels: list, nModesStart: int, 
                                   fund_harmonic_energies_array: list | np.ndarray) -> tuple:
    firstder, secder = getDipoleDers_anharm(filenamebase, labels, nModesStart)
    w_h = convNu2Ene(np.array(fund_harmonic_energies_array))
    matrix_2d = np.outer(w_h, w_h)
    
    sqrtvec = 1. / np.sqrt(w_h)
    sqrtmat = 1. / np.sqrt(matrix_2d.T)

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
    w_h = convNu2Ene(np.array([v for k, v in fundamentals_harmonic.items()]))
    matrix_2d = np.outer(w_h, w_h)
    sqrtvec = 1. / np.sqrt(w_h)
    sqrtmat = 1. / np.sqrt(matrix_2d.T)

    with open(polar_pkl_file, 'rb') as file:
        alpha = pickle.load(file)
    polgrad = alpha[0]
    polhess = alpha[1]

    logger.debug('polgrad')
    logger.debug(polgrad)

    logger.debug('polhess')
    logger.debug(polhess)

    fdpol = np.zeros_like(polgrad)
    for i in range(len(sqrtvec)):
        for j in range(3):
            for k in range(3):
                fdpol[i, j, k] = polgrad[i, j, k] / sqrtvec[i]

    sdpol = np.zeros_like(polhess)
    for i in range(len(sqrtvec)):
        for j in range(len(sqrtvec)):

            for k in range(3):
                for L in range(3):
                    sdpol[i, j, k, L] = polhess[i, j, k, L] / sqrtmat[i, j]

    return tuple([fdpol, sdpol])


def getPolarDers_pkl_au_simple(polar_pkl_file: str, fund_harmonic_energies_array: np.ndarray | list):
    w_h = convNu2Ene(np.array(fund_harmonic_energies_array))
    matrix_2d = np.outer(w_h, w_h)
    sqrtvec = 1. / np.sqrt(w_h)
    sqrtmat = 1. / np.sqrt(matrix_2d.T)

    with open(polar_pkl_file, 'rb') as file:
        alpha = pickle.load(file)
    polgrad = alpha[0]
    polhess = alpha[1]

    logger.debug('polgrad')
    logger.debug(polgrad)

    logger.debug('polhess')
    logger.debug(polhess)

    fdpol = np.zeros_like(polgrad)
    for i in range(len(sqrtvec)):
        for j in range(3):
            for k in range(3):
                fdpol[i, j, k] = polgrad[i, j, k] / sqrtvec[i]

    sdpol = np.zeros_like(polhess)
    for i in range(len(sqrtvec)):
        for j in range(len(sqrtvec)):

            for k in range(3):
                for L in range(3):
                    sdpol[i, j, k, L] = polhess[i, j, k, L] / sqrtmat[i, j]

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
def pQUADRATURE(lines: list[str]) -> tuple[np.ndarray, np.array, dict[int: np.ndarray]]:
    """Dimensionless normal coordinates are here, in QUADRATURE file"""

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

    for L in labels:
        if isinstance(L,int):
            dmudqdict[L] = np.zeros(3)
            if L in dipx:
                dmudqdict[L][0] = dipx[L]
            if L in dipy:
                dmudqdict[L][1] = dipy[L]
            if L in dipz:
                dmudqdict[L][2] = dipz[L]

    for L in dipx:
        if not isinstance(L, int):
            if len(L) == 2:
                dmudqdqdict['x'][(L[0] - 7, L[1] - 7)] = dipx[L]

    for L in dipy:
        if not isinstance(L, int):
            if len(L) == 2:
                dmudqdqdict['y'][(L[0] - 7, L[1] - 7)] = dipy[L]

    for L in dipz:
        if not isinstance(L, int):
            if len(L) == 2:
                dmudqdqdict['z'][(L[0] - 7, L[1] - 7)] = dipz[L]

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