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
#     - POLAR                        --- static polarizability (a.u.); read by CQCParse.tools.build_polar_pkl
#     - out                          --- the final output file in anharmonic parallel procedure:
#                            1) All levels with up to three quanta frequencies; 2) equilibrium geometry;
#                            3) normal coordinates, non-mass-weighted (a.u.); 4) F(IJKK)/a.u ; 5) F(IJKK)/cm-1 ;
#                            6) harmonic and fundamental frequencies and intensities
#     - dipolex(yz)                  --- dipole moment (1st-2nd-3rd) order derivatives (normal coordinates)
#     - cubic                        --- cubic force constants in cm-1 in dimensionless normal modes
#     - quartic                      --- quartic force constants in cm-1 in dimensionless normal modes
"""

import logging
import pickle

import numpy as np

from CQCParse.utils import dimensionless_to_mass_weighted

logger = logging.getLogger("CQCParse")


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
    # IXYZ = 1, 2, 3: a missing matrix raises below, instead of giving zeros
    found_ixyz = set()
    for line in lines:
        for ixyz in (1, 2, 3):
            if f'Coriolis Zeta matrix for IXYZ=                     {ixyz} :' in line:
                found_ixyz.add(ixyz)

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
    missing = sorted({1, 2, 3} - found_ixyz)
    if missing:
        raise ValueError(f'no "Coriolis Zeta matrix for IXYZ= k" section for k = {missing} in {file_path}')
    if not rotational_constant:
        raise ValueError(f'no "B in cm-1:" line (rotational constants) in {file_path}')
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
        labels = {int(m.group(1)) for m in re.finditer(r'^\s*Normal coordinate\s+(\d+)\s*$', file.read(), re.MULTILINE)}
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
        raise ValueError(f'no "All levels with up to three quanta" section (anharmonic levels) in {filepath}')

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
    if not alllines:
        raise ValueError(f'no force constants: {filepath} is empty')

    return np.array(alllines)

def getCubicPost(freq: dict, cubic: np.ndarray, startmode):
    """ Cubic force constants as a full tensor (n, n, n), every index order filled, modes 0 .. n-1.
        In the file's own unit: cm-1, in dimensionless normal coordinates (cff_rc). For atomic units (cff):
        CQCParse.utils.reduced_to_au, which explains the conversion.
        freq: one entry per vibrational mode (only its length is used); startmode: CFOUR's number of the first
        vibration (7, or 6 for a linear molecule).
    """
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

def getQuarticPost(freq: dict, quartic: np.ndarray, startmode):
    """ Quartic force constants as a full tensor (n, n, n, n), every index order filled, modes 0 .. n-1.
        In the file's own unit: cm-1, in dimensionless normal coordinates (qff_rc). For atomic units (qff):
        CQCParse.utils.reduced_to_au, which explains the conversion.
        freq, startmode: as in getCubicPost.

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

    return K4


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
    if not dct:
        raise ValueError(f'no dipole derivatives: {filenamebase} is empty')

    return dct

def getDipoleDers_anharm(filenamebase: str, labels: list, nModesStart: int):
    """
    Getting dipole moment derivatives tensors, as the files hold them: by dimensionless normal coordinates q, in
    e a0 (getDipoleDers_anharm_au turns them into wilson's units)

    AVERAGING: the dipole(xyz) files list the second derivative d2mu/dqi dqj of each mode pair twice, in both
    orders, e.g. (9,10) and (10,9). The two come from two numerical differentiations and differ slightly (up to
    ~1e-3 of the value in the example files). Both slots [i, j] and [j, i] get the mean of the two. If only one
    order is listed, it goes to both slots.

    :param nModesStart:
    :param labels:
    :param filenamebase: basename for dipole(xyz) files, e.g, 'dipole'
    :return:  first derivatives of the dipole moment, dmu/dqi, shape (3N-6, 3)
              second derivatives of the dipole moment, d2mu/dqi dqj, shape (3N-6, 3N-6, 3), symmetric in i, j
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

def getDipoleDers_anharm_au(filenamebase: str, labels: list, nModesStart: int,
                            fund_harmonic_energies_array: list | np.ndarray) -> tuple:
    """
    dipgrad (n, 3) and diphess (n, n, 3) in atomic units, as the Gaussian reader's getDipDers_au: mu in e a0, Q the
    mass-weighted normal coordinates in m_e^(1/2) a0.

    CFOUR's dipole(xyz) files hold the derivatives by dimensionless normal coordinates q, with mu in e a0
    (getDipoleDers_anharm). q_i = Q_i * sqrt(w_i), so each mode index brings sqrt(nu_i / Eh), nu_i the harmonic
    wavenumber in cm-1 (CQCParse.utils.dimensionless_to_mass_weighted):
        dipgrad_i = dmu/dq_i * sqrt(nu_i / Eh),    diphess_ij = d2mu/dq_i dq_j * sqrt(nu_i / Eh) * sqrt(nu_j / Eh)
    Example, formaldehyde HF/cc-pVTZ, mode 7 (B1, 1337.152 cm-1), x: dipolex holds 0.0141179696, so dipgrad =
    0.0141179696 * sqrt(1337.152 / 219474.63) = 1.1020e-3, as Gaussian (getDipDers_au's example);
    tests/test_cfour_vs_gaussian.py checks every value.
    """
    firstder, secder = getDipoleDers_anharm(filenamebase, labels, nModesStart)
    return (dimensionless_to_mass_weighted(firstder, fund_harmonic_energies_array, 1),
            dimensionless_to_mass_weighted(secder, fund_harmonic_energies_array, 2))


def getPolarDers_pkl_au(polar_pkl_file: str, fund_harmonic_energies_array: np.ndarray | list):
    """
    polgrad (n, 3, 3) and polhess (n, n, 3, 3) in atomic units, as getDipoleDers_anharm_au: polar.pkl holds the
    derivatives by dimensionless normal coordinates q, alpha in au (CQCParse.tools.build_polar_pkl), and each mode
    index brings sqrt(nu_i / Eh). Checked against Gaussian (tests/test_cfour_vs_gaussian.py).
    """
    with open(polar_pkl_file, 'rb') as file:
        alpha = pickle.load(file)
    polgrad, polhess = alpha[0], alpha[1]
    return (dimensionless_to_mass_weighted(polgrad, fund_harmonic_energies_array, 1),
            dimensionless_to_mass_weighted(polhess, fund_harmonic_energies_array, 2))

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

    normal_coordinates = dict(zip(np.arange(7, len(freqs)+7), dqMat, strict=True))
    return np.array(undisplaced_matrix), freqs, normal_coordinates

# polar.pkl, read by getPolarDers_pkl_au, is built by CQCParse.tools.build_polar_pkl
# -----------------------------------------------------------------------------

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
                    raise ValueError(f'{filepath}: a [FR-NORM-COORD] data line before any "vibration" line')

    if not atoms:
        raise ValueError(f'no geometry: no [FR-COORD] section in {filepath}')
    atoms = np.array(atoms)
    geometry_data = np.array(geometry_data, dtype=float)
    vibrations_data = {key: np.array(value) for key, value in vibrations_data.items()}
    # only normal modes, ignore first 6 or 5
    ndiscard = 5 if linear else 6
    normal_modes_dict = {key: value for idx, (key, value) in enumerate(vibrations_data.items()) if idx >= ndiscard}

    return geometry_data, atoms, normal_modes_dict


# -----------------------------------------------------------------------------

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