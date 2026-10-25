"""
#################################################################################################
##                                                                                             ##
##                             Parsing Gaussian output files                                   ##
##                                                                                             ##
#################################################################################################
#
# Files:
#     - .log  --- the main full output file that contains all the relevant data
"""

import logging
import sys

import numpy as np

from CQCParse.utils import AMU_IN_ME

logger = logging.getLogger("CQCParse")

import pandas as pd

pd.set_option('display.max_rows', sys.maxsize)


def parse_coriolis(lines: list[str], nModes: int, linear_molecule: bool = False)-> tuple[np.ndarray, np.ndarray]:
    """
    returns:
        rotational_constant - shape (3,); coriolis_constant - shape (3, nmodes, nmodes)
    """
    corXtuples, corYtuples, corZtuples = [], [], []
    rotational_constant = []

    start1 = False
    start_rotcont = False
    # a missing section raises below, instead of giving zeros
    found_coriolis = False
    found_rotational = False

    for i, line in enumerate(lines):

        if 'CORIOLIS COUPLINGS' in line:
            start1 = True
            found_coriolis = True
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
                rot_order = [line.strip().split()[-2][-2], line.strip().split()[-1][-2]]
                cart_lbls = ['x', 'y', 'z']
                lbls_here = [i for i in cart_lbls if i in rot_order]
                reorder = [rot_order.index(i) for i in lbls_here]

                next_line = lines[i+1]
                if 'Equilibrium Geometry' not in next_line:
                    raise ValueError("Cannot get rotational constant at Equilibrium Geometry from file")
                rotational_constant = [float(i) for i in next_line.split()[2:]]

                rotational_constant = np.array([float(rotational_constant[i]) for i in reorder])

                found_rotational = True
                break
        else:
            if 'E(harm)  E(anharm)' in line:
                if len(rotational_constant) < 3:
                    raise ValueError('no equilibrium rotational constants before the "E(harm)  E(anharm)" line '
                                     '("Rotational Constants (in cm^-1)" section)')
                rot_order = [line.strip().split()[-3][-2], line.strip().split()[-2][-2], line.strip().split()[-1][-2]]
                reorder = [rot_order.index('x'), rot_order.index('y'), rot_order.index('z')]
                rotational_constant = np.array([float(rotational_constant[i]) for i in reorder])

                next_line = lines[i+1]
                if 'Equilibrium Geometry' not in next_line:
                    raise ValueError("Cannot get rotational constant at Equilibrium Geometry from file")
                # rot_vals = [float(i) for i in next_line.split()[2:]]
                found_rotational = True
                break

    if not found_rotational:
        raise ValueError('no "E(harm)  E(anharm)" line: no rotational constants (axis order) in the file')
    if not found_coriolis:
        raise ValueError('no "CORIOLIS COUPLINGS" section')

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

def parse_Na(lines: list[str]) -> int:
    """Number of atoms, from the first 'NAtoms=' line (Gaussian prints one in every job step)"""
    for line in lines:
        if 'NAtoms=' in line:
            return int(line.split('NAtoms=')[1].split()[0])
    raise ValueError('No "NAtoms=" line in the Gaussian output')


def get_detected_resonances_g16(file_content: list[str]) -> list[tuple[int, int, int]]:
    """
    Validation only: not used by parse_from_source. For checking wilson's VPT2 against Gaussian's own.

    The Fermi resonances Gaussian's VPT2 found and treated, nu_I ~ nu_J + nu_K: one (I, J, K) per row of the
    "Fermi resonances" table in the "Resonance Analysis" section. Modes counted from 0 in Gaussian's A order, the
    order of the other mode indices of parse_from_source; wilson's identify_fermi gives the same kind of triple.
    Darling-Dennison resonances are not read.
    Example, formaldehyde HF/cc-pVTZ: [(4, 2, 5)], nu5 ~ nu3 + nu6 (130 cm-1 apart).
    Returns [] for a line "No Fermi resonance ..." (wording assumed from Gaussian's "No 1-1 Darling-Dennison
    resonance found"; no example file has it).
    Raises ValueError if the file has no "Resonance Analysis" section (e.g. CO: 1 mode), or if the table holds
    another number of rows than its "N Active Fermi resonances over M" line says (M).
    """
    lines = [line.strip() for line in file_content]
    if not any('Resonance Analysis' in line for line in lines):
        raise ValueError('no "Resonance Analysis" section')
    resonances = []
    in_table = False
    for line in lines:
        if line.startswith('No Fermi resonance'):
            return []
        if line.startswith('I      J  +   K'):
            in_table = True
        elif in_table and 'Active Fermi resonances over' in line:
            total = int(line.split()[-1])
            if len(resonances) != total:
                raise ValueError(f'Fermi resonance table: {len(resonances)} rows, "{line}"')
            return resonances
        elif in_table and line:
            i, j, k = (int(x) - 1 for x in line.split()[:3])
            resonances.append((i, j, k))
    raise ValueError('no "Fermi resonances" table with its "Active Fermi resonances over" line')

def getDipDers_log(logfile: list[str]) -> tuple:
    """
    Dipole derivatives: first order and second order, in e a0 (parse_dipole_moment converts from Debye) per
    Gaussian's normal coordinates, amu^(1/2) a0; getDipDers_au turns these into atomic units.
    Return: tuple[np.ndarray - shape(NM, 3), np.ndarray - shape(NM, NM, 3)]
    """
    dipl, _ = parse_dipole_moment(logfile)
    a2d = dipl.loc[dipl['P'] == 'P1', ['X', 'Y', 'Z']].to_numpy()
    shpNM = a2d.shape[0]
    a2d2 = dipl.loc[dipl['P'] == 'P2', ['X', 'Y', 'Z']].to_numpy()
    a2d2_3d = np.zeros((shpNM, shpNM, 3))

    for i, j, xyz in zip(dipl.loc[dipl['P'] == 'P2', 'i'], dipl.loc[dipl['P'] == 'P2', 'j'], a2d2, strict=True):
        a2d2_3d[int(i) - 1, int(j) - 1] = xyz
        a2d2_3d[int(j) - 1, int(i) - 1] = xyz

    return tuple([a2d, a2d2_3d])

def getDipDers_au(logfile: list[str]) -> tuple:
    """
    dipgrad (n, 3) and diphess (n, n, 3) in atomic units: dmu/dQ_i and d2mu/dQ_i dQ_j, with mu in e a0 and Q the
    mass-weighted normal coordinates in m_e^(1/2) a0 (wilson's contract).

    Gaussian prints the derivatives by its normal coordinates in amu^(1/2) a0, with mu in Debye ("Unit of the
    property: Debye", checked by parse_dipole_moment). Two factors:
        mu: 1 Debye = 0.393430 e a0 (parse_dipole_moment, debye_to_au)
        Q: Q[m_e^(1/2) a0] = sqrt(m_u / m_e) * Q[amu^(1/2) a0], so each mode index brings 1 / sqrt(m_u / m_e)
           = 1 / 42.6953 (AMU_IN_ME = m_u / m_e = 1822.888)
        dipgrad = dmu/dQ [Debye amu^(-1/2) a0^(-1)] * 0.393430 / 42.6953
        diphess = d2mu/dQ2 [Debye amu^(-1) a0^(-2)] * 0.393430 / 1822.888
    Example, formaldehyde HF/cc-pVTZ, mode 4 (B1, 1337 cm-1), x: Gaussian prints 0.119586, so dipgrad =
    0.119586 * 0.393430 / 42.6953 = 1.1020e-3. CFOUR gives the same by another route (getDipoleDers_anharm_au);
    tests/test_cfour_vs_gaussian.py checks every value.
    """
    a2d, a2d2_3d = getDipDers_log(logfile)

    firstder = a2d / np.sqrt(AMU_IN_ME)
    secder = a2d2_3d / AMU_IN_ME
    return tuple([firstder, secder])

def getPolarDers_log(logfile: list[str]) -> tuple:
    """
    Polarizability derivatives: first order and second order
    Return: tuple[np.ndarray - shape(NM, 3, 3), np.ndarray - shape(NM, NM, 3, 3)]
    """
    pol = parse_polarizability(logfile)
    # columns: 0 P, 1 i, 2 j, 3 k, 4 component (X, Y, Z), 5-7 the X, Y, Z values; one row per component, so each
    # mode (P1) or mode pair (P2) has 3 rows. One pass over the rows: a pandas filter per mode pair took over 1 s on
    # ACDM (42 modes)
    kind = pol[0].to_numpy()
    modes_i, modes_j = pol[1].to_numpy(), pol[2].to_numpy()
    values = pol[[5, 6, 7]].to_numpy(dtype=np.float64)

    def rows_by_modes(p, n_modes):
        rows = {}
        for idx in np.flatnonzero(kind == p):
            key = (int(modes_i[idx]),) if n_modes == 1 else (int(modes_i[idx]), int(modes_j[idx]))
            rows.setdefault(key, []).append(values[idx])
        return rows

    p1 = rows_by_modes('P1', 1)
    shpNM = max(i for i, in p1)
    p1_3d = np.zeros((shpNM, 3, 3))
    for (i,), xyz in p1.items():
        p1_3d[i - 1] = xyz

    p2 = rows_by_modes('P2', 2)
    nm_i = max(i for i, _ in p2)
    nm_j = max(j for _, j in p2)

    p2_4d = np.zeros((nm_i, nm_j, 3, 3))
    # Gaussian prints each pair once (i >= j): it goes to both orders
    for (i, j), xyz in p2.items():
        p2_4d[i - 1, j - 1] = xyz
        p2_4d[j - 1, i - 1] = xyz

    return tuple([p1_3d, p2_4d])

def getPolarDers_au(logfile: list[str]) -> tuple:
    """
    polgrad (n, 3, 3) and polhess (n, n, 3, 3) in atomic units: d alpha/dQ_i and d2 alpha/dQ_i dQ_j, with alpha in
    au and Q in m_e^(1/2) a0, as getDipDers_au.

    Gaussian prints alpha in au already ("Unit of the property: a.u.", checked by parse_polarizability), per its
    normal coordinates in amu^(1/2) a0. So only the coordinate changes: each mode index brings 1 / sqrt(m_u / m_e)
    = 1 / 42.6953 (see getDipDers_au). Checked against CFOUR (tests/test_cfour_vs_gaussian.py).
    """
    p1_3d, p2_4d = getPolarDers_log(logfile)
    fdpol = p1_3d / np.sqrt(AMU_IN_ME)
    sdpol = p2_4d / AMU_IN_ME

    return tuple([fdpol, sdpol])

# used in parse_gaussian16_output
def parse_cubic_constants(lines: list[str]) -> tuple[pd.DataFrame, list]:

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
    # a section with no rows (every term under Gaussian's cut-off) would give zeros; a missing one raises
    if not start:
        raise ValueError('no "CUBIC FORCE CONSTANTS IN NORMAL MODES" section')
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
    # as in parse_cubic_constants: no rows would give zeros, a missing section raises
    if not start:
        raise ValueError('no "QUARTIC FORCE CONSTANTS IN NORMAL MODES" section')
    df = pd.DataFrame(results, columns=["I", "J", "K", "L", "FI(I,J,K,L)", "k(I,J,K,L)", "K(I,J,K,L)", "SymmetryForbidden"])

    return df, units_lines

def get_cubic_post(len_freq: int, cubic: np.ndarray):
    """
    Cubic force constants as a full tensor (n, n, n), every index order filled, modes 0 .. n-1 (Gaussian's are 1-based).
    cubic: rows (I, J, K, value) from parse_cubic_constants; the tensor keeps the unit of the value column.
    The reader passes column FI (cm-1, dimensionless normal coordinates; cff_rc) and converts that to atomic units
    with CQCParse.utils.reduced_to_au (cff), which explains the conversion.
    """
    K3 = np.zeros((len_freq, len_freq, len_freq), dtype=np.float64)

    for fijk in cubic:
        i = int(fijk[0]) -1
        j = int(fijk[1]) -1
        k = int(fijk[2]) -1
        d = np.float64(fijk[3])

        K3[i, j, k] = d
        K3[i, k, j] = d
        K3[k, j, i] = d
        K3[k, i, j] = d
        K3[j, i, k] = d
        K3[j, k, i] = d

    return K3

def get_quartic_post(len_freq: int, quartic: np.ndarray):
    """
    Quartic force constants as a full tensor (n, n, n, n), every index order filled, modes 0 .. n-1.
    quartic: rows (I, J, K, L, value) from parse_quartic_constants; the tensor keeps the unit of the value column.
    The reader passes column FI (cm-1; qff_rc) and converts it with CQCParse.utils.reduced_to_au (qff).
    """
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

    return K4

# used in getDipDers_log
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

    if not results:
        raise ValueError('no dipole derivatives: no "Electric Dipole" section with P rows')
    # the values were converted with debye_to_au above: that holds only for Debye
    if units_line is None or 'Debye' not in units_line:
        raise ValueError(f'dipole derivatives: unit line {units_line!r}, expected "Unit of the property: Debye"')
    df = pd.DataFrame(results, columns=column_names)

    return df, units_line

# used in getPolarDers_log
def parse_polarizability(lines: list[str]) -> pd.DataFrame:

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

            elif ('|  X  |' in line or '|  Z  |' in line) and len(line.split('|')) == 4 and 'i' not in line:
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
    if not results:
        raise ValueError('no polarizability derivatives: no "Polarizability Tensor" section with P rows')
    # the values are taken as atomic units, with no conversion
    if units_line is None or 'a.u.' not in units_line:
        raise ValueError(f'polarizability derivatives: unit line {units_line!r}, expected "Unit of the property: a.u."')
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
    floats = nm_floats(filename)
    if not floats:
        raise ValueError('no normal-mode table with 3 modes per block ("Atom  AN  X Y Z  X Y Z  X Y Z"): the reader '
                         'reads only those, so not a molecule with fewer than 3 modes')
    m1 = np.reshape(np.array(floats), (-1, Na, 3, 3))
    result = np.transpose(m1, (0, 2, 1, 3)).reshape(-1, 3, Na, 3)
    modes = result.reshape(-1, 3 * Na)

    # a last block of 1 or 2 modes has another header and is not read
    if len(modes) not in (3 * Na - 6, 3 * Na - 5):
        raise ValueError(f'{len(modes)} normal modes read for {Na} atoms, expected {3 * Na - 6} or {3 * Na - 5}: '
                         f'the reader reads only blocks of 3 modes')

    return modes


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
            atomic_number = line.strip().split()[1]
            if atomic_number not in dictelem:
                raise ValueError(f'atomic number {atomic_number}: get_equil_geo knows only {dictelem}')
            elements.append(dictelem[atomic_number])
            coords.append([float(line.strip().split()[-3]),
                           float(line.strip().split()[-2]),
                           float(line.strip().split()[-1])])

    if not elements:
        raise ValueError('no optimized geometry: no "Standard orientation" after "Optimization completed."')
    if len(elements) != parse_Na(allines):
        raise ValueError(f'{len(elements)} atoms in the optimized geometry, NAtoms= says {parse_Na(allines)}')
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
    Validation only: not used by parse_from_source. For checking wilson's VPT2 against Gaussian's own.

    Parse the Anharmonic X Matrix section from Gaussian output file lines: the anharmonic constants X_ij in cm-1,
    in Gaussian's A order (the order of the other mode indices of parse_from_source). Checked on formaldehyde:
    E(2 nu_i) - 2 E(nu_i) = 2 X_ii and E(nu_i + nu_j) - E(nu_i) - E(nu_j) = X_ij from the anharmonic states, to
    the states' rounding, for the modes outside Gaussian's resonance treatment.

    Args:
        file_lines (list): List of strings, each representing a line from the file

    Returns:
        dict: Dictionary containing parsed anharmonic X matrix data with keys:
            - 'coriolis_contributions': (n, n) array of coriolis contribution values
            - '3rd_deriv_contributions': (n, n) array of 3rd derivative contribution values
            - '4th_deriv_contributions': (n, n) array of 4th derivative contribution values
            - 'total_anharmonic': (n, n) array of total anharmonic X matrix values (the sum of the three, within
              the file's print rounding)
            - 'matrix_size': the shape, (n, n)
    Raises ValueError if the file has no "Anharmonic X Matrix" section.
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
        raise ValueError('no "Anharmonic X Matrix" section')
    
    
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
    if not a_vals:
        raise ValueError('no A to H mode table ("The connection between this new numbering")')
    return dict(zip(a_vals, h_vals, strict=True))
