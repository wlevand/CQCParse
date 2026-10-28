"""
Building polar.pkl: CFOUR polarizability derivatives by finite differences.

polar.pkl is not a CFOUR file. The CFOUR reader (parse_from_source: 'polgrad', 'polhess') reads it as a pickle of
(first, second):
    first   (n, 3, 3)      d alpha / d q_k
    second  (n, n, 3, 3)   d2 alpha / d q_k d q_l
with alpha in au and the n vibrational modes in ascending CFOUR order (index 0 = CFOUR mode 7 for a nonlinear
molecule). The reader takes q as dimensionless normal coordinates (getPolarDers_pkl_au).

Input: one CFOUR polarizability run per displaced geometry, and the reference (undisplaced) run:
    <polar_dir>/<k>p, <polar_dir>/<k>n               mode k displaced by +h, -h
    <polar_dir>/<k>_<l>pp, _pn, _np, _nn             modes k < l displaced by (+h, +h), (+h, -h), (-h, +h), (-h, -h)
    <polar_dir>/../anharm/POLAR                      the reference run
k and l are CFOUR mode numbers. Each run folder holds POLAR (the polarizability tensor, read by pTensor) and
outfile0.out (for OMAT). How the displaced geometries were made is not recorded here.

Rotation: CFOUR puts each displaced geometry in its own orientation and prints the rotation in outfile0.out
(OMAT, "QCOM = OMAT * QCOMP", printed with PRINT_LEVEL=1). Each displaced tensor is rotated back:
alpha' = OMAT^T alpha OMAT. The reference POLAR is used as it is: it is already in that frame. Checked on the
three example folders below: the reference tensor equals the mean of each mode's two rotated displaced tensors
within 1.5e-5 au; rotated by its own OMAT (which swaps x and y there) it is off by up to 5 au.

Finite differences, step h = STEP = 0.01:
    first      (p - n) / (2h)
    mixed      (pp - pn - np + nn) / (4 h^2)
    diagonal   (p - 2 reference + n) / h^2

The displaced runs themselves are not in the repo. Their POLAR tensors and OMATs are, in polarData_raw.pkl
(pklPoldata) in files_examples/refinedc4/CCSDTcc_pVQZ, HFcc_pVQZ and HFcc_pVTZ: polar_derivatives rebuilds each
folder's polar.pkl from them (tests/test_build_polar_pkl.py). refinedc4/HFcc_pVDZ has polar.pkl only.

Use: pklPolder(polar_dir) writes polar.pkl into the current folder; copy it into the folder of the anharmonic run.
"""

import logging
import os
import pickle
import re

import numpy as np

logger = logging.getLogger("CQCParse")

STEP = 0.01


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
        raise ValueError(f'no OMAT in {filepath} (CFOUR prints it with PRINT_LEVEL=1)')


def getDisplacementsPolarData(polar_dir: str, raw: bool = False):
    """
    Collecting the POLAR files data and also rotates is according to the matrix from the corresponding outfile0.out
    :param raw: True: {run folder: (POLAR as read, OMAT)}; False: {run folder: rotated POLAR}
    :param polar_dir: where polar calcs were done (one folder per displaced run, see the module docstring)

    :return:
    """
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

        allpolardata[directory] = alpha_prime

    if raw:
        return allpolardataRaw
    else:
        return allpolardata


def polar_derivatives(rotated: dict, reference: np.ndarray, step: float = STEP) -> tuple[np.ndarray, np.ndarray]:
    """
    The finite differences of the module docstring.

    :param rotated: {run folder name, e.g. '7p' or '7_8pn': rotated POLAR}, as getDisplacementsPolarData returns
    :param reference: the reference run's POLAR, as read (not rotated)
    :return: first (n, 3, 3), second (n, n, 3, 3), modes in ascending CFOUR order
    """
    modes = sorted({int(m) for name in rotated for m in re.findall(r'\d+', name)})

    first = np.array([(rotated[f'{k}p'] - rotated[f'{k}n']) / (2 * step) for k in modes])

    second = np.zeros((len(modes), len(modes), 3, 3))
    for a, k in enumerate(modes):
        # (∂²α_ij/∂Q_k²) ≈ [α_ij(Q_k + ΔQ_k) - 2α_ij(Q_k) + α_ij(Q_k - ΔQ_k)] / (ΔQ_k)²
        second[a, a] = (rotated[f'{k}p'] - 2 * reference + rotated[f'{k}n']) / step ** 2
        for b in range(a + 1, len(modes)):
            m = modes[b]
            # (∂²α_ij/∂Q_k∂Q_l) ≈ [α_ij(Q_k + ΔQ_k, Q_l + ΔQ_l) - α_ij(Q_k + ΔQ_k, Q_l - ΔQ_l)
            #   - α_ij(Q_k - ΔQ_k, Q_l + ΔQ_l) + α_ij(Q_k - ΔQ_k, Q_l - ΔQ_l)] / (4 * ΔQ_k * ΔQ_l)
            second[a, b] = second[b, a] = (rotated[f'{k}_{m}pp'] - rotated[f'{k}_{m}pn'] - rotated[f'{k}_{m}np']
                                           + rotated[f'{k}_{m}nn']) / (4 * step * step)

    return first, second


def getPolarDers(polar_dir: str):
    """
    Computing polarizability derivatives
    :param polar_dir: where the displaced runs are (see the module docstring)
    :return: first, second (polar_derivatives)
    """
    reference_file = polar_dir + '/../anharm/POLAR'
    logger.info(f'reference polarizability: {reference_file}')
    return polar_derivatives(getDisplacementsPolarData(polar_dir), pTensor(reference_file))


def pklPolder(polar_dir):
    first, second = getPolarDers(polar_dir)
    filenamec = 'polar.pkl'

    with open(filenamec, 'wb') as file:
        pickle.dump((first, second), file)

    logger.info(f'{filenamec} file created in {os.getcwd()}')

    return filenamec


def pklPoldata(polar_dir):
    """
    Writes the inputs of polar.pkl, so that it can be rebuilt without the runs:
        polarData_raw.pkl  {run folder: (POLAR as read, OMAT)}
        polarData.pkl      {run folder: rotated POLAR}
    Both also hold 'equil', the reference run, ROTATED BY ITS OWN OMAT. That is not the frame of the other entries
    (see the module docstring): to use it with them, undo the rotation, R @ alpha' @ R.T with (alpha', R) =
    polarData_raw['equil']. Kept as is, so that the example polarData_raw.pkl files stay what this function writes.
    """
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

    filenamec2 = 'polarData_raw.pkl'

    with open(filenamec2, 'wb') as file:
        pickle.dump(rawdata, file)

    logger.info(f'{filenamec} and {filenamec2} files created in {os.getcwd()}')

    return filenamec
