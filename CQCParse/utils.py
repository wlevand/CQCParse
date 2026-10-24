import numpy as np
from scipy import constants

import os
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def convNu2Ene(reciprocal_cm: float | np.ndarray) -> float | np.ndarray:
    """Convert wavenumber (cm-1) to energy (Hartree)"""
    hartree2J = constants.physical_constants['hartree-joule relationship'][0]
    return reciprocal_cm * (100 * constants.h * constants.c / hartree2J)


def reduced_to_au(fc_cm: np.ndarray, harmonic_cm: np.ndarray) -> np.ndarray:
    """
    Force constants of any order n, from reduced form in cm-1 to atomic units, Hartree m_e^(-n/2) a0^(-n).

    Both programs print cubic and quartic force constants phi in dimensionless normal coordinates q, in cm-1
    (Gaussian: column FI; CFOUR: the files cubic and quartic):
        V = V0 + 1/2 sum_i w_i q_i^2 + 1/6 sum phi_ijk q_i q_j q_k + 1/24 sum phi_ijkl q_i q_j q_k q_l + ...
    cff and qff are the plain derivatives in mass-weighted normal coordinates Q instead:
        K_ijk = d3V / dQ_i dQ_j dQ_k,    K_ijkl = d4V / dQ_i dQ_j dQ_k dQ_l
    The two coordinates differ by a factor per mode: q_i = Q_i * sqrt(w_i / hbar), with w_i the harmonic angular
    frequency of mode i. So each index of a force constant brings one factor sqrt(w / hbar):
        K_ijk = phi_ijk * sqrt(w_i / hbar) * sqrt(w_j / hbar) * sqrt(w_k / hbar)        (quartic: 4 factors)
    In atomic units hbar = 1, and an energy E in cm-1 is E / Eh in Hartree, with Eh = 219474.63 cm-1 (1 Hartree
    as a wavenumber). A harmonic wavenumber nu_i in cm-1 gives w_i = nu_i / Eh. So, with phi and nu in cm-1:
        K_ij... [Hartree m_e^(-n/2) a0^(-n)] = (phi_ij... / Eh) * sqrt(nu_i / Eh) * sqrt(nu_j / Eh) * ...

    Example, formaldehyde HF/cc-pVTZ (Gaussian), term (1,1,1): phi = -1312.24856 cm-1, nu_1 = 3084.240 cm-1:
        K_111 = (-1312.24856 / 219474.63) * (3084.240 / 219474.63)^(3/2) = -9.9604e-6
    Gaussian's own column K prints -0.77521 Hartree amu^(-3/2) Bohr^(-3); divided by (m_u / m_e)^(3/2) =
    1822.888^(3/2) that is -9.9605e-6, the same within K's 5 decimals.

    fc_cm: the force constants in cm-1, a tensor of shape (n_modes,) * n (any order n)
    harmonic_cm: the harmonic wavenumbers in cm-1, in the same mode order, shape (n_modes,)
    """
    hartree_cm = constants.physical_constants['hartree-inverse meter relationship'][0] / 100
    sqrt_w = np.sqrt(np.asarray(harmonic_cm, dtype=np.float64) / hartree_cm)    # sqrt(w_i / hbar) in au
    fc_au = np.asarray(fc_cm, dtype=np.float64) / hartree_cm
    for axis in range(fc_au.ndim):                                               # one factor per index
        shape = [1] * fc_au.ndim
        shape[axis] = -1
        fc_au = fc_au * sqrt_w.reshape(shape)
    return fc_au


def change_idx_modes(parserObj, new_idx_dict, list2exclude=None, only_modes = None,
                     data2transform=''):
    """
    new_idx_dict = {oldkey:newkey}

    To change:
        all_states - after vpt2, so no need to upd cubic_cm_1, quartic_cm_1
        fundamentals_harmonic - because used in calculation of prefactors
        all_states_harmonic - used if harmonic energy levels used

        deriv_data, list2exclude

            ddata = [parserObj.dipgrad,
             parserObj.diphess,
             parserObj.polgrad,
             parserObj.polhess,
             parserObj.cff]
    self.deriv_data = dict(zip(['dipgrad', 'diphess', 'polgrad', 'polhess', 'cff'], ddata))
    """

    if data2transform=='states':
        new_dict1 = {}
        new_dict2 = {}
        # new_dict3 = {}
        # new_dict4 = {}

        # upd self.all_states
        for oldkey, val in parserObj.anharmonic_states.items():

            newkey = tuple([str(j) for j in sorted([new_idx_dict[int(i)]  for i in oldkey if i!='zero'])])
            new_dict1[newkey] = val

        sorted_keys = sorted(new_dict1.keys(), key=lambda x: tuple(map(int, x)))
        new_dict1_sort = {key: new_dict1[key] for key in sorted_keys}
        all_states = new_dict1_sort

        # upd self.all_states_harmonic
        for oldkey, val in parserObj.harmonic_states.items():
            newkey = tuple([str(j) for j in sorted([new_idx_dict[int(i)]  for i in oldkey if i!='zero'])])
            new_dict2[newkey] = val

        sorted_keys = sorted(new_dict2.keys(), key=lambda x: tuple(map(int, x)))
        new_dict2_sort = {key: new_dict2[key] for key in sorted_keys}

        all_states_harmonic = new_dict2_sort

        # upd self.fundamentals_harmonic
        # for oldkey, val in parserObj.fundamentals_harmonic_str.items():
        #     newkey = str(new_idx_dict[int(oldkey)])
        #     new_dict3[newkey] = val
        # sortKeys = list(new_dict3.keys())
        # sortKeys.sort()
        # new_dict3_sort = {i: new_dict3[i] for i in sortKeys}

        # fundamentals_harmonic = new_dict3_sort

        # upd self.fundamentals
        # for oldkey, val in parserObj.fundamentals_anharmonic_str.items():
        #     newkey = str(new_idx_dict[int(oldkey)])
        #     new_dict4[newkey] = val
        # sortKeys = list(new_dict3.keys())
        # sortKeys.sort()
        # new_dict4_sort = {i: new_dict4[i] for i in sortKeys}

        # fundamentals = new_dict4_sort

        return all_states, all_states_harmonic#, fundamentals, fundamentals_harmonic

    elif data2transform=='mode_indices':
        # for old in self.list2exclude:
        #     new_list.append(new_idx_dict[old])
        # self.list2exclude = new_list
        nmodes = parserObj.nmodes
        # input list2exclude and only_modes would be already with new indices
        if list2exclude is not None:
            mode_indices = [i for i in np.arange(nmodes) if i not in list2exclude]
            nmodes -= len(list2exclude)
        else:
            if only_modes is not None:
                mode_indices = only_modes
            else:
                mode_indices = [i for i in np.arange(nmodes)]

        return mode_indices

    elif data2transform=='derivatives':
        newmu1 = np.zeros_like(parserObj.dipgrad)
        newmu2 = np.zeros_like(parserObj.diphess)
        newalpha1 = np.zeros_like(parserObj.polgrad)
        newalpha2 = np.zeros_like(parserObj.polhess)
        newF = np.zeros_like(parserObj.cff)
        newQFF = np.zeros_like(parserObj.qff)

        # cff_cm_1_new = np.zeros_like(parserObj.cubic_cm_1)
        # qff_cm_1_new = np.zeros_like(parserObj.quartic_cm_1)
        # cor_c_new = np.zeros_like(parserObj.coriolis_constant)

        for oldkey, newkey in new_idx_dict.items():
            newmu1[newkey, :] = parserObj.dipgrad[oldkey, :]
            newalpha1[newkey, :, :] = parserObj.polgrad[oldkey, :, :]

        new_idx_dict_2d = {}
        for old_i, new_i in new_idx_dict.items():
            for old_j, new_j in new_idx_dict.items():
                new_idx_dict_2d[(old_i, old_j)] = (new_i, new_j)

        for (old_i, old_j), (new_i, new_j) in new_idx_dict_2d.items():
            newmu2[new_i, new_j, :] = parserObj.diphess[old_i, old_j, :]
            newalpha2[new_i, new_j, :, :] = parserObj.polhess[old_i, old_j, :, :]

        new_idx_dict_3d = {}
        for old_i, new_i in new_idx_dict.items():
            for old_j, new_j in new_idx_dict.items():
                for old_k, new_k in new_idx_dict.items():
                    new_idx_dict_3d[(old_i, old_j, old_k)] = (new_i, new_j, new_k)

        for (old_i, old_j, old_k), (new_i, new_j, new_k) in new_idx_dict_3d.items():
            newF[new_i, new_j, new_k] = parserObj.cff[old_i, old_j, old_k]


        ddata = [newmu1, newmu2, newalpha1, newalpha2, newF]
        deriv_data = dict(zip(['dipgrad', 'diphess', 'polgrad', 'polhess', 'cff'], ddata))

        return deriv_data

    elif data2transform=='force_consts':

        newCFF = np.zeros_like(parserObj.cubic_cm_1)
        newQFF = np.zeros_like(parserObj.quartic_cm_1)

        new_idx_dict_3d = {}
        for old_i, new_i in new_idx_dict.items():
            for old_j, new_j in new_idx_dict.items():
                for old_k, new_k in new_idx_dict.items():
                    new_idx_dict_3d[(old_i, old_j, old_k)] = (new_i, new_j, new_k)

        for (old_i, old_j, old_k), (new_i, new_j, new_k) in new_idx_dict_3d.items():
            newCFF[new_i, new_j, new_k] = parserObj.cubic_cm_1[old_i, old_j, old_k]

        new_idx_dict_4d = {}
        for old_i, new_i in new_idx_dict.items():
            for old_j, new_j in new_idx_dict.items():
                for old_k, new_k in new_idx_dict.items():
                    for old_l, new_l in new_idx_dict.items():
                        new_idx_dict_4d[(old_i, old_j, old_k, old_l)] = (new_i, new_j, new_k, new_l)

        for (old_i, old_j, old_k, old_l), (new_i, new_j, new_k, new_l) in new_idx_dict_4d.items():
            newQFF[new_i, new_j, new_k, new_l] = parserObj.quartic_cm_1[old_i, old_j, old_k, old_l]

        return newCFF, newQFF

    elif data2transform=='coriolis':
        coriolis_constants = np.zeros_like(parserObj.coriolis_constants)
        for oldkey, newkey in new_idx_dict.items():
            coriolis_constants[:, newkey] = parserObj.coriolis_constants[:, oldkey]
        # print(coriolis_constants)
        return coriolis_constants

    elif data2transform=='fermi_resonance':
        fermi_resonances_new = []
        for fr in parserObj.fermi_resonance:
            fr_new = [new_idx_dict[fr[0]]]+[new_idx_dict[fr[1]]]+[new_idx_dict[fr[2]]]+[fr[3]]
            fermi_resonances_new.append(fr_new)

        return fermi_resonances_new

    else:
        return

def change_idx_states(dict2upd, new_idx_dict):
    new_dict1 = {}

    for oldkey, val in dict2upd.items():
        if isinstance(oldkey, tuple):
            typekey = tuple
            newkey = tuple([str(j) for j in sorted([new_idx_dict[int(i)] for i in oldkey])])
        elif isinstance(oldkey, int):
            typekey = int
            newkey = new_idx_dict[oldkey]
        elif isinstance(oldkey, str):
            typekey = str
            newkey = str(new_idx_dict[int(oldkey)])
        else:
            typekey = type(oldkey)
            print('Unknown dict2upd')

        new_dict1[newkey] = val

    if typekey is str:
        sorted_keys = [str(j) for j in sorted([int(i) for i in new_dict1.keys()])]

    elif typekey is int:
        sorted_keys = sorted(list(new_dict1.keys()))

    elif typekey is tuple:
        sorted_keys = sorted(new_dict1.keys(), key=lambda x: tuple(map(int, x)))
    else:
        print('typekey is', typekey)

    new_dict1_sort = {key: new_dict1[key] for key in sorted_keys}
    all_states = new_dict1_sort

    return all_states
    # return new_dict1


def make_modes_idx(nmodes, modes=None, include=False):
    """
    modes have new idx
        for all: modes=[], include=False or modes=np.arange(nmodes), include=True
    """
    if modes is None:
        modes = []

    if include:
        return [i for i in np.arange(nmodes) if i in modes]
    else:
        return [i for i in np.arange(nmodes) if i not in modes]

