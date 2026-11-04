import os

import numpy as np
from scipy import constants

PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HARTREE_CM = constants.physical_constants['hartree-inverse meter relationship'][0] / 100   # 1 Hartree in cm-1
# m_u / m_e: a mass-weighted coordinate in amu^(1/2) a0 is sqrt(AMU_IN_ME) times the same in m_e^(1/2) a0
AMU_IN_ME = (constants.physical_constants['atomic mass constant'][0]
             / constants.physical_constants['atomic unit of mass'][0])
BOHR_ANGSTROM = constants.physical_constants['Bohr radius'][0] * 1e10        # 1 bohr in angstrom


def center_of_mass_frame(geometry: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """The geometry (atoms, 3) moved so that its center of mass is at the origin; masses in any unit."""
    masses = np.asarray(masses, dtype=np.float64)
    return geometry - masses @ geometry / masses.sum()


def normal_modes_au(modes: np.ndarray, masses_amu: np.ndarray | None = None) -> np.ndarray:
    """
    Normal modes as the Cartesian displacement per unit normal coordinate, d = dx/dQ, in atomic units: x in bohr and
    Q the mass-weighted normal coordinate in m_e^(1/2) bohr (the Q of dipgrad, cff, ...), so d is in m_e^(-1/2).
    Each mode then has sum_a m_a |d_a|^2 = 1, with the atoms' masses m_a in electron masses.

    modes: shape (n_modes, atoms, 3); each mode's direction, in any length
    masses_amu: the atoms' masses in amu. Given, each mode is first scaled to sum_a m_a |d_a|^2 = 1 in amu (Gaussian
        prints modes of length 1). None: the modes are scaled that way already (CFOUR's MOLDEN).
    Then 1 amu = AMU_IN_ME electron masses: each mode is divided by sqrt(AMU_IN_ME) = 42.6953.
    Example, formaldehyde HF/cc-pVTZ (Gaussian), mode 0 (1337 cm-1): sum_a m_a |c_a|^2 of the printed mode c is
    1.3698 amu (the mode's reduced mass, which Gaussian prints), so d = c / sqrt(1.3698 * 1822.888) = c / 49.97.
    """
    modes = np.asarray(modes, dtype=np.float64)
    if masses_amu is not None:
        norms = np.einsum('a,kax->k', np.asarray(masses_amu, dtype=np.float64), modes ** 2)
        modes = modes / np.sqrt(norms)[:, None, None]
    return modes / np.sqrt(AMU_IN_ME)


def dimensionless_to_mass_weighted(tensor: np.ndarray, harmonic_cm: np.ndarray, n_mode_axes: int) -> np.ndarray:
    """
    Derivatives by dimensionless normal coordinates q -> by mass-weighted normal coordinates Q, in atomic units.

    q_i = Q_i * sqrt(w_i / hbar) (see reduced_to_au), so d/dQ_i = sqrt(w_i / hbar) * d/dq_i: each mode index
    brings one factor sqrt(nu_i / Eh), with nu_i the harmonic wavenumber in cm-1 (hbar = 1, Eh = HARTREE_CM).
    The value's own unit stays (e.g. a dipole in e a0); only the coordinate changes, to m_e^(1/2) a0.

    tensor: its first n_mode_axes axes are mode indices, in the order of harmonic_cm; the other axes (x, y, z)
        get no factor
    harmonic_cm: the harmonic wavenumbers in cm-1, shape (n_modes,)
    """
    sqrt_w = np.sqrt(np.asarray(harmonic_cm, dtype=np.float64) / HARTREE_CM)    # sqrt(w_i / hbar) in au
    out = np.asarray(tensor, dtype=np.float64)
    for axis in range(n_mode_axes):                                             # one factor per mode index
        shape = [1] * out.ndim
        shape[axis] = -1
        out = out * sqrt_w.reshape(shape)
    return out


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
    fc_au = np.asarray(fc_cm, dtype=np.float64) / HARTREE_CM                    # the energy, cm-1 -> Hartree
    return dimensionless_to_mass_weighted(fc_au, harmonic_cm, fc_au.ndim)       # every index is a mode index


