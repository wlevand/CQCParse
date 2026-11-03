import numpy as np
import pytest
from scipy import constants
from scipy.spatial.transform import Rotation

from CQCParse.parsing import parse_from_source
from CQCParse.parsing.parseGaussian_forWilson import (
    get_equil_geo,
    get_normal_modes,
    parse_Na,
)
from CQCParse.utils import PKG_ROOT

EXAMPLES = PKG_ROOT + '/CQCParse/files_examples'
G16_FILES = ['co/g16_inputFull_3q.out', 'dftGaussian/ACAC/B3LYPcc-pVQZ/g16_inputFull_3q.out',
             'dftGaussian/ACDM/B3LYPcc_pVQZ/g16_b3lyp_corr_inputFull_3q.out',
             'dftGaussian/FORM/B3LYPcc_pVDZ/g16_inputFull_3q.out', 'dftGaussian/FORM/B3LYPcc_pVQZ/g16_inputFull_3q.out',
             'dftGaussian/FORM/HFcc_pVQZ/g16_inputFull_3q.out', 'dftGaussian/FORM/HFcc_pVTZ/g16_inputFull_3q.out']


def read_lines(out_file):
    with open(EXAMPLES + '/' + out_file) as file:
        return [line.strip() for line in file]


MASS = {'H': 1.00782503, 'C': 12.0, 'O': 15.99491462}   # amu, the most abundant isotopes, as Gaussian uses them


def rotational_constants_cm(atoms, geometry):
    """Rotational constants (cm-1) of a geometry in angstrom; 0 moments (a linear molecule's axis) left out."""
    masses = np.array([MASS[a] for a in atoms])
    r = geometry - masses @ geometry / masses.sum()
    inertia = sum(m * (x @ x * np.eye(3) - np.outer(x, x)) for m, x in zip(masses, r, strict=True))
    moments = np.linalg.eigvalsh(inertia)
    factor = constants.h / (8 * np.pi**2 * constants.c * 100 * constants.atomic_mass * 1e-20)
    return np.sort(factor / moments[moments > 1e-8])


@pytest.mark.parametrize('out_file', G16_FILES)
def test_equil_geo_gives_b(out_file):
    """The geometry is the one of the frequencies: its rotational constants are B, which the reader takes
    from another part of the file, the anharmonic analysis at "Equilibrium Geometry"."""
    atoms, geometry = get_equil_geo(read_lines(out_file))
    b = parse_from_source(['B'], 'gaussian', EXAMPLES + '/' + out_file, lvl_theory='', basis_set='')['B']
    b = np.asarray(b, float)
    assert rotational_constants_cm(atoms, geometry) == pytest.approx(np.sort(b[b > 0]), rel=1e-5)


def test_equil_geo_acac_from_the_anharmonic_analysis():
    """ACAC's only geometry block is the start of an optimization; its frequencies come from another geometry, which
    the file prints only in the anharmonic analysis, in another frame. The reader takes that one, turned
    into the frame of the normal modes (else get_equil_geo raises), with the origin at the center of nuclear charge."""
    lines = read_lines('dftGaussian/ACAC/B3LYPcc-pVQZ/g16_inputFull_3q.out')
    atoms, geometry = get_equil_geo(lines)
    start = lines.index('Standard orientation:')
    block = np.array([line.split()[3:] for line in lines[start + 5:start + 13]], float)
    assert np.abs(geometry - block).max() > 0.1
    # atomic numbers Z: a nucleus has charge Z e, so the center of nuclear charge is sum(Z r) / sum(Z)
    z = np.array([{'H': 1, 'C': 6, 'O': 8}[a] for a in atoms])
    assert z @ geometry / z.sum() == pytest.approx(np.zeros(3), abs=1e-12)


def test_equil_geo_refuses_another_frame():
    """A geometry block turned by 30 degrees keeps its rotational constants, but no longer shares the frame of the
    normal modes: an error, not a geometry that cannot be drawn with the modes."""
    lines = read_lines('dftGaussian/FORM/HFcc_pVTZ/g16_inputFull_3q.out')
    harmonic = next(i for i, line in enumerate(lines) if 'Harmonic frequencies (cm**-1)' in line)
    start = max(i for i in range(harmonic) if 'Standard orientation:' in lines[i])
    turn = Rotation.from_euler('y', 30, degrees=True).as_matrix()
    for i in range(start + 5, start + 9):      # the 4 atom rows
        center, atomic_number, kind, *xyz = lines[i].split()
        x, y, z = turn @ np.array(xyz, float)
        lines[i] = f'{center} {atomic_number} {kind} {x:.6f} {y:.6f} {z:.6f}'
    with pytest.raises(ValueError, match='not in the frame of the normal modes'):
        get_equil_geo(lines)


@pytest.mark.parametrize('out_file', G16_FILES)
def test_normal_modes_round_to_the_2_decimal_table(out_file):
    """The 5-decimal modes (Freq=HPModes) round to the 2-decimal table Gaussian prints after them, which the reader
    reads in a file without the 5-decimal table: here, the same file without its "Coord Atom Element:" lines."""
    lines = read_lines(out_file)
    n_atoms = parse_Na(lines)
    five = get_normal_modes(lines, n_atoms)
    two = get_normal_modes([line for line in lines if 'Coord Atom Element:' not in line], n_atoms)
    assert np.abs(five - two).max() <= 0.005 + 1e-12
    assert np.abs(five - np.round(five, 5)).max() < 1e-12 and np.abs(five - two).max() > 0