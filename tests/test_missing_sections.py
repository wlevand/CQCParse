"""
A section missing from a file raises ValueError naming it (todo_cleanup Q12). Before, most names gave zeros or an
empty result (e.g. cff all zeros), the others an unclear error (IndexError, max() of an empty list).
Each case: a copy of an example with one section taken out, made in the test.
"""
import shutil

import pytest

from CQCParse.parsing import parse_from_source
from CQCParse.utils import PKG_ROOT

EXAMPLES = PKG_ROOT + '/CQCParse/files_examples'
G16_FORM = EXAMPLES + '/dftGaussian/FORM/HFcc_pVTZ/g16_inputFull_3q.out'
C4_FORM = EXAMPLES + '/refinedc4/HFcc_pVTZ'


def parse(source_type, location, name):
    return parse_from_source(['nc_sqrt_eigval', name], source_type, str(location), lvl_theory='', basis_set='')


@pytest.fixture
def g16_freqs_only(tmp_path):
    """A Gaussian output with only the NAtoms= line and the anharmonic frequency block."""
    with open(G16_FORM) as file:
        lines = file.readlines()
    start = next(i for i, line in enumerate(lines) if 'Anharmonic Infrared Spectroscopy' in line
                 or 'Vibrational Energies at Anharmonic Level' in line)
    end = next(i for i, line in enumerate(lines) if 'Dipole strengths (DS) in 10^-40 esu^2.cm^2' in line)
    natoms = next(line for line in lines if 'NAtoms=' in line)
    path = tmp_path / 'g16.out'
    path.write_text(natoms + ''.join(lines[start - 2:end + 2]))
    return path


@pytest.mark.parametrize('name, section', [
    ('B', 'rotational constants'), ('coriolis', 'rotational constants'),
    ('dipgrad', 'Electric Dipole'), ('diphess', 'Electric Dipole'),
    ('polgrad', 'Polarizability Tensor'), ('polhess', 'Polarizability Tensor'),
    ('cff', 'CUBIC FORCE CONSTANTS'), ('qff', 'QUARTIC FORCE CONSTANTS'),
    ('modes_mapping', 'The connection between this new numbering'),
    ('equilibrium_geometry', 'Standard orientation'), ('atoms', 'Standard orientation'),
    ('normal_modes', 'normal-mode table')])
def test_gaussian_missing_section(g16_freqs_only, name, section):
    with pytest.raises(ValueError, match=section):
        parse('gaussian', g16_freqs_only, name)


def test_gaussian_missing_coriolis(tmp_path):
    with open(G16_FORM) as file:
        lines = [line for line in file if 'CORIOLIS COUPLINGS' not in line]
    path = tmp_path / 'g16.out'
    path.write_text(''.join(lines))
    with pytest.raises(ValueError, match='CORIOLIS COUPLINGS'):
        parse('gaussian', path, 'coriolis')


@pytest.mark.parametrize('name, unit, other', [('dipgrad', 'Debye', 'a.u.'), ('polgrad', 'a.u.', 'Debye')])
def test_gaussian_unexpected_unit(tmp_path, name, unit, other):
    """The conversions hold only for the dipole in Debye and the polarizability in a.u. (todo R4): another unit
    line raises. Each example file has one unit line of each."""
    with open(G16_FORM) as file:
        text = file.read()
    path = tmp_path / 'g16.out'
    path.write_text(text.replace(f'Unit of the property: {unit}', f'Unit of the property: {other}'))
    with pytest.raises(ValueError, match='unit line'):
        parse('gaussian', path, name)


def edit_copy(tmp_path, filename, keep):
    """A copy of C4_FORM, with only the lines of filename for which keep(line) is true."""
    folder = tmp_path / 'c4'
    shutil.copytree(C4_FORM, folder)
    with open(folder / filename) as file:
        lines = [line for line in file if keep(line)]
    (folder / filename).write_text(''.join(lines))
    return folder


def before(marker):
    """keep() for edit_copy: the lines before the first one holding marker."""
    seen = []

    def keep(line):
        if marker in line:
            seen.append(line)
        return not seen
    return keep


@pytest.mark.parametrize('filename, keep, name, message', [
    ('cubic', lambda line: False, 'cff', 'no force constants'),
    ('quartic', lambda line: False, 'qff', 'no force constants'),
    ('dipolex', lambda line: False, 'dipgrad', 'no dipole derivatives'),
    ('MOLDEN', lambda line: False, 'atoms', 'FR-COORD'),
    ('MOLDEN', before('[FR-NORM-COORD]'), 'normal_modes', 'FR-NORM-COORD'),
    ('out', lambda line: 'Coriolis Zeta matrix' not in line, 'coriolis', 'Coriolis Zeta matrix'),
    ('out', lambda line: 'B in cm-1:' not in line, 'B', 'B in cm-1'),
    ('out', lambda line: 'All levels with up to three quanta' not in line, 'nc_sqrt_eigval',
     'All levels with up to three quanta')])
def test_cfour_missing_section(tmp_path, filename, keep, name, message):
    with pytest.raises(ValueError, match=message):
        parse('cfour', edit_copy(tmp_path, filename, keep), name)
