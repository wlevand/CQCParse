"""
Outputs that differ from the examples in ways a real run can. Each case: a copy of an example, edited in the test.
    Gaussian  an imaginary frequency; no anharmonic IR tables; the Coriolis couplings turned off; atoms other than
              H, C, N, O
    CFOUR     values run together in the levels table; an imaginary frequency; outfile0.out and pol/polar.pkl for
              out and polar.pkl; a MOLDEN of another geometry
"""
import re
import shutil

import numpy as np
import pytest

from CQCParse.parsing import parse_from_source
from CQCParse.parsing.parse_wilson_obtainer import make_cfour_files_dict
from CQCParse.parsing.parseGaussian_forWilson import parse_anharmonic_x_matrix
from CQCParse.utils import PKG_ROOT

EXAMPLES = PKG_ROOT + '/CQCParse/files_examples'
G16_FORM = EXAMPLES + '/dftGaussian/FORM/HFcc_pVTZ/g16_inputFull_3q.out'
C4_FORM = EXAMPLES + '/refinedc4/HFcc_pVTZ'
STATES = ['nc_sqrt_eigval', 'harmonic_states', 'anharmonic_states']


def parse(source_type, location, names):
    return parse_from_source(names, source_type, str(location), lvl_theory='', basis_set='')


def g16_copy(tmp_path, edit):
    """A copy of G16_FORM whose text is edit(text)."""
    with open(G16_FORM) as file:
        text = file.read()
    path = tmp_path / 'g16.out'
    path.write_text(edit(text))
    return path


def c4_copy(tmp_path, filename, edit):
    """A copy of the folder C4_FORM whose file filename has the text edit(text)."""
    folder = tmp_path / 'c4'
    shutil.copytree(C4_FORM, folder)
    text = (folder / filename).read_text()
    (folder / filename).write_text(edit(text))
    return folder


def test_gaussian_imaginary_frequency(tmp_path):
    """A negative harmonic frequency (Gaussian's way to print an imaginary one) raises. Unchecked, cff and qff would
    be NaN: their conversion takes the square root of each frequency."""
    path = g16_copy(tmp_path, lambda text: text.replace('3084.240', '-3084.24'))    # nu1, in the anharmonic tables
    with pytest.raises(ValueError, match='imaginary frequency'):
        parse('gaussian', path, ['cff'])


def test_gaussian_without_ir_tables(tmp_path):
    """The end line of the anharmonic IR tables, which closes the states block, taken out (an output without those
    tables has no such line): the states come from the section "Vibrational Energies at Anharmonic Level" alone, the
    same as from the whole file. Before, this raised."""
    path = g16_copy(tmp_path, lambda text: text.replace('Dipole strengths (DS) in 10^-40 esu^2.cm^2', ''))
    edited, whole = parse('gaussian', path, STATES), parse('gaussian', G16_FORM, STATES)
    for name in STATES:
        assert edited[name] == whole[name]


def test_gaussian_x_matrix_coriolis_turned_off(tmp_path):
    """With the Coriolis couplings turned off, the X matrix section prints a note where the Coriolis matrix stands:
    the X matrix reader raises and names the matrix."""
    def edit(text):
        return re.sub(r'(Coriolis contributions to X Matrix.*?\n).*?( *3rd Deriv\. contributions to X Matrix)',
                      r'\1 NOTE: Calculations of Coriolis contributions deactivated.\n\n\2', text, count=1, flags=re.DOTALL)
    path = g16_copy(tmp_path, edit)
    with open(path) as file:
        lines = file.readlines()
    with pytest.raises(ValueError, match='Coriolis contributions'):
        parse_anharmonic_x_matrix(lines)


def test_gaussian_coriolis_turned_off(tmp_path):
    """A run with the Coriolis couplings turned off has no "CORIOLIS COUPLINGS" section: B is still read, and
    coriolis raises, saying why."""
    def edit(text):
        return text.replace('CORIOLIS COUPLINGS', '') + ' NOTE: Calculations of Coriolis couplings deactivated\n'
    path = g16_copy(tmp_path, edit)
    np.testing.assert_array_equal(parse('gaussian', path, ['B'])['B'], parse('gaussian', G16_FORM, ['B'])['B'])
    with pytest.raises(ValueError, match='the run turned them off'):
        parse('gaussian', path, ['coriolis'])


@pytest.mark.parametrize('number, symbol', [(9, 'F'), (16, 'S'), (17, 'Cl'), (35, 'Br')])
def test_gaussian_elements(tmp_path, number, symbol):
    """The atomic number of atom 1 (O) changed in the geometry blocks of the final geometry; the masses, and so the
    geometry checks, stay O's."""
    path = g16_copy(tmp_path, lambda text: re.sub(r'^(\s+1\s+)8(\s+0\s+0\.000000\s+0\.000000\s+0\.661432)$',
                                                  rf'\g<1>{number}\g<2>', text, flags=re.MULTILINE))
    assert parse('gaussian', path, ['atoms'])['atoms'] == [symbol, 'C', 'H', 'H']


def test_cfour_values_run_together(tmp_path):
    """
    CFOUR's levels table has fixed-width columns: a value that fills its whole column has no blank before it, and
    one too wide for its column is printed as stars. Here nu7's intensity is 13283.548330 km/mol (in the file
    2.396791), and nu8's is stars:
        7  0  0  0  0  1  0  0  0  0  1320.21013283.548330   1337.151674
        8  0  0  0  0  1  0  0  0  0  1351.528************   1370.508073
    The states are those of the unedited file. Before, the first row raised.
    """
    def edit(text):
        return (text.replace('1320.210    2.396791', '1320.21013283.548330')
                .replace('1351.528   23.647074', '1351.528************'))
    edited, whole = parse('cfour', c4_copy(tmp_path, 'out', edit), STATES), parse('cfour', C4_FORM, STATES)
    for name in STATES:
        assert edited[name] == whole[name]


def test_cfour_imaginary_frequency(tmp_path):
    folder = c4_copy(tmp_path, 'out', lambda text: text.replace('2.396791   1337.151674', '2.396791  -1337.151674'))
    with pytest.raises(ValueError, match='imaginary frequency'):
        parse('cfour', folder, ['nc_sqrt_eigval'])


def test_cfour_other_file_names(tmp_path):
    """A folder with outfile0.out and pol/polar.pkl instead of out and polar.pkl gives the same result. A folder with
    both out and outfile0.out (the CCSD(T) example, whose outfile0.out has no levels table) uses out."""
    folder = tmp_path / 'c4'
    shutil.copytree(C4_FORM, folder)
    (folder / 'out').rename(folder / 'outfile0.out')
    (folder / 'pol').mkdir()
    (folder / 'polar.pkl').rename(folder / 'pol' / 'polar.pkl')
    names = STATES + ['B', 'coriolis', 'dipgrad', 'polgrad', 'cff', 'qff', 'equilibrium_geometry']
    edited, whole = parse('cfour', folder, names), parse('cfour', C4_FORM, names)
    for name in names:
        if isinstance(whole[name], dict):
            assert edited[name] == whole[name]
        else:
            np.testing.assert_array_equal(edited[name], whole[name])
    assert make_cfour_files_dict(EXAMPLES + '/refinedc4/CCSDTcc_pVQZ')['out_file'].endswith('/out')


def test_cfour_molden_of_another_geometry(tmp_path):
    """A MOLDEN whose geometry is not the out file's reference structure (here O moved by 0.01 bohr): the geometry
    and the normal modes raise, the atoms are still read. Before, both were returned with no error."""
    def edit(text):
        head, geometry = text.split('[FR-COORD]')       # the geometry pMOLDEN reads; MOLDEN has others before it
        return head + '[FR-COORD]' + geometry.replace('1.1131676508', '1.1231676508', 1)
    folder = c4_copy(tmp_path, 'MOLDEN', edit)
    for name in ('equilibrium_geometry', 'normal_modes'):
        with pytest.raises(ValueError, match='not the reference structure'):
            parse('cfour', folder, [name])
    assert parse('cfour', folder, ['atoms'])['atoms'] == ['O', 'C', 'H', 'H']
