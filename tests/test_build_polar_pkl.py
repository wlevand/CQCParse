import pickle

import numpy as np
import pytest

from CQCParse.tools.build_polar_pkl import getRotationMatrix, polar_derivatives, pTensor
from CQCParse.utils import PKG_ROOT

C4 = PKG_ROOT + '/CQCParse/files_examples/refinedc4'


def load(path):
    """The content of a pickle file (polar.pkl, polarData_raw.pkl)."""
    with open(path, 'rb') as file:
        return pickle.load(file)


def test_getRotationMatrix():
    rotation_matrix = getRotationMatrix(C4 + '/HFcc_pVDZ/outfile0.out')
    ref_rotmat = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    assert np.allclose(rotation_matrix, ref_rotmat)


def test_pTensor_polar():
    alpha = pTensor(C4 + '/HFcc_pVDZ/POLAR')
    assert alpha.shape == (3, 3)
    np.testing.assert_allclose(alpha, alpha.T, atol=1e-12)


def runs(folder):
    """
    The inputs of polar_derivatives from a folder's polarData_raw.pkl: the displaced runs' POLAR, rotated as in
    getDisplacementsPolarData, and the reference unrotated (pklPoldata stored it rotated by its own OMAT, so undo that).
    """
    raw = load(f'{C4}/{folder}/polarData_raw.pkl')
    alpha_eq, omat_eq = raw.pop('equil')
    rotated = {name: np.einsum('ij,jk->ik', np.einsum('ij,jk->ik', omat.T, alpha), omat)
               for name, (alpha, omat) in raw.items()}
    return rotated, omat_eq @ alpha_eq @ omat_eq.T


@pytest.mark.parametrize('folder', ['CCSDTcc_pVQZ', 'HFcc_pVQZ', 'HFcc_pVTZ'])
def test_rebuild_polar_pkl(folder):
    """polar_derivatives rebuilds polar.pkl from the displaced runs' POLAR and OMAT in polarData_raw.pkl."""
    first, second = polar_derivatives(*runs(folder))

    first_pkl, second_pkl = load(f'{C4}/{folder}/polar.pkl')
    np.testing.assert_array_equal(first, first_pkl)
    # the reference goes through the rotation and back: rounding 1e-15, divided by h^2 = 1e-4 on the diagonal
    np.testing.assert_allclose(second, second_pkl, rtol=0, atol=1e-9)


def test_reference_from_another_calculation():
    """A reference off by 1e-4 au, which would put 2 au into every diagonal term, raises; every mode pair shows it."""
    rotated, reference = runs('HFcc_pVTZ')
    with pytest.raises(ValueError, match='15 of 15 mode pairs give another reference'):
        polar_derivatives(rotated, reference + 1e-4)


def test_pair_run_from_another_calculation():
    """One pair run off by 1e-4 au raises, and the error names its pair (CFOUR modes 9 and 11)."""
    rotated, reference = runs('HFcc_pVTZ')
    rotated['9_11pn'] = rotated['9_11pn'] + 1e-4
    with pytest.raises(ValueError, match=r'1 of 15 mode pairs .* 9_11 2\.5e-05'):
        polar_derivatives(rotated, reference)
