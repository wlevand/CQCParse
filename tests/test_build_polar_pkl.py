import pickle

import numpy as np
import pytest

from CQCParse.tools.build_polar_pkl import getRotationMatrix, polar_derivatives, pTensor
from CQCParse.utils import PKG_ROOT

C4 = PKG_ROOT + '/CQCParse/files_examples/refinedc4'


def load(path):
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


@pytest.mark.parametrize('folder', ['CCSDTcc_pVQZ', 'HFcc_pVQZ', 'HFcc_pVTZ'])
def test_rebuild_polar_pkl(folder):
    """
    polar_derivatives rebuilds polar.pkl from the displaced runs' POLAR and OMAT in polarData_raw.pkl (todo_cleanup
    Q5, todo R5). The rotation as in getDisplacementsPolarData; the reference unrotated: pklPoldata stored it rotated
    by its own OMAT, so undo that.
    """
    raw = load(f'{C4}/{folder}/polarData_raw.pkl')
    alpha_eq, omat_eq = raw.pop('equil')
    rotated = {name: np.einsum('ij,jk->ik', np.einsum('ij,jk->ik', omat.T, alpha), omat)
               for name, (alpha, omat) in raw.items()}
    first, second = polar_derivatives(rotated, omat_eq @ alpha_eq @ omat_eq.T)

    first_pkl, second_pkl = load(f'{C4}/{folder}/polar.pkl')
    np.testing.assert_array_equal(first, first_pkl)
    # the reference goes through the rotation and back: rounding 1e-15, divided by h^2 = 1e-4 on the diagonal
    np.testing.assert_allclose(second, second_pkl, rtol=0, atol=1e-9)
