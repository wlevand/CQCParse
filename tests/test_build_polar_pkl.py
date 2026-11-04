import pickle
import shutil

import numpy as np
import pytest

from CQCParse.tools import build_polar_pkl
from CQCParse.tools.build_polar_pkl import (
    getPolarDers,
    getRotationMatrix,
    pklPoldata,
    polar_derivatives,
    pTensor,
)
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


def run_folders(tmp_path, monkeypatch, polar_equil=None, anharm=None):
    """
    The folders of a run, with formaldehyde HF/cc-pVTZ's displaced runs standing in for polar/. polar_equil, anharm: the
    tensor to write as that folder's POLAR, or None for no such folder. Returns polar/, and the correct reference.
    """
    rotated, reference = runs('HFcc_pVTZ')
    monkeypatch.setattr(build_polar_pkl, 'getDisplacementsPolarData', lambda folder, raw=False: dict(rotated))
    for name, tensor in (('polar_equil', polar_equil), ('anharm', anharm)):
        if tensor is not None:
            (tmp_path / name).mkdir()
            np.savetxt(tmp_path / name / 'POLAR', tensor(reference))
    (tmp_path / 'polar').mkdir()
    return str(tmp_path / 'polar'), reference


def scf_like(reference):
    """A tensor that is not the displaced runs' reference, as an anharmonic job's SCF POLAR is not."""
    return reference - 0.5


def same(reference):
    return reference


def test_reference_from_polar_equil(tmp_path, monkeypatch):
    """With both runs, polar_equil gives the reference, though anharm/POLAR is an SCF value (a correlated job)."""
    polar, _ = run_folders(tmp_path, monkeypatch, polar_equil=same, anharm=scf_like)
    first, second = getPolarDers(polar)
    expected_first, expected_second = polar_derivatives(*runs('HFcc_pVTZ'))
    np.testing.assert_array_equal(first, expected_first)
    np.testing.assert_allclose(second, expected_second, rtol=0, atol=1e-9)


def test_reference_from_anharm_without_polar_equil(tmp_path, monkeypatch):
    """Without polar_equil, anharm/POLAR gives the reference: right in HF, where SCF is the method."""
    polar, _ = run_folders(tmp_path, monkeypatch, anharm=same)
    np.testing.assert_array_equal(getPolarDers(polar)[0], polar_derivatives(*runs('HFcc_pVTZ'))[0])


def test_scf_reference_refused(tmp_path, monkeypatch):
    """Without polar_equil, in a correlated job, anharm/POLAR does not fit the displaced runs: it raises and says why."""
    polar, _ = run_folders(tmp_path, monkeypatch, anharm=scf_like)
    with pytest.raises(ValueError, match='15 of 15 mode pairs give another reference.*anharm/POLAR'):
        getPolarDers(polar)


def test_pklPoldata_reference_from_polar_equil(tmp_path, monkeypatch):
    """pklPoldata takes the reference from the same run as getPolarDers, with that run's OMAT."""
    work = tmp_path / 'work'
    work.mkdir()
    polar, reference = run_folders(work, monkeypatch, polar_equil=same, anharm=scf_like)
    shutil.copy(C4 + '/HFcc_pVDZ/outfile0.out', work / 'polar_equil')
    monkeypatch.setattr(build_polar_pkl, 'getDisplacementsPolarData', lambda folder, raw=False: {})
    monkeypatch.chdir(tmp_path)
    pklPoldata(polar)
    alpha, omat = load(tmp_path / 'polarData_raw.pkl')['equil']
    np.testing.assert_allclose(omat @ alpha @ omat.T, reference, atol=1e-12)

