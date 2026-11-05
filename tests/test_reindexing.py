"""
build_permutation, reindex_tensor and reindex_dict on small made-up data. The mappings are cycles (A1->H2, A2->H3,
A3->H1), not swaps: a swap is its own reverse, so it cannot tell A->H from H->A.
"""

import numpy as np

from CQCParse.relay.modes_reindexing import (
    build_permutation,
    reindex_dict,
    reindex_tensor,
)


def test_permutation_is_valid():
    """perm should be a proper permutation of 0..nmodes-1."""
    mapping = {12: 1, 8: 2, 7: 3, 11: 4, 6: 5, 10: 6, 5: 7, 4: 8, 3: 9, 9: 10, 2: 11, 1: 12}
    perm = build_permutation(mapping, 12)
    assert sorted(perm) == list(range(12))


def test_identity_mapping():
    """Identity mapping should leave tensors unchanged."""
    nmodes = 4
    mapping = {1: 1, 2: 2, 3: 3, 4: 4}
    perm = build_permutation(mapping, nmodes)
    t = np.random.rand(nmodes, 3)
    np.testing.assert_array_equal(reindex_tensor(t, perm, [0]), t)


def test_rank2_mixed():
    """(Nmodes, Nmodes, 3): mode axes permuted, spatial axis untouched."""
    nmodes = 3
    mapping = {1: 3, 2: 1, 3: 2}
    perm = build_permutation(mapping, nmodes)
    t = np.random.rand(nmodes, nmodes, 3)
    result = reindex_tensor(t, perm, [0, 1])
    for h1 in range(nmodes):
        for h2 in range(nmodes):
            np.testing.assert_array_equal(result[h1, h2, :], t[perm[h1], perm[h2], :])


def test_roundtrip():
    """Reindexing A->H then H->A should recover the original."""
    nmodes = 5
    mapping = {1: 3, 2: 5, 3: 1, 4: 2, 5: 4}
    perm_forward = build_permutation(mapping, nmodes)
    # inverse mapping: H->A
    inv_mapping = {h: a for a, h in mapping.items()}
    perm_inverse = build_permutation(inv_mapping, nmodes)

    t = np.random.rand(nmodes, nmodes, 3)
    t_h = reindex_tensor(t, perm_forward, [0, 1])
    t_back = reindex_tensor(t_h, perm_inverse, [0, 1])
    np.testing.assert_array_almost_equal(t_back, t)


def test_rank1_cyclic():
    """(Nmodes, 3): cyclic shift A1->H2, A2->H3, A3->H1."""
    nmodes = 3
    mapping = {1: 2, 2: 3, 3: 1}
    perm = build_permutation(mapping, nmodes)
    t = np.array([[10, 11, 12],
                  [20, 21, 22],
                  [30, 31, 32]], dtype=float)
    result = reindex_tensor(t, perm, [0])
    expected = np.array([[30, 31, 32],   # H1 <- A3
                         [10, 11, 12],   # H2 <- A1
                         [20, 21, 22]],  # H3 <- A2
                        dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_spatial33_explicit():
    """(Nmodes, 3, 3): only axis 0 permuted, spatial 3x3 blocks move as wholes."""
    nmodes = 3
    mapping = {1: 3, 2: 1, 3: 2}
    perm = build_permutation(mapping, nmodes)
    block_a1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    block_a2 = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=float)
    block_a3 = np.array([[100, 200, 300], [400, 500, 600], [700, 800, 900]], dtype=float)
    t = np.stack([block_a1, block_a2, block_a3])  # shape (3, 3, 3)
    result = reindex_tensor(t, perm, [0])
    np.testing.assert_array_equal(result[0], block_a2)   # H1 <- A2
    np.testing.assert_array_equal(result[1], block_a3)   # H2 <- A3
    np.testing.assert_array_equal(result[2], block_a1)   # H3 <- A1


def test_reindex_dict():
    states = {('0',): 2732.431, ('1',): 1787.487, ('2',): 1497.353, ('3',): 1181.244,
              ('0', '0'): 5401.529, ('2', '2'): 2996.706, ('0', '1'): 4520.625,
              ('0', '3', '3'): 6290.409, ('0', '0', '1'): 7190.43,  ('1', '2'): 3279.201,
              ('1', '2', '2'): 4772.914, ('0', '1', '2'): 5980.034, ('1', '2', '3'): 5076.067}

    states_reind = reindex_dict(states, mapping={1: 3, 2: 1, 3: 2, 0: 0})

    assert states[('1',)] == states_reind[('3',)]
    assert states[('2',)] == states_reind[('1',)]
    assert states[('1', '2')] == states_reind[('1', '3')]
    assert states[('0', '0')] == states_reind[('0', '0')]
    assert states[('2', '2')] == states_reind[('1', '1')]
    assert states[('0', '1')] == states_reind[('0', '3')]
    assert states[('0', '3', '3')] == states_reind[('0', '2', '2')]
    assert states[('0', '0', '1')] == states_reind[('0', '0', '3')]
    assert states[('1', '2', '2')] == states_reind[('1', '1', '3')]
    assert states[('0', '1', '2')] == states_reind[('0', '1', '3')]
    assert states[('1', '2', '3')] == states_reind[('1', '2', '3')]

    nc_sqrt_eigval = {0: 2884.644, 1: 1813.05, 2: 1530.173, 3: 1198.158, 4: 2939.354, 5: 1262.894}
    nc_sqrt_eigval_reind = reindex_dict(nc_sqrt_eigval, mapping={1: 3, 2: 1, 3: 2, 0: 0, 4: 5, 5: 4})

    assert nc_sqrt_eigval[0] == nc_sqrt_eigval_reind[0]
    assert nc_sqrt_eigval[1] == nc_sqrt_eigval_reind[3]
    assert nc_sqrt_eigval[2] == nc_sqrt_eigval_reind[1]
    assert nc_sqrt_eigval[3] == nc_sqrt_eigval_reind[2]
    assert nc_sqrt_eigval[4] == nc_sqrt_eigval_reind[5]
    assert nc_sqrt_eigval[5] == nc_sqrt_eigval_reind[4]
