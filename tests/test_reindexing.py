"""
Claude Opus 4.6 
"""

from CQCParse.relay.modes_reindexing import build_permutation, reindex_tensor, clean_noise, reindex_dict
import numpy as np

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


def test_rank1_single_swap():
    """Swapping two modes in a (Nmodes, 3) tensor."""
    nmodes = 3
    mapping = {1: 2, 2: 1, 3: 3}  # A1->H2, A2->H1, A3->H3
    perm = build_permutation(mapping, nmodes)
    t = np.array([[10, 11, 12],
                  [20, 21, 22],
                  [30, 31, 32]], dtype=float)
    result = reindex_tensor(t, perm, [0])
    # H1 should get A2's data, H2 should get A1's data
    np.testing.assert_array_equal(result[0], t[1])  # H1 <- A2
    np.testing.assert_array_equal(result[1], t[0])  # H2 <- A1
    np.testing.assert_array_equal(result[2], t[2])  # H3 <- A3


def test_rank2_modes_only():
    """(Nmodes, Nmodes) tensor: both axes permuted."""
    nmodes = 3
    mapping = {1: 3, 2: 1, 3: 2}
    perm = build_permutation(mapping, nmodes)
    t = np.arange(9).reshape(3, 3).astype(float)
    result = reindex_tensor(t, perm, [0, 1])
    # result[h1, h2] = t[perm[h1], perm[h2]]
    for h1 in range(nmodes):
        for h2 in range(nmodes):
            assert result[h1, h2] == t[perm[h1], perm[h2]]


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


def test_rank3_all_modes():
    """(Nmodes, Nmodes, Nmodes): all axes permuted."""
    nmodes = 4
    mapping = {1: 4, 2: 3, 3: 2, 4: 1}
    perm = build_permutation(mapping, nmodes)
    t = np.random.rand(nmodes, nmodes, nmodes)
    result = reindex_tensor(t, perm, [0, 1, 2])
    for h1 in range(nmodes):
        for h2 in range(nmodes):
            for h3 in range(nmodes):
                assert result[h1, h2, h3] == t[perm[h1], perm[h2], perm[h3]]


def test_spatial_axes_unchanged():
    """(Nmodes, 3, 3): only axis 0 permuted, spatial axes 1,2 unchanged."""
    nmodes = 3
    mapping = {1: 2, 2: 3, 3: 1}
    perm = build_permutation(mapping, nmodes)
    t = np.random.rand(nmodes, 3, 3)
    result = reindex_tensor(t, perm, [0])
    for h in range(nmodes):
        np.testing.assert_array_equal(result[h, :, :], t[perm[h], :, :])


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

def test_rank1_reverse():
    """(Nmodes, 3): full reversal of 4 modes."""
    nmodes = 4
    mapping = {1: 4, 2: 3, 3: 2, 4: 1}  # A1->H4, A2->H3, A3->H2, A4->H1
    perm = build_permutation(mapping, nmodes)
    t = np.array([[10, 11, 12],
                  [20, 21, 22],
                  [30, 31, 32],
                  [40, 41, 42]], dtype=float)
    result = reindex_tensor(t, perm, [0])
    expected = np.array([[40, 41, 42],   # H1 <- A4
                         [30, 31, 32],   # H2 <- A3
                         [20, 21, 22],   # H3 <- A2
                         [10, 11, 12]],  # H4 <- A1
                        dtype=float)
    np.testing.assert_array_equal(result, expected)


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


def test_rank2_swap_explicit():
    """(Nmodes, Nmodes): swap two modes, check every element."""
    nmodes = 3
    mapping = {1: 2, 2: 1, 3: 3}
    perm = build_permutation(mapping, nmodes)
    # label elements as t[a1, a2] = a1*10 + a2 (1-indexed for readability)
    t = np.array([[11, 12, 13],
                  [21, 22, 23],
                  [31, 32, 33]], dtype=float)
    result = reindex_tensor(t, perm, [0, 1])
    # H1<-A2, H2<-A1, H3<-A3 on both axes
    expected = np.array([[22, 21, 23],   # result[H1,H1]=t[A2,A2], result[H1,H2]=t[A2,A1], ...
                         [12, 11, 13],
                         [32, 31, 33]], dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_rank2_mixed_explicit():
    """(Nmodes, Nmodes, 3): two mode axes + spatial, explicit values."""
    nmodes = 2
    mapping = {1: 2, 2: 1}  # swap
    perm = build_permutation(mapping, nmodes)
    t = np.array([[[10, 11, 12],    # t[A1, A1, :]
                   [20, 21, 22]],   # t[A1, A2, :]
                  [[30, 31, 32],    # t[A2, A1, :]
                   [40, 41, 42]]],  # t[A2, A2, :]
                 dtype=float)
    result = reindex_tensor(t, perm, [0, 1])
    expected = np.array([[[40, 41, 42],   # result[H1,H1,:] = t[A2,A2,:]
                          [30, 31, 32]],  # result[H1,H2,:] = t[A2,A1,:]
                         [[20, 21, 22],   # result[H2,H1,:] = t[A1,A2,:]
                          [10, 11, 12]]], # result[H2,H2,:] = t[A1,A1,:]
                        dtype=float)
    np.testing.assert_array_equal(result, expected)


def test_rank3_explicit():
    """(Nmodes, Nmodes, Nmodes): swap on 2 modes, all explicit."""
    nmodes = 2
    mapping = {1: 2, 2: 1}
    perm = build_permutation(mapping, nmodes)
    t = np.array([[[111, 112],
                   [121, 122]],
                  [[211, 212],
                   [221, 222]]], dtype=float)
    result = reindex_tensor(t, perm, [0, 1, 2])
    expected = np.array([[[222, 221],   # result[H1,H1,:] = t[A2,A2,reversed]
                          [212, 211]],
                         [[122, 121],
                          [112, 111]]], dtype=float)
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


def test_real_data():
    """
    comparing non-zero values patterns in tensors
    """
    from wilson_suite.wilson_utils.serialization import unpickle_smth_from
    # CQCParse/files_examples/FORM_conf1_B3LYP_aug_cc_pVTZ.pkl
    unpickled_g16 = unpickle_smth_from('./CQCParse/files_examples/FORM_conf1_B3LYP_aug_cc_pVTZ.pkl')
    unpickled_c4 = unpickle_smth_from('./CQCParse/files_examples/FORM_conf1_CCSD(T)_cc_pVQZ.pkl')
    
    mapping = unpickled_g16['modes_mapping']

    perm = build_permutation(mapping, nmodes=6)
    
    # set to zeros vals below tol
    dipgrad_g16 = clean_noise(unpickled_g16['dipgrad'], tol=1e-14)

    reindexed_dipgrad = reindex_tensor(dipgrad_g16, perm, [0])
    dipgrad_cfour = unpickled_c4['dipgrad']


    pattern_cfour = (dipgrad_cfour != 0)
    pattern_reindexed = (reindexed_dipgrad != 0)
    pattern_original = (dipgrad_g16 != 0)

    # Original matches reference pattern: 
    assert not np.array_equal(pattern_cfour, pattern_original)
    # Reindexed matches reference pattern:
    assert np.array_equal(pattern_cfour, pattern_reindexed)

    # ----------
    # set to zeros vals below tol
    diphess_g16 = clean_noise(unpickled_g16['diphess'], tol=1e-13)

    reindexed_diphess = reindex_tensor(diphess_g16, perm, [0,1])
    diphess_cfour = unpickled_c4['diphess']


    pattern_cfour = (diphess_cfour != 0)
    pattern_reindexed = (reindexed_diphess != 0)
    pattern_original = (diphess_g16 != 0)
    
    # Original matches reference pattern: 
    assert not np.array_equal(pattern_cfour, pattern_original)
    # Reindexed matches reference pattern:
    assert np.array_equal(pattern_cfour, pattern_reindexed)

    # ----------
    # set to zeros vals below tol
    cff_g16 = clean_noise(unpickled_g16['cff'], tol=1e-13)

    reindexed_cff = reindex_tensor(cff_g16, perm, [0,1,2])
    cff_cfour = unpickled_c4['cff']

    pattern_cfour = (cff_cfour != 0)
    pattern_reindexed = (reindexed_cff != 0)
    pattern_original = (cff_g16 != 0)
    
    # Original matches reference pattern: 
    assert not np.array_equal(pattern_cfour, pattern_original)
    # Reindexed matches reference pattern:
    assert np.array_equal(pattern_cfour, pattern_reindexed)

    # ----------
    # set to zeros vals below tol
    cff_rc_g16 = unpickled_g16['cff_rc']

    reindexed_cff_rc = reindex_tensor(cff_rc_g16, perm, [0,1,2])
    cff_rc_cfour = unpickled_c4['cff_rc']

    pattern_cfour = (cff_rc_cfour != 0)
    pattern_reindexed = (reindexed_cff_rc != 0)
    pattern_original = (cff_rc_g16 != 0)
    
    # Original matches reference pattern: 
    assert not np.array_equal(pattern_cfour, pattern_original)
    # Reindexed matches reference pattern:
    assert np.array_equal(pattern_cfour, pattern_reindexed)


    # ----------
    # set to zeros vals below tol
    qff_g16 = clean_noise(unpickled_g16['qff'], tol=1e-13)

    reindexed_qff = reindex_tensor(qff_g16, perm, [0,1,2,3])
    qff_cfour = unpickled_c4['qff']

    pattern_cfour = (qff_cfour != 0)
    pattern_reindexed = (reindexed_qff != 0)
    pattern_original = (qff_g16 != 0)
    
    # Original matches reference pattern: 
    assert not np.array_equal(pattern_cfour, pattern_original)
    # Reindexed matches reference pattern:
    assert np.array_equal(pattern_cfour, pattern_reindexed)

    # ----------
    # set to zeros vals below tol
    qff_rc_g16 = unpickled_g16['qff_rc']

    reindexed_qff_rc = reindex_tensor(qff_rc_g16, perm, [0,1,2,3])
    qff_rc_cfour = unpickled_c4['qff_rc']

    pattern_cfour = (qff_rc_cfour != 0)
    pattern_reindexed = (reindexed_qff_rc != 0)
    pattern_original = (qff_rc_g16 != 0)
    
    # Original matches reference pattern: 
    assert not np.array_equal(pattern_cfour, pattern_original)
    # Reindexed matches reference pattern:
    assert np.array_equal(pattern_cfour, pattern_reindexed)


    # ----------
    # set to zeros vals below tol
    polgrad_g16 = clean_noise(unpickled_g16['polgrad'], tol=1e-12)

    reindexed_polgrad = reindex_tensor(polgrad_g16, perm, [0])
    # set to zeros vals below tol
    polgrad_cfour = clean_noise(unpickled_c4['polgrad'], tol=1e-12)

    pattern_cfour = (polgrad_cfour != 0)
    pattern_reindexed = (reindexed_polgrad != 0)
    pattern_original = (polgrad_g16 != 0)

    # Original matches reference pattern: 
    assert not np.array_equal(pattern_cfour, pattern_original)
    # Reindexed matches reference pattern:
    assert np.array_equal(pattern_cfour, pattern_reindexed)

    # ----------
    # set to zeros vals below tol
    polhess_g16 = clean_noise(unpickled_g16['polhess'], tol=7e-9) # to match tolerance of cfour data

    reindexed_polhess = reindex_tensor(polhess_g16, perm, [0,1])
    # set to zeros vals below tol
    polhess_cfour = clean_noise(unpickled_c4['polhess'], tol=7e-9) # NOTE!: tolerance is high (data is from numerical derivatives)

    pattern_cfour = (polhess_cfour != 0)
    pattern_reindexed = (reindexed_polhess != 0)
    pattern_original = (polhess_g16 != 0)

    # Original matches reference pattern: 
    assert not np.array_equal(pattern_cfour, pattern_original)
    # Reindexed matches reference pattern:
    assert np.array_equal(pattern_cfour, pattern_reindexed)

    # ----------
    coriolis_g16 = unpickled_g16['coriolis']
    reindexed_coriolis = reindex_tensor(coriolis_g16, perm, [1,2])
    coriolis_cfour = unpickled_c4['coriolis']

    pattern_cfour = (coriolis_cfour != 0)
    pattern_reindexed = (reindexed_coriolis != 0)
    pattern_original = (coriolis_g16 != 0)

    # Original matches reference pattern: 
    assert not np.array_equal(pattern_cfour, pattern_original)
    # Reindexed matches reference pattern:
    assert np.array_equal(pattern_cfour, pattern_reindexed)

    print(list(unpickled_g16.keys()))
    for i in ['anharmonic_states', 'harmonic_states', 'nc_sqrt_eigval']:
        print(unpickled_g16[i])

def test_reindex_dict():
    print()

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


def test_real_dicts():

    from wilson_suite.wilson_utils.serialization import unpickle_smth_from
    # CQCParse/files_examples/FORM_conf1_B3LYP_aug_cc_pVTZ.pkl
    unpickled_g16 = unpickle_smth_from('./CQCParse/files_examples/FORM_conf1_B3LYP_aug_cc_pVTZ.pkl')
    
    mapping = unpickled_g16['modes_mapping']
    
    upd_mapping = {}
    for h in mapping:
        upd_mapping[h - 1] = mapping[h] - 1
    
    states_reind = reindex_dict(unpickled_g16['harmonic_states'], mapping=mapping)

    for oldkey in unpickled_g16['harmonic_states']:
        # using upd_mapping, because oldkey has numbering from 0, but in mapping it starts with 1
        newlist = sorted([upd_mapping[int(i)] for i in oldkey])
        newkey = tuple([str(i) for i in newlist])
        assert unpickled_g16['harmonic_states'][oldkey] == states_reind[newkey]

    nc_sqrt_eigval_reind = reindex_dict(unpickled_g16['nc_sqrt_eigval'], mapping=mapping)

    for oldkey in unpickled_g16['nc_sqrt_eigval']:
        # using upd_mapping, because oldkey has numbering from 0, but in mapping it starts with 1
        newkey = upd_mapping[oldkey]
        assert unpickled_g16['nc_sqrt_eigval'][oldkey] == nc_sqrt_eigval_reind[newkey]
    