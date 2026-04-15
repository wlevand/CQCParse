"""
Claude Opus 4.6 
"""

import numpy as np


def build_permutation(mapping: dict[int, int], nmodes: int) -> np.ndarray:
    """Build array perm such that perm[h] = a (0-indexed).
    mapping is A(1-indexed) -> H(1-indexed)."""
    perm = np.empty(nmodes, dtype=int)
    for a, h in mapping.items():
        perm[h - 1] = a - 1
    return perm


def reindex_tensor(tensor: np.ndarray, perm: np.ndarray, mode_axes: list[int]) -> np.ndarray:
    """Reindex a tensor from A-numbering to H-numbering along the specified axes."""
    result = tensor
    for ax in mode_axes:
        result = np.take(result, perm, axis=ax)
    return result


def clean_noise(arr, tol=1e-10):
    """
    set to zero vals below tolerace
    """
    return np.where(np.abs(arr) < tol, 0.0, arr)

def reindex_dict(states_dict, mapping):
    """
    Will updata mapping here for numbering from 0, if no 0 key in mapping
    """
    new_dict = {}
    
    # to python numbering from 0
    upd_mapping = {}

    if 0 not in mapping:
        for h in mapping:
            upd_mapping[h - 1] = mapping[h] - 1
    else:
        upd_mapping = mapping

    # for nc_sqrt_eigval dict
    if all([isinstance(i, int) for i in list(states_dict.keys())]):
        for k,v in states_dict.items():
            new_k = upd_mapping[k]
            new_dict[new_k] = v
    
    # for anharmonic_states, harmonic_states
    if all([isinstance(i, tuple) and isinstance(i[0], str) for i in list(states_dict.keys())]):
        for k,v in states_dict.items():
            new_list_int = sorted([upd_mapping[int(i)] for i in k])
            new_k = tuple([str(i) for i in new_list_int])
            new_dict[new_k] = v

    return new_dict

def validate_mapping(mapping: dict[int, int], nmodes: int):
    assert len(mapping) == nmodes, f"Expected {nmodes} entries, got {len(mapping)}"
    assert set(mapping.keys()) == set(range(1, nmodes + 1)), "A-indices not complete"
    assert set(mapping.values()) == set(range(1, nmodes + 1)), "H-indices not complete"


def reindex_result_data(results_dict: dict):
    """
    results_dict is generally parse_gaussian16_output return dict;
        or return of parse_from_source, when no reindexing was requested.
        
        Should contain 'modes_mapping' key and 'reindex_modes' key

    """
    if results_dict['reindex_modes']:
        raise ValueError('Reindexing had been performed on this data')
    
    validate_mapping(results_dict['modes_mapping'], len(results_dict['normal_modes']))

    perm = build_permutation(results_dict['modes_mapping'], nmodes=len(results_dict['normal_modes']))

    for key in ['coriolis', 'dipgrad', 'diphess', 'polgrad', 'polhess', 'cff', 'cff_rc', 'qff', 'qff_rc']:
        
        if key == 'coriolis':
            results_dict[key] = reindex_tensor(results_dict[key], perm, [1,2])
        elif 'grad' in key:
            results_dict[key] = reindex_tensor(results_dict[key], perm, [0])
        elif 'hess' in key:
            results_dict[key] = reindex_tensor(results_dict[key], perm, [0,1])
        elif 'cff' in key:
            results_dict[key] = reindex_tensor(results_dict[key], perm, [0,1,2])
        elif 'qff' in key:
            results_dict[key] = reindex_tensor(results_dict[key], perm, [0,1,2,3])
    
    for key in ['anharmonic_states', 'harmonic_states', 'nc_sqrt_eigval']:
        results_dict[key] = reindex_dict(results_dict[key], results_dict['modes_mapping'])

    results_dict['reindex_modes'] = True
    
    return results_dict