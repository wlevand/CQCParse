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


def reindex_dict(states_dict, mapping):
    """
    Renumber the mode labels of a dict from A to H. mapping: A -> H, counted from 1 (from 0 if it has the key 0).
    Keys: a mode number (nc_sqrt_eigval), or a state, a tuple of mode numbers as strings sorted as strings, as the
    readers write them (e.g. ('10', '2'); () for the ground state).
    """
    shift = 0 if 0 in mapping else 1
    to_h = {a - shift: h - shift for a, h in mapping.items()}

    new_dict = {}
    for key, value in states_dict.items():
        if isinstance(key, tuple):
            new_dict[tuple(sorted(str(to_h[int(m)]) for m in key))] = value
        else:
            new_dict[to_h[key]] = value
    return new_dict

def validate_mapping(mapping: dict[int, int], nmodes: int):
    """mapping: A -> H, both counted from 1, a permutation of the nmodes modes; else ValueError."""
    if not (len(mapping) == nmodes and set(mapping) == set(mapping.values()) == set(range(1, nmodes + 1))):
        raise ValueError(f'the A to H mode table is not a permutation of modes 1 .. {nmodes}: {mapping}')


