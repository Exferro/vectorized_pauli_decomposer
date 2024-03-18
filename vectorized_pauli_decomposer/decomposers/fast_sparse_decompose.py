import numpy as np
import scipy as sp

from ..core.conversions import yz_mask_xy_mask2pauli_idx
from ..core.utils import find_a_pos_in_b
from ..core.popcount import popcount


def fast_sparse_decompose(sparse_target: sp.sparse.csc_matrix = None,
                          basis_pauli_indices: np.ndarray = None,
                          qubit_num: int = None):
    assert sparse_target is not None
    assert isinstance(sparse_target, sp.sparse.csc_matrix)
    assert basis_pauli_indices is not None
    assert len(basis_pauli_indices.shape) == 1
    assert qubit_num is not None

    coeffs = np.zeros_like(basis_pauli_indices, dtype=np.complex128)
    non_zero_rows = np.repeat(np.arange(sparse_target.shape[0]), np.diff(sparse_target.indptr))
    non_zero_cols = sparse_target.indices
    non_zero_vals = sparse_target.data

    # Evaluate xy masks of involved Pauli indices
    xy_masks = np.bitwise_xor(non_zero_rows, non_zero_cols)
    # Create all possible yz masks
    yz_masks = np.arange(2 ** qubit_num)

    # Tile them to fit xy masks
    yz_masks = np.tile(yz_masks.reshape((1, -1)),
                       (xy_masks.shape[0], 1)).reshape((-1,))

    # Tile them to a 1D array to account for every possible yz mask
    xy_masks = np.tile(xy_masks.reshape((-1, 1)),
                       (1, 2 ** qubit_num)).reshape((-1,))
    y_masks = np.bitwise_and(xy_masks, yz_masks)

    non_zero_pauli_indices = yz_mask_xy_mask2pauli_idx(yz_mask=yz_masks,
                                                       xy_mask=xy_masks,
                                                       qubit_num=qubit_num)

    non_zero_cols = np.tile(non_zero_cols.reshape((-1, 1)),
                            (1, 2 ** qubit_num)).reshape((-1,))
    non_zero_vals = np.tile(non_zero_vals.reshape((-1, 1)),
                            (1, 2 ** qubit_num)).reshape((-1,))

    if basis_pauli_indices.shape[0] != 4 ** qubit_num:
        non_zero_pos_in_basis, non_zero_in_basis_mask = find_a_pos_in_b(a=non_zero_pauli_indices,
                                                                        b=basis_pauli_indices)
        non_zero_cols = non_zero_cols[non_zero_in_basis_mask]
        non_zero_vals = non_zero_vals[non_zero_in_basis_mask]

        yz_masks = yz_masks[non_zero_in_basis_mask]
        y_masks = y_masks[non_zero_in_basis_mask]

        basis_indices = non_zero_pos_in_basis[non_zero_in_basis_mask]
    else:
        basis_indices = non_zero_pauli_indices

    non_zero_pauli_vals = (((-1) ** popcount(np.bitwise_and(non_zero_cols, yz_masks)))
                           * ((1j) ** popcount(y_masks)))

    np.add.at(coeffs, basis_indices, non_zero_vals * non_zero_pauli_vals)

    return coeffs / 2 ** qubit_num
