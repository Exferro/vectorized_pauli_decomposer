import numpy as np

from ..core.popcount import popcount, popcount_


def fast_dense_decompose(target: np.ndarray = None,
                         pauli_indices: np.ndarray = None,
                         qubit_num: int = None):
    assert target is not None
    assert pauli_indices is not None
    assert qubit_num is not None

    assert target.shape[-1] == 2**qubit_num
    assert target.shape[-2] == 2**qubit_num

    pauli_xy_masks = pauli_indices >> qubit_num
    pauli_yz_masks = np.bitwise_and(pauli_indices, 2**qubit_num - 1)
    pauli_y_masks = np.bitwise_and(pauli_xy_masks, pauli_yz_masks)

    non_zero_rows = np.tile(np.reshape(np.arange(2**qubit_num), (1, -1)),
                            (pauli_indices.shape[0], 1))
    non_zero_cols = np.bitwise_xor(np.reshape(pauli_xy_masks, (-1, 1)),
                                   np.reshape(np.arange(2**qubit_num), (1, -1)))
    non_zero_vals = (((-1) ** popcount_(np.bitwise_and(non_zero_cols, np.expand_dims(pauli_yz_masks, axis=1))))
                     * ((1j)**popcount(np.expand_dims(pauli_y_masks, axis=1))))
    target_vals = target[non_zero_cols, non_zero_rows]

    coeffs = (non_zero_vals * target_vals).sum(axis=-1)

    return coeffs / 2**qubit_num
