import numpy as np


def pauli_list2pauli_idx(pauli_list: np.ndarray = None,
                         qubit_num: int = None):
    assert len(pauli_list.shape) == 2
    assert pauli_list.shape[1] == qubit_num

    assert qubit_num is not None

    powers_of_two = 2 ** np.arange(qubit_num)
    yz_part = np.sum(np.multiply(pauli_list % 2, powers_of_two), axis=-1)
    xy_part = np.sum(np.multiply(pauli_list // 2, powers_of_two), axis=-1)

    return yz_part + (xy_part << qubit_num)


def pauli_idx2pauli_list(pauli_idx: np.ndarray = None,
                         qubit_num: int = None):
    assert qubit_num is not None

    shifts = np.arange(qubit_num)
    yz_part = np.bitwise_and(pauli_idx.reshape(-1, 1) >> shifts, 1)
    xy_part = np.bitwise_and((pauli_idx.reshape(-1, 1) >> qubit_num) >> shifts, 1)

    return yz_part + 2 * xy_part


def pauli_idx2yz_mask_xy_mask(pauli_idx: np.ndarray = None,
                              qubit_num: int = None):
    assert qubit_num is not None

    return np.bitwise_and(pauli_idx, 2**qubit_num - 1), pauli_idx >> qubit_num


def yz_mask_xy_mask2pauli_idx(yz_mask: np.ndarray = None,
                              xy_mask: np.ndarray = None,
                              qubit_num: int = None):
    assert qubit_num is not None

    return yz_mask + (xy_mask << qubit_num)
