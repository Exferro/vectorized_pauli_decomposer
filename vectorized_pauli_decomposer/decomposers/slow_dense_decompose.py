import numpy as np

from ..core.pauli_matrices import CORE_PAULI_STACK
from ..core.conversions import pauli_idx2pauli_list
from ..core.batched import batched_rolling_kron


def slow_dense_decompose(target: np.ndarray = None,
                          pauli_indices: np.ndarray = None,
                          qubit_num: int = None):
    assert target is not None
    assert pauli_indices is not None
    assert qubit_num is not None

    assert target.shape[-1] == 2**qubit_num
    assert target.shape[-2] == 2**qubit_num

    pauli_lists = pauli_idx2pauli_list(pauli_indices, qubit_num=qubit_num)
    pauli_stacks = CORE_PAULI_STACK[pauli_lists]

    pauli_elems = batched_rolling_kron(pauli_stacks)
    coeffs = np.einsum('bij,ji->b', pauli_elems, target)

    return coeffs / 2**qubit_num
