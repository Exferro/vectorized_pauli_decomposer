import numpy as np

from vectorized_pauli_decomposer.core.conversions import pauli_idx2pauli_list

from vectorized_pauli_decomposer.decomposers.fast_dense_decompose import fast_dense_decompose


if __name__ == '__main__':
    qubit_num = 8
    pauli_indices = np.arange(2**(2 * qubit_num))
    pauli_lists = pauli_idx2pauli_list(pauli_indices, qubit_num=qubit_num)

    for rep_idx in range(20):
        target = np.random.normal(size=(2 ** qubit_num, 2 ** qubit_num)) + 1j * np.random.normal(
            size=(2 ** qubit_num, 2 ** qubit_num))
        fast_coeffs = fast_dense_decompose(target=target,
                                           basis_pauli_indices=pauli_indices,
                                           qubit_num=qubit_num)
