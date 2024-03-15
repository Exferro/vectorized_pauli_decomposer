import numpy as np

PAULI_I = np.asarray(
    [[1.0, 0.0],
     [0.0, 1.0]],
    dtype=np.cdouble
)

PAULI_X = np.asarray(
    [[0.0, 1.0],
     [1.0, 0.0]],
    dtype=np.cdouble
)

PAULI_Y = np.asarray(
    [[0.0, -1.0j],
     [1.0j, 0.0]],
    dtype=np.cdouble
)

PAULI_Z = np.asarray(
    [[1.0, 0.0],
     [0.0, -1.0]],
    dtype=np.cdouble
)

CORE_PAULI_STACK = np.asarray([PAULI_I, PAULI_Z, PAULI_X, PAULI_Y])
