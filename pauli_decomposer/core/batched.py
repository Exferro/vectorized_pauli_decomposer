import numpy as np


def batched_kron(a, b):
    assert len(a.shape) == len(b.shape)
    assert a.shape[:-2] == b.shape[:-2]

    new_a = np.repeat(a, b.shape[-2], axis=-2)
    new_a = np.repeat(new_a, b.shape[-1], axis=-1)
    new_b = np.tile(b, (*([1] * (len(a.shape) - 2)), a.shape[-1], a.shape[-2]))
    new_a *= new_b
    return new_a


def batched_rolling_kron(stack):
    assert len(stack.shape) >= 3
    stack_height = stack.shape[-3]
    result = stack[..., 0, :, :]
    for layer_idx in range(1, stack_height):
        result = batched_kron(stack[..., layer_idx, :, :], result)

    return result
