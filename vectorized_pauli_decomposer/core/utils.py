import numpy as np


def find_a_pos_in_b(a: np.ndarray = None,
                    b: np.ndarray = None):
    assert a.ndim == 1
    assert b.ndim == 1

    a_cat_b = np.concatenate((a, b))
    unq_a_cat_b, unq_a_cat_b_inv = np.unique(a_cat_b, return_inverse=True)

    # Find at which position a[idx] is in b; -1 if a[idx] is not in b
    unq_a_cat_b_pos_in_b = -1 * np.ones_like(unq_a_cat_b, dtype=np.int64)
    unq_a_cat_b_pos_in_b[unq_a_cat_b_inv[a.shape[0]:]] = np.arange(b.shape[0], dtype=np.int64)
    a_pos_in_b = unq_a_cat_b_pos_in_b[unq_a_cat_b_inv[:a.shape[0]]]

    # Find if a[idx] is in b
    unq_a_cat_b_in_b_mask = np.zeros_like(unq_a_cat_b, dtype=bool)
    unq_a_cat_b_in_b_mask[unq_a_cat_b_inv[a.shape[0]:]] = 1
    a_in_b_mask = unq_a_cat_b_in_b_mask[unq_a_cat_b_inv[:a.shape[0]]]

    return a_pos_in_b, a_in_b_mask
