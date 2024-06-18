"""Evaluation function"""

import numpy as np


def grid_edit_distance(
    x: np.ndarray,
    y: np.ndarray,
    normalize: bool = False,
):
    """Calculate the grid edit distance between two arrays."""
    num_edits = 0

    if x.shape != y.shape:
        num_edits = max(np.prod(x.shape), np.prod(y.shape))
        if normalize:
            num_edits = 1.0
    else:
        num_edits = np.sum(x != y)
        if normalize:
            num_edits /= np.prod(x.shape)

    return num_edits
