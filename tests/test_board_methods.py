import numpy as np

from arc.board_methods import norm_pts, translational_order


def test_norm_pts():
    pts = [(1, 1, 1), (3, 1, 1)]
    seed, normed = norm_pts(pts)
    assert seed == (1, 1)
    assert normed == [(0, 0, 1), (2, 0, 1)]

    pts = [(4, 4), (3, 1)]
    seed, normed = norm_pts(pts)
    assert seed == (3, 1)
    assert normed == [(1, 3), (0, 0)]


def _disorder(grid, seed=7):
    messy = grid.copy()
    rows, cols = grid.shape
    for ct in range(1, rows * cols, seed):
        i, j = (ct // cols) % rows, ct % cols
        messy[i, j] = (messy[i, j] + 1) % 11
    return messy


def _get_leading_order(grid):
    row_o = translational_order(grid, True)
    col_o = translational_order(grid, False)
    return (row_o[0][0], col_o[0][0])


def test_order():
    tile2x2 = np.tile([[1, 2], [3, 4]], (3, 3))
    assert (2, 2) == _get_leading_order(tile2x2)
    tile1x4 = np.tile([np.arange(4)], (8, 2))
    assert (1, 4) == _get_leading_order(tile1x4)
    tile2x5 = np.tile([np.arange(5), np.arange(5) + 1], (2, 2))
    assert (2, 5) == _get_leading_order(tile2x5)
    tile4x4 = np.tile([np.arange(4) + i for i in range(4)], (2, 2))
    assert (4, 4) == _get_leading_order(tile4x4)
    assert (2, 2) == _get_leading_order(_disorder(tile2x2))
    assert (1, 4) == _get_leading_order(_disorder(tile1x4))
    assert (4, 4) == _get_leading_order(_disorder(tile4x4))
