import numpy as np

from arc.board_methods import norm_pts, translational_order
from arc.concepts import Act
from arc.object import Object


def test_anchor():
    """Test the seed/anchor of objects: row, column, color."""
    pt1 = Object(1, 1, 1)
    assert pt1.seed == (1, 1, 1)
    assert pt1.anchor == (1, 1, 1)

    pt2 = Object(1, 1, 1, children=[Object(0, 0, 2), Object(1, 1)])
    assert pt2.seed == (1, 1, 1)
    assert pt2.children[0].seed == (0, 0, 2)
    assert pt2.children[0].anchor == (1, 1, 2)
    assert pt2.children[1].seed == (1, 1, 10)
    assert pt2.children[1].anchor == (2, 2, 1)


def test_comparisons():
    """Test operators between objects"""
    pt1 = Object(1, 1, 1)
    group1 = Object(1, 1, 2, children=[pt1, Object(1, 1), Object(3, 3)], bound=(2, 2))
    assert group1 != pt1
    assert not group1.sim(pt1)
    assert group1.sil(pt1)
    assert group1 == Object(2, 2, 2)
    assert group1.sim(Object(1, 1, 2))
    assert group1.sil(Object(children=[Object(1, 1, 1)]))


def test_actions():
    """Test each action on Objects"""
    pt1 = Object(1, 1, 1)
    pt2 = Act.right(pt1)
    assert pt2 == Object(1, 2, 1)
    assert pt2 != pt1

    pt3 = Act.left(Act.right(Act.down(Act.up(pt1))))
    assert pt3 == pt1
    assert pt3 != pt2


def test_board_methods():
    pts = [(1, 1, 1), (3, 1, 1)]
    seed, normed = norm_pts(pts)
    assert seed == (1, 1)
    assert normed == [(0, 0, 1), (2, 0, 1)]

    pts = [(4, 4), (3, 1)]
    seed, normed = norm_pts(pts)
    assert seed == (3, 1)
    assert normed == [(1, 3), (0, 0)]


def test_generation():
    dots = Object(0, 0, children=[Object(0, 0), Object(0, 4), Object(2, 2)])
    line = Object(4, 0, gens=["C4"])
    face = Object(5, 2, 4, children=[dots, line], name="face")
    rect = Object(0, 1, 9, gens=["R3", "C4"])
    brim = Object(3, 0, 9, gens=["C6"])
    hat = Object(1, 1, 9, children=[rect, brim])
    background = Object(0, 0, 0, gens=["R10", "C8"])
    dude = Object(children=[background, face, hat])
    assert dude.props == 47

    sim_rect = Object(1, 4, 9, gens=["R3", "C4"])
    assert rect.sim(sim_rect)
    return dude


def test_props():
    background = Object(0, 0, 0, gens=["R9", "C9"], name="BG")
    rectangle = Object(0, 0, 1, gens=["R4", "C9"], name="Rect")
    square = Object(gens=["R4", "C4"], name="Square")
    stripes = Object(0, 0, 1, gens=["RR4", "C9"], name="Lines")
    squares = Object(0, 0, 1, children=[square], gens=["RC1"])
    checkers = Object(0, 0, 1, gens=["rr4", "dd4", "rd1"])
    board1 = Object(children=[background], name="blank")
    board2 = Object(children=[background, rectangle], name="Split")
    board3 = Object(children=[background, stripes], name="Stripes")
    board4 = Object(children=[background, checkers], name="Checkers")
    board5 = Object(children=[background, squares], name="Squares")
    boards = [board1, board2, board3, board4, board5]
    assert boards == sorted(boards, key=lambda x: x.props)
    return boards


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
