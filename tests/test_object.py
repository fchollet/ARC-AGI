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


def test_flatten():
    cluster = Object(0, 0, 2, children=[Object(0, 0), Object(1, 1)])
    line = Object(0, 2, 3, gens=["R1"])
    rect = Object(0, 3, 4, gens=["R1", "C1"])
    middle_man = Object(children=[line, rect])
    root = Object(children=[cluster, middle_man])

    flat = root.flatten()[0]
    assert len(flat.children) == 3
    grid = [2, 10, 3, 4, 4] + [10, 2, 3, 4, 4]
    assert all(flat.grid.ravel() == grid)
