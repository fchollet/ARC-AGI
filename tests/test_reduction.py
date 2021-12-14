import pytest

from arc import ARC


@pytest.fixture(scope="module")
def decomposition_samples() -> ARC:
    return ARC(idxs={8, 10, 16, 30})


def test_8(decomposition_samples: ARC):
    board = decomposition_samples.tasks[8].cases[0].input
    board.decompose()
    child_names = sorted([kid._id for kid in board.rep.children])
    assert child_names == [
        "Cluster(2x4)@(2, 0, 2)",
        "Rect(14x9)@(0, 0, 0)",
        "Rect(2x2)@(10, 3, 8)",
    ]


def test_10(decomposition_samples: ARC):
    board = decomposition_samples.tasks[10].cases[0].input
    board.decompose()
    child_names = sorted([kid._id for kid in board.rep.children])
    assert child_names == [
        "Line(3x1)@(6, 7, 5)",
        "Line(6x1)@(3, 3, 5)",
        "Line(8x1)@(1, 1, 5)",
        "Line(9x1)@(0, 5, 5)",
        "Rect(9x9)@(0, 0, 0)",
    ]


def test_16(decomposition_samples: ARC):
    board = decomposition_samples.tasks[16].cases[0].input
    board.decompose()
    child_names = sorted([kid._id for kid in board.rep.children])
    assert child_names == [
        "Line(3x1)@(0, 0, 3)",
        "Line(3x1)@(0, 1, 1)",
        "Line(3x1)@(0, 2, 2)",
    ]


def test_30(decomposition_samples: ARC):
    board = decomposition_samples.tasks[30].cases[0].input
    board.decompose()
    child_names = sorted([kid._id for kid in board.rep.children])
    assert child_names == [
        "Rect(2x2)@(0, 1, 2)",
        "Rect(2x2)@(1, 7, 1)",
        "Rect(2x2)@(2, 4, 4)",
        "Rect(5x10)@(0, 0, 0)",
    ]
