# A few scattered constants
class Constants:
    N_TRAIN = 400
    FOLDER_TRAIN = "../data/training"
    NULL_COLOR = 10
    MARKED_COLOR = -1

    MAX_ROWS = 30
    MAX_COLS = 30

    START_DIST = 10000

    STEPS_BASE = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    STEPS_DIAG = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    ALL_STEPS = STEPS_BASE + STEPS_DIAG

    CONTEXT_PREFIX = "_CTXT"

    cname = {
        -1: "Trans",
        0: "Black",
        1: "Blue",
        2: "Red",
        3: "Green",
        4: "Yellow",
        5: "Gray",
        6: "Magenta",
        7: "Orange",
        8: "SkyBlue",
        9: "Brown",
        10: "Trans",
    }
