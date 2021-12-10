from typing import Literal
import numpy as np

# Compiling certain methods gives some speedup
# import numba as nb

from arc.util import logger
from arc.definitions import Constants as cst

log = logger.fancy_logger("BoardMethods", level=30)


def norm_pts(points):
    """Used during Object init, ensures normalized seed"""
    minrow, mincol = cst.MAX_ROWS, cst.MAX_COLS
    for pt in points:
        minrow = min(minrow, pt[0])
        mincol = min(mincol, pt[1])
    result = []
    if len(points[0]) == 3:
        for pt in points:
            result.append((pt[0] - minrow, pt[1] - mincol, pt[2]))
    elif len(points[0]) == 2:
        for pt in points:
            result.append((pt[0] - minrow, pt[1] - mincol))
    return (minrow, mincol), result


def norm_children(children):
    """Makes sure the parent/kid position relationship is normalized"""
    if not children:
        return (0, 0)
    minrow, mincol = cst.MAX_ROWS, cst.MAX_COLS
    for obj in children:
        minrow = min(minrow, obj.row)
        mincol = min(mincol, obj.col)
    for obj in children:
        obj.row -= minrow
        obj.col -= mincol
    return (minrow, mincol)


def layer_pts(objects, bound=(cst.MAX_ROWS, cst.MAX_COLS)):
    """Handles occlusion due to layering in objects
    Startiwith the lowest layer, and assign points to a dictionary
    in the form {(x, y): color}
    """
    pts = {}
    maxrow, maxcol = bound
    for obj in objects:
        for pt in obj.pts:
            if pt[0] < maxrow and pt[1] < maxcol:
                pts[(pt[0], pt[1])] = pt[2]
    ordered = sorted([(*pos, color) for pos, color in pts.items()])
    return ordered


def grid_filter(grid, colors):
    colors = [colors] if isinstance(colors, int) else colors
    results = []
    for color in colors:
        match_pts = list(zip(*np.where(grid == color)))
        results.append({"color": color, "pos": match_pts})

    if len(colors) == 1:
        mask = (grid != colors[0]) & (grid != cst.NULL_COLOR)
        other_pts = list((*pt, grid[pt]) for pt in zip(*np.where(mask)))
        results.append({"pts": other_pts})
    return results


def intersect(grids):
    base = grids[0].copy()
    for comp in grids[1:]:
        if base.shape != comp.shape:
            base[:, :] = cst.MARKED_COLOR
            return base
        base[base != comp] = cst.MARKED_COLOR
    return base


def expand(grid, mult):
    M, N = grid.shape
    out = np.full((M * mult[0], N * mult[1]), -1)
    for i in range(M):
        rows = slice(i * mult[0], (i + 1) * mult[0], 1)
        for j in range(N):
            cols = slice(j * mult[1], (j + 1) * mult[1], 1)
            out[rows, cols] = grid[i, j]
    return out


def color_connect(marked, max_ct=10):
    """Try connecting groups of points based on colors

    If we only produce 1 group, or more than max_ct, we Fail.
    If all groups are only size 1, we Fail.
    """
    blobs = []
    max_size = 0
    for start in zip(*np.where(marked != cst.MARKED_COLOR)):
        if marked[start] == cst.MARKED_COLOR:
            continue
        pts = get_blob(marked, start)
        max_size = max(max_size, len(pts))
        blobs.append(pts)
        if len(blobs) > max_ct:
            return [], True
    if len(blobs) <= 1:
        return [], True
    elif max_size <= 1:
        return [], True
    return blobs, False


def get_blob(marked, start):
    M, N = marked.shape
    pts = [(*start, marked[start])]
    marked[start] = cst.MARKED_COLOR
    idx = 0
    while idx < len(pts):
        c_row, c_col, _ = pts[idx]
        idx += 1
        for dr, dc in cst.ALL_STEPS:
            new_r, new_c = (c_row + dr, c_col + dc)
            if 0 <= new_r < M and 0 <= new_c < N:
                if marked[new_r][new_c] != cst.MARKED_COLOR:
                    pts.append((new_r, new_c, marked[new_r][new_c]))
                    marked[new_r][new_c] = cst.MARKED_COLOR
    return pts


# @nb.njit
def _eval_mesh(grid: np.ndarray, stride: int) -> tuple[int, float]:
    """Compiled subroutine to measure order in a strided grid"""
    R, C = grid.shape
    hits = 0
    # cst.NULL_COLOR is hardcoded here
    n_colors = 10 + 1
    for j in range(stride):
        active_mesh = grid[j::stride]
        rebase = (active_mesh + (n_colors * np.arange(C))).ravel()
        cts = np.bincount(rebase, minlength=n_colors * C).reshape(C, -1).T
        for k in range(C):
            hits += np.max(cts[:, k])
    # We adjust the order measurement to unbias larger order params.
    # A given order defect will fractionally count against a smaller grid more,
    # so without adjusting we will end up favoring larger order strides.
    # NOTE: The current fractional power isn't rigorously motivated...
    order = np.power(hits / grid.size, stride / R)
    return (stride, order)


def _skewroll_grid(grid: np.ndarray, skew: tuple[int, int]) -> np.ndarray:
    if 0 in skew:
        log.warning("_skewroll_grid doesn't support uniform rolling")
    result = np.ones(grid.shape, dtype=int)
    for idx, row in enumerate(grid):
        result[idx] = np.roll(row, idx * skew[1] // skew[0])

    return result


def translational_order(grid: np.ndarray, row_axis: bool) -> list[tuple[int, float]]:
    """Measure and rank the order for every 2D stride"""
    grid = grid if row_axis else grid.T
    params = []
    if grid.shape[0] == 1:
        return [(1, 1)]
    for stride in range(1, grid.shape[0] // 2 + 1):
        params.append(_eval_mesh(grid, stride))
    return sorted(params, key=lambda x: x[1], reverse=True)
