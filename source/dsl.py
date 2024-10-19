import numpy as np
from copy import deepcopy
from scipy.stats import mode

from source.objects import ARC_Object

'''
All DSL operations return a new copy of object
'''

def color(obj: ARC_Object, color: int) -> ARC_Object:
    new_obj = deepcopy(obj)
    new_obj.grid[new_obj.grid != 0] = color
    return new_obj

def recolor(obj: ARC_Object, orig_color:int, new_color: int) -> ARC_Object:
    new_obj = deepcopy(obj)
    new_obj.grid[new_obj.grid == orig_color] = new_color
    return new_obj

def rotate(obj: ARC_Object) -> ARC_Object:
    new_obj = deepcopy(obj)
    new_obj.grid = np.rot90(new_obj.grid, k=-1)
    new_obj.height, new_obj.width = new_obj.grid.shape
    return new_obj

def flip(obj: ARC_Object, axis: int) -> ARC_Object:
    new_obj = deepcopy(obj)
    new_obj.grid = np.flip(new_obj.grid, axis=axis)
    return new_obj

def or_obj(obj1: ARC_Object, obj2: ARC_Object) -> ARC_Object:
    # color of obj1 takes precedence in new object
    mask1 = obj1.grid != 0
    mask2 = obj2.grid != 0
    image = np.where(mask1 | mask2, np.where(obj1.grid == 0, obj2.grid, obj1.grid), 0)
    return ARC_Object(image, np.ones_like(image))

def and_obj(obj1: ARC_Object, obj2: ARC_Object) -> ARC_Object:
    # color of obj1 takes precedence in new object
    mask1 = obj1.grid != 0
    mask2 = obj2.grid != 0
    image = np.where(mask1 & mask2, obj1.grid, 0)
    return ARC_Object(image, np.ones_like(image))

def most_common(objs: list[ARC_Object]) -> ARC_Object:
    unique, count = np.unique([o.grid for o in objs], axis=0, return_counts=True)
    image = unique[np.argmax(count)]
    return ARC_Object(image, np.ones_like(image))

def crop(obj: ARC_Object, top_left: (int, int), width: int, height: int) -> ARC_Object:
    image = obj.grid[top_left[1] : top_left[1] + height, top_left[0] : top_left[0] + width]
    return ARC_Object(image, np.ones_like(image))

def remove_loose(obj: ARC_Object) -> ARC_Object:
    '''
    For cleaning up grids with several clusters and random loose pixels.
    Check for all 2-by-2 sub-grids a pixel belongs to, if any one of them is fully colored;
    if none, remove the pixel.
    Still leaves some loose ends, but 2-by-2 seems to work the best overall.
    '''
    mask = obj.grid != 0
    color = np.argwhere(mask)
    for x, y in color:
        retain = False
        for i in range(max(0, x - 1), min(mask.shape[0] - 1, x) + 1):
            for j in range(max(0, y - 1), min(mask.shape[1] - 1, y) + 1):
                if i + 1 < mask.shape[0] and j + 1 < mask.shape[1]:
                    if np.all(mask[i : i + 2, j : j + 2]):
                        retain = True
                        break
            if retain:
                break
        if not retain:
            mask[x][y] = False
    image = np.where(mask, obj.grid, 0)
    return ARC_Object(image, np.ones_like(image))

def majority(objs: list[ARC_Object]) -> ARC_Object:
    height, h_count = np.unique([o.height for o in objs], axis=0, return_counts=True)
    width, w_count = np.unique([o.width for o in objs], axis=0, return_counts=True)
    h = height[np.argmax(h_count)]
    w = width[np.argmax(w_count)]
    grids = []
    # Standardize sizes of all inputs. Only handles cases when outlier is larger than others,
    # which appears to be the most common case
    for o in objs:
        grid = o.grid
        while grid.shape[0] > h:
            count = np.count_nonzero(grid, axis=1)
            if count[0] >= count[-1]:
                grid = np.delete(grid, -1, 0)
            else:
                grid = np.delete(grid, 0, 0)
        while grid.shape[1] > w:
            count = np.count_nonzero(grid, axis=0)
            if count[0] >= count[-1]:
                grid = np.delete(grid, -1, 1)
            else:
                grid = np.delete(grid, 0, 1)
        grids.append(grid)
    stacked = np.stack(grids, axis=0)
    majority, _ = mode(stacked, axis=0)
    image = majority.squeeze()
    return ARC_Object(image, np.ones_like(image))
