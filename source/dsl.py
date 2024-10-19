import numpy as np
from copy import deepcopy

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

def or_obj(obj1: ARC_Object, obj2: ARC_Object, color: int) -> ARC_Object:
    mask1 = obj1.grid != 0
    mask2 = obj2.grid != 0
    image = np.where(mask1 | mask2, np.where(obj1.grid == obj2.grid, obj1.grid, color), 0)
    return ARC_Object(image, np.ones_like(image))

def and_obj(obj1: ARC_Object, obj2: ARC_Object, color: int) -> ARC_Object:
    mask1 = obj1.grid != 0
    mask2 = obj2.grid != 0
    image = np.where(mask1 & mask2, np.where(obj1.grid == obj2.grid, obj1.grid, color), 0)
    return ARC_Object(image, np.ones_like(image))