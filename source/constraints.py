import numpy as np
from typing import List, Tuple
from objects import ARC_Object


# filter objects (List -> smaller List or single object)
def filter_by_color(objects: List[ARC_Object], color: int) -> List[ARC_Object]:
    """
    Filter a set of objects based on their color.
    
    """
    return [obj for obj in objects if np.all(obj.grid == color)]

def most_common(objs: list[ARC_Object]) -> ARC_Object:
    unique, count = np.unique([o.grid for o in objs], axis=0, return_counts=True)
    image = unique[np.argmax(count)]
    return ARC_Object(image, np.ones_like(image))

# filter a set of objects based on their properties
def filter_by_shape(objects: List[ARC_Object], target: ARC_Object) -> List[ARC_Object]:
    """
    Filter a set of objects based on their shape.
    
    """
    return [obj for obj in objects if (obj.grid !=0) == (target.grid !=0)]

def filter_by_size(objects: List[ARC_Object], target: ARC_Object) -> List[ARC_Object]:
    """
    Filter a set of objects based on their size (number of pixels).
    
    """
    return [obj for obj in objects if np.sum(obj.grid != 0) == np.sum(target.grid != 0)]

def manhatten_distance(obj1: ARC_Object, obj2: ARC_Object) -> Tuple[int, int]:
    return abs(obj1.top_left[0] - obj2.top_left[0]), abs(obj1.top_left[1] - obj2.top_left[1])

# retrieve useful integer properties of objects
def get_color(obj: ARC_Object) -> int:
    """
    Get the color of the object.
    np.unique[1] returns the unique values in the object's grid, excluding 0.
    """
    return np.unique(obj.grid)[1]

def get_size(obj: ARC_Object) -> int:
    """
    Get the size of the object (number of pixels).
    
    """
    return np.sum(obj.grid != 0)

def count_objects(objects: List[ARC_Object]) -> int:
    """
    Count the number of objects in the list.
    
    """
    return len(objects)

def count_mask(mask: np.ndarray) -> int:
    """
    Count the number of pixels in the mask.
    
    """
    return np.sum(mask != 0)

# retrieve useful shape properties of objects
def detect_symmetry(obj: ARC_Object) -> np.ndarray:
    matrix = obj.grid
    rows, cols = matrix.shape
    
    # Check for symmetry
    horizontal_symmetry = np.all(matrix == matrix[::-1])
    vertical_symmetry = np.all(matrix == matrix[:, ::-1])
    point_symmetry = np.all(matrix == np.flip(matrix, axis=(0, 1)))
    main_diagonal_symmetry = np.all(matrix == matrix.T)
    anti_diagonal_symmetry = np.all(matrix == np.flip(matrix.T, axis=1))
    
    if point_symmetry:
        # Generate mask for point symmetry (upper-left quadrant excluding center row and column if odd)
        half_rows = rows // 2
        half_cols = cols // 2
        mask = np.zeros_like(matrix, dtype=bool)
        mask[:half_rows, :half_cols] = True
        return mask
    
    elif horizontal_symmetry:
        # Generate mask for horizontal symmetry (upper half excluding center row if odd)
        half_rows = rows // 2
        mask = np.zeros_like(matrix, dtype=bool)
        mask[:half_rows, :] = True
        return mask
    
    elif vertical_symmetry:
        # Generate mask for vertical symmetry (left half excluding center column if odd)
        half_cols = cols // 2
        mask = np.zeros_like(matrix, dtype=bool)
        mask[:, :half_cols] = True
        return mask
    
    elif main_diagonal_symmetry:
        # Generate mask for main diagonal symmetry (upper triangle excluding diagonal)
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        return mask

    elif anti_diagonal_symmetry:
        # Generate mask for anti-diagonal symmetry (lower triangle flipped along anti-diagonal)
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)[::-1, ::-1]
        return mask
    else:
        # No symmetry detected
        return np.zeros_like(matrix, dtype=bool)
    
def get_shape(obj: ARC_Object) -> np.ndarray:
    """
    Get the shape of the object as a bit mask.
    
    """
    return obj.grid != 0


def get_contour(obj: ARC_Object) -> np.ndarray:
    """
    Get the contour of the object as a bit mask.
    
    """
    grid = obj.grid
    rows, cols = grid.shape
    contour_mask = np.zeros_like(grid)
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] != 0:
                # Check the neighbors
                neighbors = [
                    (i-1, j),  # Up
                    (i+1, j),  # Down
                    (i, j-1),  # Left
                    (i, j+1)   # Right
                ]
                for x, y in neighbors:
                    if x < 0 or x >= rows or y < 0 or y >= cols or grid[x, y] == 0:
                        contour_mask[i, j] = 1
                        break
    
    return contour_mask

# retrieve useful boolean properties of objects
def is_color(obj: ARC_Object, color: int) -> bool:
    return np.all(obj.grid == color)

def isAdjacent(obj1: ARC_Object, obj2: ARC_Object) -> bool:
    mask1 = obj1.grid != 0
    mask2 = obj2.grid != 0
    for i in range(obj1.height):
        for j in range(obj1.width):
            if mask1[i, j]:
                if i > 0 and mask2[i - 1, j]:
                    return True
                if i < obj1.height - 1 and mask2[i + 1, j]:
                    return True
                if j > 0 and mask2[i, j - 1]:
                    return True
                if j < obj1.width - 1 and mask2[i, j + 1]:
                    return True
    return False

def getOverlap(obj1: ARC_Object, obj2: ARC_Object) -> bool:
    mask1 = obj1.grid != 0
    mask2 = obj2.grid != 0
    return mask1 & mask2

