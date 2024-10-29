import cv2
import numpy as np
from collections import deque

from objects import ARC_Object

def extract_objects(source_object, method='color', print_on_init=False):
    """
        Given an ARC_Object and extraction method, return a list of sub-objects for that ARC_Object.

        Args:
            object (ARC_Object): The input image.
            method (str): The method to use for object extraction. Options are 'color' and 'contour'.
            print_on_init (bool): If True, print the grid upon initialization of the object.    
    """
    objects = []
    image = source_object.get_grid()

    if method == 'color':
        color_masks = get_color_masks(image)
        for mask in color_masks:
            new_object = ARC_Object(image, mask, source_object)
            if print_on_init:
                new_object.plot_grid() 
            
            objects.append(new_object)
            source_object.add_child(new_object)
    elif method == 'contour':
        contour_masks, hierarchy = get_contour_masks(image)
        for mask in contour_masks:
            new_object = ARC_Object(image, mask, source_object)
            if print_on_init:
                new_object.plot_grid() 
            
            objects.append(new_object)
            source_object.add_child(new_object)
    elif method == 'loop':
        loops = get_loops(image)
        for loop_mask in loops:
            loop_object = ARC_Object(image, loop_mask, source_object)
            if print_on_init:
                loop_object.plot_grid()
            objects.append(loop_object)
            source_object.add_child(loop_object)
    else:
        raise ValueError(f"Invalid method: {method}")

    return objects


def get_color_masks(image):
    """
    Generate masks for each color in a 2D NumPy array image where pixel values are integers from 0 to 9.

    Args:
        image (numpy.ndarray): A 2D NumPy array representing the image with integer values between 0 and 9.

    Returns:
        list of numpy.ndarray: A list of binary masks where '1' represents the presence of a specific color.
                               Masks are only returned for colors present in the image.
    """
    masks = []
    
    # Iterate over the possible values (0 to 9)
    for color in range(10):
        # Create a binary mask where the pixel value matches the current color
        mask = (image == color).astype(np.uint8)
        
        # Only append the mask if it contains any pixels (i.e., if the color is present)
        if np.any(mask):
            masks.append(mask)
    
    return masks


def get_contour_masks(image):
    """
    Apply contour detection to the input image and return the contour masks and hierarchy.

    Args:
        image (numpy.ndarray): A 2D numpy array representing the input image, with values expected to be between 0 and 9.

    Returns:
        tuple: A tuple containing:
            - list: A list of 2D numpy arrays (mask) where '1' represents contour areas for each contour.
            - numpy.ndarray: A hierarchy array describing parent-child contour relationships.
    """
    # Scale image values from 0-9 to 0-255
    scaled_image = (image * 255 / 9).astype(np.uint8)
    
    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(scaled_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a list of masks for each contour
    contour_masks = []
    for i in range(len(contours)):
        mask = np.zeros_like(scaled_image, dtype=np.uint8)
        cv2.drawContours(mask, contours, i, 1, thickness=cv2.FILLED)
        contour_masks.append(mask)
    
    return contour_masks, hierarchy

def get_loops(image):
    """
    Detect loops for each color in an integer array using flood fill with 8-connectivity.
    
    Args:
        image: 2D numpy array where integers represent different colors
        
    Returns:
        A list of Arc Objects, each representing a loop of a specific color
    """
    # Create a copy of the image to avoid modifying the original
    visited = np.zeros_like(image)
    # Dictionary to store loops for each color
    loops = []
    h,w = image.shape
                
    for i in range(h):
        for j in range(w):
            if visited[i, j] != 1:  # Unvisited pixel
                color = image[i, j]
                loop_mask = np.zeros_like(image)
                loop_coords = cbfs(image, j, i, color)
                for x, y in loop_coords:
                    loop_mask[y, x] = 1
                    visited[y, x] = 1
                    
                #find internal point
                ipx, ipy = -1,-1
                for x, y in loop_coords:
                    # Check all adjacent points
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        ipx, ipy = x + dx, y + dy
                        if (0 <= ipx < w and 
                            0 <= ipy < h and 
                            loop_mask[ipy, ipx] == 0):
                            # Count walls in each direction from this point
                            wall_counts = [0, 0, 0, 0]
                            
                            # Count walls to the right
                            wall_counts[0] = np.sum(loop_mask[ipy, ipx:])
                            # Count walls to the left
                            wall_counts[2] = np.sum(loop_mask[ipy, :ipx])
                            # Count walls down
                            wall_counts[1] = np.sum(loop_mask[ipy:, ipx])
                            # Count walls up
                            wall_counts[3] = np.sum(loop_mask[:ipy, ipx])
                            
                            # If we have walls in all directions, this is an internal point
                            if min(wall_counts) == 1:
                                break
                            
                # no internal points so not a loop
                if ipx == -1:
                    continue
                # print("found internal point: ", ipx, ipy)
                #find bounded space
                internal_space = cbfs(loop_mask, ipx, ipy, 0)
                # space is not bounded by loop
                if len(internal_space) + len(loop_coords) ==h*w:
                    continue
                bounded_space_mask = np.zeros_like(image)
                for x, y in internal_space:
                    bounded_space_mask[y, x] = 1
                
                
                
                loops.append((loop_mask, bounded_space_mask))
    return loops

# color bfs
def cbfs(image: np.array, x: int, y: int, target_color: int):
    frontier = deque([(x, y)])
    visited = set()
    while frontier:
        x, y = frontier.popleft()
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < image.shape[1] and 0 <= ny < image.shape[0] and image[ny, nx] == target_color and (nx, ny) not in visited:
                frontier.append((nx, ny))
                visited.add((nx, ny))
    return visited


def split_by_color(obj: ARC_Object) -> list[ARC_Object]:
    """
    Split an object into sub-objects based on color.

    Args:
        obj (ARC_Object): The input object to split.

    Returns:
        list[ARC_Object]: A list of sub-objects.
    """
    return extract_objects(obj, method='color')