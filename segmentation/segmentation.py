import cv2
import numpy as np

from source.objects import ARC_Object


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