from typing import List
from PIL import Image
import numpy as np

COLOR_MAPPING = {
    0: "#000000",
    1: "#0074D9",
    2: "#FF4136",
    3: "#2ECC40",
    4: "#FFDC00",
    5: "#AAAAAA",
    6: "#F012BE",
    7: "#FF851B",
    8: "#7FDBFF",
    9: "#870C25",
}


def image_from_matrix(matrix: List[List[int]] | np.ndarray) -> Image:
    """
    Convert a matrix of integers into an image, using hexadecimal color codes.
    """
    if isinstance(matrix, list):
        matrix = np.array(matrix)
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Matrix must be a list or numpy array")

    # Convert matrix to uint8 type for image representation
    matrix = matrix.astype(np.uint8)

    # Create an empty image with the same shape as the matrix
    height, width = matrix.shape
    image = Image.new("RGB", (width, height))

    # Set each pixel in the image according to the color mapping
    for y in range(height):
        for x in range(width):
            color_hex = COLOR_MAPPING[matrix[y, x]]
            rgb_tuple = tuple(
                int(color_hex[i : i + 2], 16) for i in (1, 3, 5)
            )  # Convert hex to RGB tuple
            image.putpixel((x, y), rgb_tuple)

    return image


if __name__ == "__main__":
    matrix = np.random.randint(0, 10, (32, 32))
    image = image_from_matrix(matrix)

    # save the image
    image.save("image.png")
