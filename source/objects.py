import numpy as np

from source.util import plot_image_and_mask
from image_encoding.embedding import get_embedding


class ARC_Object:
    def __init__(self, image, mask, parent=None):
        """
            ARC_Object class to store the grid and other information of the object in the image.

            Args:
                image (numpy.ndarray): A 2D numpy array representing the image.
                mask (numpy.ndarray): A 2D numpy array representing the mask of the object.
                parent (ARC_Object): If provided, assign a pointer to your parent object.
        """
        # Get our positional information and num active pixels
        self._init_information(mask)
        self.parent = parent
        self.children = set()

        # Get grid
        self.grid = (image * mask)[self.top_left[0]:self.top_left[0] + self.height,
                                 self.top_left[1]:self.top_left[1] + self.width]


    def _init_information(self, mask):
        self.active_pixels = np.sum(mask)
        
        # Compute our positions
        x_nonzeros = np.nonzero(np.sum(mask, axis=0))[0]  # Columns with non-zero values
        y_nonzeros = np.nonzero(np.sum(mask, axis=1))[0]  # Rows with non-zero values
        self.top_left = (int(y_nonzeros[0]), int(x_nonzeros[0]))
        self.width = x_nonzeros[-1] - x_nonzeros[0] + 1
        self.height = y_nonzeros[-1] - y_nonzeros[0] + 1


    def set_parent(self, parent):
        self.parent = parent


    def add_child(self, child):
        self.children.add(child)


    def remove_child(self, child):
        self.children.discard(child)


    def get_grid(self):
        return self.grid
    

    def plot_grid(self):
        plot_image_and_mask(self.grid)


    def encode_image(self, image):
        self.embedding = get_embedding(image, display=False)