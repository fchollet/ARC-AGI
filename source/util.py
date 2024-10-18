import numpy as np
import matplotlib.pyplot as plt

COLOR_TO_HEX = {
    -1: '#FF6700',  # blaze orange
    0:  '#000000',  # black
    1:  '#1E93FF',  # blue
    2:  '#F93C31',  # orange
    3:  '#4FCC30',  # green
    4:  '#FFDC00',  # yellow
    5:  '#999999',  # grey
    6:  '#E53AA3',  # pink
    7:  '#FF851B',  # light orange
    8:  '#87D8F1',  # cyan
    9:  '#921231',  # red
    10: '#555555',  # border
}


def hex_to_rgb(hex_color):
    """ Convert a hex color to an RGB tuple with values in the range [0, 1]. """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))


def plot_tensors_with_colors(tensors):
    """ Provide as an iterable of 2D tensors. """
    
    num_examples = len(tensors)
    fig, axes = plt.subplots(1, num_examples, figsize=(num_examples * 3, 3))
    for i, tensor in enumerate(tensors):
        tensor_np = tensor.numpy()
        img_rgb = np.array([[hex_to_rgb(COLOR_TO_HEX[val]) for val in row] for row in tensor_np])
        axes[i].imshow(img_rgb, interpolation='nearest')
        axes[i].axis('off')  # Hide axes    
    plt.show()


# PLOTTING WITH MASKING
# ====================================================

def plot_image_and_mask(image, mask=None, title=""):
    """
    Plot an image tensor with the corresponding mask.

    Args:
        image (torch.Tensor): A 2D tensor representing the image, with integer values corresponding to keys in COLOR_TO_HEX.
        mask (torch.Tensor, optional): A 2D tensor representing the mask, where '1' indicates masked areas.
        title (str, optional): The title for the plot.

    Returns:
        None. The function displays the image with the mask applied (if provided).
    """
    result_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)
    
    for key, hex_value in COLOR_TO_HEX.items():
        rgb_value = hex_to_rgb(hex_value)
        result_image[image == key] = rgb_value

    if mask is not None:
        yellow_tint = np.array([255, 255, 153]) / 255.0
        result_image[mask == 1] = result_image[mask == 1] * 0.6 + yellow_tint * 0.4

    plt.imshow(result_image)
    plt.title(title)
    plt.axis('off')
    plt.show()