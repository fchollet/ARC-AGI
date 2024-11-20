import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import json

BLACK = 0
BLUE = 1
ORANGE = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
LIGHT_ORANGE = 7
CYAN = 8
RED = 9
BORDER = 10

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
    
    
    
def visualize_problem(puzzle_id: str):
    with open(f"data/training/{puzzle_id}.json", 'r') as f:
        data = json.load(f)
    examples = data['train']
    
    input_grids = [(np.array(example['input']), np.array(example['output'])) for example in examples]
    test_grid = (np.array(data['test'][0]['input']), np.array(data['test'][0]['output']))
    # add borders to all grids
    input_grids = [(np.pad(input_grid, 1, constant_values=BORDER), np.pad(output_grid, 1, constant_values=BORDER)) for input_grid, output_grid in input_grids]
    test_grid = (np.pad(test_grid[0], 1, constant_values=BORDER), np.pad(test_grid[1], 1, constant_values=BORDER))
    
    #find the max dimensions of all the grids: input, output, and test
    max_height = max([grid.shape[0] for grid, _ in input_grids] + [test_grid[0].shape[0], test_grid[1].shape[0]])
    max_width = max([grid.shape[1] for grid, _ in input_grids] + [test_grid[0].shape[1], test_grid[1].shape[1]])
                    
    # Create a new grid to hold all the grids
    combined_height = (1+ max_height) * (len(input_grids )+1)+3
    combined_width = 2 * max_width + 1
    combined_grid = np.zeros((combined_height, combined_width), dtype=int)

    # Paste all input grids into the combined grid
    current_y = 0
    for input_grid, output_grid in input_grids:
        combined_grid[current_y:current_y + input_grid.shape[0], 0:input_grid.shape[1]] = input_grid
        combined_grid[current_y:current_y + output_grid.shape[0], 
                      input_grid.shape[1] + 1:input_grid.shape[1] + 1 + output_grid.shape[1]] = output_grid
        current_y += input_grid.shape[0] + 1
        
    # draw a line to separate the input and output grids
    combined_grid[current_y, :] = BORDER
    current_y += 2
    # Paste the test grid into the combined grid
    combined_grid[current_y:current_y + test_grid[0].shape[0], 0:test_grid[0].shape[1]] = test_grid[0]
    combined_grid[current_y:current_y + test_grid[1].shape[0], 
                  test_grid[0].shape[1] + 1:test_grid[0].shape[1] + 1 + test_grid[1].shape[1]] = test_grid[1]
    current_y += max_height
    # remove extra space at the bottom
    combined_grid = combined_grid[:current_y]
    combined_height = current_y
    # draw this grid
    border_size = 1
    cell_size = 13

    # Calculate image dimensions
    img_width = combined_width * cell_size + (combined_width + 1) * border_size
    img_height = combined_height * cell_size + (combined_height + 1) * border_size

    img = Image.new('RGB', (img_width, img_height), COLOR_TO_HEX[BORDER])
    draw = ImageDraw.Draw(img)

    # Draw colored rectangles for each cell
    for i, row in enumerate(combined_grid):
        for j, color in enumerate(row):
            x = j * (cell_size + border_size) + border_size
            y = i * (cell_size + border_size) + border_size
            draw.rectangle([x, y, x + cell_size, y + cell_size], fill=COLOR_TO_HEX[color], outline=COLOR_TO_HEX[BORDER])
    
    return img

def visualize_set(objects):
    max_height = max([obj.grid.shape[0] for obj  in objects])
    max_width = max([obj.grid.shape[1] for obj  in objects])
    # number of objects to render in each row
    closest_square = int(np.ceil(np.sqrt(len(objects))))
    
    combined_height = (len(objects) // closest_square) * (max_height+1)+1
    combined_width = closest_square * (max_width+1)+1
    
    combined_grid = np.zeros((combined_height, combined_width), dtype=int)
    current_y = 0
    current_x = 0
    for obj in objects:
        combined_grid[current_y:current_y + obj.grid.shape[0], current_x:current_x + obj.grid.shape[1]] = obj.grid
        current_x += max_width
        # draw a vertical line seperating the objects
        combined_grid[current_y:current_y + max_height, current_x+max_width] = BORDER
        current_x += 1
        if current_x >= combined_width:
            current_x = 0
            current_y += max_height+1