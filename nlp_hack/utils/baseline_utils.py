import json 
import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

PROMPT = """
Task Introduction:

I will give you an example problem that is part of a psychometric intelligence test. It is targeted at both humans and artificially intelligent systems that aim at emulating a human-like form of general fluid intelligence. I would like you to try to solve it in the best of your capability, hopefully outperforming human intelligence.

Description:

A test-taker is said to solve a task when, upon seeing the task for the first time, they are able to produce the correct output grid for all test inputs in the task (this includes picking the dimensions of the output grid). For each test input, the test-taker is allowed 3 trials.
The tasks are stored in JSON format. Each task JSON file contains a dictionary with two fields:
    * "train": demonstration input/output pairs. It is a list of "pairs" (typically 3 pairs).
    * "test": test input/output pairs. It is a list of "pairs" (typically 1 pair).
    A "pair" is a dictionary with two fields:
    * "input": the input "grid" for the pair.
    * "output": the output "grid" for the pair.
A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive). The smallest possible grid size is 1x1 and the largest is 30x30.
When looking at a task, a test-taker has access to inputs & outputs of the demonstration pairs, plus the input of the test pair. The goal is to construct the output grid corresponding to the test input grid, using 3 trials. "Constructing the output grid" involves picking the height and width of the output grid, then filling each cell in the grid with a symbol (integer between 0 and 9, which could be visualized as colors). Only exact solutions (all cells match the expected answer) can be said to be correct.

I need you to come up with the output corresponding to the test input.
Please, follow these steps:
    1. Clearly describe the transformations you observe in the train input/output pairs for each sample separate. To give you some potential directions, patterns can be both based on colors as well as location within the grid and how the grid shape changes. Please, be as detailed as possible and vocalize any concerns you have.
    2. Look at the descriptions of the transformations and see if you can find patters that would work generally for solving more similar cases.
    3. Apply the general solution developed in the previous steps to the test input.
    4. Evaluate whether or not the test output is the correct output. Specifically, answer the following questions: 
        - Does the transformation follows a similar pattern to the one observed in training samples?
        - Do the colors seem to be changing in a way that is consistent with the training samples?
        - Does the grid size seem to be changing in a way that is consistent with the training samples?
    5. Please provide your final thought-out solution in a form of 2D python array. Please, don't be lazy! You need to run the process thoroughly, use the knowledge you accumulated, gradually improving upon your solution.
I will then take your solutions and if any one of them is correct, you will be greatly rewarded!

Here's the data for the task:
{json_data}
"""

def load_all_json_files(folder_path):
    """
    Load all JSON files in the specified folder.

    Parameters:
    folder_path: The path to the folder containing JSON files.

    Returns:
    A list of dictionaries, each containing the contents of a JSON file.
    """
    json_data_list = {}
    
    # Get a list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out JSON files
    json_files = [file for file in files if file.endswith('.json')]
    
    # Load each JSON file
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        with open(file_path, 'r') as f:
            data = json.load(f)
            json_data_list[json_file] =data
    
    return json_data_list


# Define the color mapping
color_map = {
    0: '#000000',
    1: '#0074D9',
    2: '#FF4136',
    3: '#2ECC40',
    4: '#FFDC00',
    5: '#AAAAAA',
    6: '#F012BE',
    7: '#FF851B',
    8: '#7FDBFF',
    9: '#870C25'
}

def visualize_data(data):
    """
    Visualizes the input and output matrices in the provided data using Matplotlib.

    Parameters:
    data (dict): The data containing 'train' and 'test' sets with 'input' and 'output' matrices.
    """
    def plot_matrices(matrices, title_prefix, start_row, axes):
        for i, matrix in enumerate(matrices):
            input_matrix = np.array(matrix['input'])
            output_matrix = np.array(matrix['output'])

            ax_input = axes[start_row + i, 0]
            ax_output = axes[start_row + i, 1]

            # Create custom colormap
            cmap = mcolors.ListedColormap([color_map[num] for num in range(10)])
            bounds = np.arange(-0.5, 10, 1)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            # Plot input matrix
            im1 = ax_input.imshow(input_matrix, cmap=cmap, norm=norm)
            ax_input.set_title(f'{title_prefix} Input {i+1}')
            ax_input.set_xticks(np.arange(-.5, input_matrix.shape[1], 1))
            ax_input.set_yticks(np.arange(-.5, input_matrix.shape[0], 1))
            ax_input.grid(which='both', color='gray', linestyle='-', linewidth=1)
            ax_input.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

            # Plot output matrix
            if title_prefix != 'Test':
                im2 = ax_output.imshow(output_matrix, cmap=cmap, norm=norm)
                ax_output.set_title(f'{title_prefix} Output {i+1}')
                ax_output.set_xticks(np.arange(-.5, output_matrix.shape[1], 1))
                ax_output.set_yticks(np.arange(-.5, output_matrix.shape[0], 1))
                ax_output.grid(which='both', color='gray', linestyle='-', linewidth=1)
                ax_output.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            else:
                ax_output.axis('off')

    num_train = len(data.get('train', []))
    num_test = len(data.get('test', []))
    total_matrices = num_train + num_test

    fig, axes = plt.subplots(total_matrices, 2, figsize=(10, 5 * total_matrices))

    # Plot train data
    if 'train' in data:
        plot_matrices(data['train'], 'Train', 0, axes)

    # Plot test data
    if 'test' in data:
        plot_matrices(data['test'], 'Test', num_train, axes)

    plt.tight_layout()
    plt.savefig('output.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

def compare_model_output_and_gt(model_output, ground_truth):
    """
    Compares the model output and ground truth matrices using Matplotlib.

    Parameters:
    model_output (list of list of int): The model output matrix.
    ground_truth (list of list of int): The ground truth matrix.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 10))

    # Create custom colormap
    cmap = mcolors.ListedColormap([color_map[num] for num in range(10)])
    bounds = np.arange(-0.5, 10, 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot model output
    im1 = axes[0].imshow(model_output, cmap=cmap, norm=norm)
    axes[0].set_title('Model output')
    axes[0].set_xticks(np.arange(-.5, len(model_output[0]), 1))
    axes[0].set_yticks(np.arange(-.5, len(model_output), 1))
    axes[0].grid(which='both', color='gray', linestyle='-', linewidth=1)
    axes[0].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Plot ground truth
    im2 = axes[1].imshow(ground_truth, cmap=cmap, norm=norm)
    axes[1].set_title('Ground truth')
    axes[1].set_xticks(np.arange(-.5, len(ground_truth[0]), 1))
    axes[1].set_yticks(np.arange(-.5, len(ground_truth), 1))
    axes[1].grid(which='both', color='gray', linestyle='-', linewidth=1)
    axes[1].tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    plt.tight_layout()
    plt.savefig('comparison.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()