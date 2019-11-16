import argparse
import json
import numpy as np
import solver_utils

def solve(grid_in):
    """
    This function contains the hand-coded solution for the data in 
    b91ae062.json of the Abstraction and Reasoning Corpus (ARC)

    Transformation Description: Each box in the 3x3 input grid is replaced by
    a n x n grid of the same colour where n is the number of non-black unique
    colours in the input grid.
    
    Inputs: grid_in - A python list of lists containing the unsolved grid data
    
    Returns: grid_out - A python list of lists containing the solved grid data
    """
    # Convert to numpy array
    grid_in_np = np.array(grid_in)
    
    # ----------------------------------------------------------------------- #
    # ------------------------- Solve the problem --------------------------- #
    # ----------------------------------------------------------------------- #
    # Find the number of non black unique colours
    black = solver_utils.get_colour_code('black')
    colours = grid_in_np[grid_in_np != black]
    n = len(np.unique(colours))
    
    n_rows, n_cols = grid_in_np.shape # This is always 3 x 3
    grid_out_np = np.zeros((n_rows*n, n_cols*n), dtype='int64')
    
    for i in range(n_rows):
        for j in range(n_cols):
            colour = grid_in_np[i,j] # Colour of pixel (i, j)
            grid_out_np[i*n:(i*n)+n, j*n:(j*n)+n] = colour
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #
    
    # Convert back to list of lists
    grid_out = grid_out_np.tolist()
    
    return grid_out

if __name__=='__main__':
    # Get the file name passed from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json_file")
    args = parser.parse_args()
    
    # Read the file to a string
    with open(args.input_json_file) as f:
        text = f.read()
    
    # Convert from JSON to Python Dictionary
    data = json.loads(text)
    
    # Iterate through training grids and test grids
    for data_train in data['train']:
        solver_utils.solve_wrapper(data_train['input'], solve)
        
    for data_test in data['test']:
        solver_utils.solve_wrapper(data_test['input'], solve)
    
