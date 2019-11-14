import argparse
import json
import numpy as np
import solver_utils

def solve(grid_in):
    """
    This function contains the hand-coded solution for the data in 
    2dc579da.json of the Abstraction and Reasoning Corpus (ARC)
    
    Inputs: grid_in - A python list of lists containing the unsolved grid data
    
    Returns: grid_out - A python list of lists containing the solved grid data
    """
    # Convert to numpy array
    grid_in_np = np.array(grid_in)
    
    # ----------------------------------------------------------------------- #
    # ------------------------- Solve the problem --------------------------- #
    # ----------------------------------------------------------------------- #
    midpoint = grid_in_np.shape[0] // 2
    
    # Source : https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-vector
    (values,counts) = np.unique(grid_in_np, return_counts=True)
    ind = np.argmin(counts)
    minority_colour = values[ind]
    
    squares = [
            grid_in_np[0:midpoint,0:midpoint],    # Top-left
            grid_in_np[midpoint+1:,0:midpoint],   # Bottom-left
            grid_in_np[0:midpoint,midpoint+1:],   # Top-right
            grid_in_np[midpoint+1:,midpoint+1:]   # Bottom-right
            ]
    
    for square in squares:
        if minority_colour in square:
            grid_out_np = square
            break
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
    