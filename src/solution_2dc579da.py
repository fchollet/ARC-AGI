import argparse
import json
import numpy as np

def solve(grid_in):
    # Convert from JSON format
    grid_in_list = json.loads(grid_in)
    # Convert to numpy array
    grid_in_np = np.array(grid_in_list)
    
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
    grid_out_list = grid_out_np.tolist()
    # Convert back to JSON format
    output_grid = json.dumps(grid_out_list)
    return output_grid

def print_grid(grid_in):
    for row in grid_in:
        for elem in row:
            print(elem, end=' ')
        print()
        
def solve_wrapper(data_in):
    # Convert to JSON
    data_in_json = json.dumps(data_in)
    
    # Call the solve function
    data_out_json = solve(data_in_json)    
    
    # Convert the result from JSON
    data_out = json.loads(data_out_json)
    
    # Print the results
    print_grid(data_out)
    print()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json_file")
    args = parser.parse_args()
    
    with open(args.input_json_file) as f:
        text = f.read()
    
    data = json.loads(text)
    
    for data_train in data['train']:
        solve_wrapper(data_train['input'])
        
    for data_test in data['test']:
        solve_wrapper(data_test['input'])
    