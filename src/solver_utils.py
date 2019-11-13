import json

def print_grid(grid_in):
    #iterate through grid
    for row in grid_in:
        for elem in row:
            # print element followed by space
            print(elem, end=' ') 
        print() # go to new line
    

def solve_wrapper(data_in, solver):
    # Convert to JSON
    data_in_json = json.dumps(data_in)
    
    # Call the solve function
    data_out_json = solver(data_in_json)    
    
    # Convert the result from JSON
    data_out = json.loads(data_out_json)
    
    # Print the results
    print_grid(data_out)
    print()
 
