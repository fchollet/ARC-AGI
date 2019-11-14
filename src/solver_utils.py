def print_grid(grid_in):
    #iterate through grid
    for row in grid_in:
        for elem in row:
            # print element followed by space
            print(elem, end=' ') 
        print() # go to new line

def solve_wrapper(data_in, solver):
    # Call the solver function
    data_out = solver(data_in)    
    
    # Print the results
    print_grid(data_out)
    print()
 
