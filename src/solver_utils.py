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
    
def get_colour_code(colour):
    colour_mapping = {'black':0,
                      'blue':1,
                      'red':2,
                      'green':3,
                      'yellow':4,
                      'grey':5,
                      'pink':6,
                      'orange':7,
                      'babyblue':8,
                      'maroon':9}
    
    return colour_mapping[colour]
 
