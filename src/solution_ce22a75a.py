import sys
import numpy as np
import utils.common_utility as cu 
'''
Function to solve ARC task
---------------------------
Task Number : ce22a75a
'''

'''
Function call to utilitites package for reading the input json file.
'''

with open(sys.argv[1], 'r') as j_file:
    train_in,train_out,test_in,test_out = cu.json_src_reader(j_file)

def solve(inputs):
    '''
    The function returns a numpy array by replacing all zeros around a pattern with the corresponding pattern value.
    
    Parameters:
    -----------
    c = ARC test list
    
    Returns:
    --------
    test_result = returns a numpy ndarray, size of which depends on the input.
    '''
    test_result = np.array(inputs)
    res = np.where(test_result>0)
    try:
        
        for i in list(zip(res[0], res[1])):     
            for x in range(-1,2):
                for y in range(-1,2):
                    if (x != 0 or y != 0):
                        test_result[i[0]-x][i[1]-y] = 1
                        test_result[i[0]][i[1]] = 1 
    except IndexError: # catch the error
        pass #we are passing the if any point is at the corner                         
    return test_result


for inputs in (train_in + test_in):
    outputs = solve(inputs)
    print(outputs)
    print()

    'Additional line to visualize the test input and test output'
    #plt_list = [inputs, outputs]
    #cu.visualize(plt_list)