import sys
import utils.common_utility as cu
import numpy as np

'''
Function to solve ARC task
---------------------------
Task Number : 6150a2bd
'''

'''
Function call to utilitites package for reading the input json file.
'''
with open(sys.argv[1], 'r') as j_file:
    train_in,train_out,test_in,test_out = cu.json_src_reader(j_file) 

def solve(inputs):
    '''
    Used flip function from numpy package to reverse the order of elements in an array, 
    the elements are reordered but the shape is preserved.
    
    Parameters:
    -----------
    c = ARC test list
    
    Returns:
    --------
    test_result = returns a numpy ndarray, size of which depends on the task.
    '''

    inp_array = np.array(inputs)
    test_result = np.flip(inp_array)
    return test_result

for inputs in (train_in + test_in):
    outputs = solve(inputs)
    print(outputs)
    print()

'Additional line to visualize the test input and test output'
#plt_list = [inputs, outputs]
#cu.visualize(plt_list)