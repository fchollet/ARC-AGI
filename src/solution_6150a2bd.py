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
    train_in,train_out,test_in,test_out = cu.json_arc_reader(j_file) 

def solve(c):
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
    cnp = np.array(c[0])
    test_result = np.flip(cnp)
    return test_result

output = solve(test_in)
print(output)
#plt_list = [c[0], output]
#cu.visualize(plt_list)
