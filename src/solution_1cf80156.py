import sys
import numpy as np
import utils.common_utility as cu 


'''
Function to solve ARC task
---------------------------
Task Number : 1cf80156
'''

'''
Function call to utilitites package for reading the input json file.
'''

with open(sys.argv[1], 'r') as j_file:
    train_in,train_out,test_in,test_out = cu.json_src_reader(j_file)


def solve(inputs):
    '''
    The function returns an numpy array by slicing the min and max value of rows and columns for the given input.
    
    Parameters:
    -----------
    c = ARC test list
    
    Returns:
    --------
    test_result = returns a numpy ndarray, size of which depends on the task.
    '''
    test_res = []
    for m in range(len(inputs)):
        inp_array = np.array(inputs[m])
        res = np.where(inp_array>0)
        col_min = min(res[1])
        col_max = max(res[1])
        row_min = min(res[0])
        row_max = max(res[0])
        test_result = []
        for i in range(row_min,row_max+1):
            for j in range(col_min,col_max+1):
                test_result.append(inp_array[i][j])
        test_result = np.reshape(test_result,(row_max-row_min+1,col_max-col_min+1))
        test_res.append(test_result.tolist())
    return test_res


output1 = solve(train_in)
print(output1)
print()
output2 = solve(test_in)
print(output2)
 
#plt_list = [c[0], output]
#cu.visualize(plt_list)
