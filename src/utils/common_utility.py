'''
Code for common utilies used by all 3 tasks.
'''
import json
import matplotlib.pyplot as plt


def json_src_reader(json_file):
    '''
    Function to read and return the json file and split the data into train input, train output, test input, test output. 
    
    Parameters: 
    -----------
        data = reads the input json file
   
    Returns
    -------
        train_inputs,train_outputs,test_inputs,test_outputs = lists of test and train set.
    '''    
    data = json.load(json_file)
    
    train_inputs = [data['train'][i]['input'] for i in range(len(data['train']))]
    train_outputs = [data['train'][i]['output'] for i in range(len(data['train']))]
    test_inputs = [data['test'][i]['input'] for i in range(len(data['test']))]
    test_outputs = [data['test'][i]['output'] for i in range(len(data['test']))]
    return train_inputs,train_outputs,test_inputs,test_outputs


def visualize(input):
    '''
    Function to plot grids and emulate testing interface.
    
    Parameters: 
    -----------
    input = A list of test input and computed output.
    '''
    for i in range(len(input)):
        plt.matshow(input[i])
    plt.show()
