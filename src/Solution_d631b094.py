# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:33:23 2019

@author: Sampritha H M
"""

import json
import numpy as np
import os
print(os.getcwd())

def json_arc_reader(file_name):
    json_file = open(file_name)
    data = json.load(json_file)
    print(data)
    train_inputs = [data['train'][i]['input'] for i in range(len(data['train']))]
    train_outputs = [data['train'][i]['output'] for i in range(len(data['train']))]
    test_inputs = [data['test'][i]['input'] for i in range(len(data['test']))]
    test_outputs = [data['test'][i]['output'] for i in range(len(data['test']))]
    return train_inputs,train_outputs,test_inputs,test_outputs

os.chdir("..")
os.chdir("data/training")
train_input,train_output,test_input,test_output = json_arc_reader('d631b094.json')

def manipulation(inputData):
    array = inputData[0]
    print("\n Input: \n",array)
    output = [[]]
    for i in range(len(array)):
        arr = np.array(array[i])
        for j in range(len(arr)):
            if arr[j] != 0:
                output[0].append(arr[j].tolist())
    return output


print("\n\nTRANING")

result = []
result.append(manipulation(train_input))
print(" Output:\n",result)
trainOutput = []
trainOutput.append(train_output[0])
if(trainOutput == result):
    print("Training Successfull")
    

print("\n\nTESTING")

result = []
result.append(manipulation(test_input))
print(" Output:\n",result)
if(test_output == result):
    print(" Testing Successfull")