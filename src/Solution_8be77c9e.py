# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:51:45 2019

@author: sampritha
"""

import json
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
train_input,train_output,test_input,test_output = json_arc_reader('8be77c9e.json')

def manipulate(inputData):
    array = inputData[0]
    print("\n Input: \n",array)
    reverse = array[::-1]
    output = []
    output.append(array + reverse)
    return output

print("\n\nTRANING")

result = manipulate(train_input)
print(" Output:\n",result)
trainOutput = []
trainOutput.append(train_output[0])
if(trainOutput == result):
    print("Training Successfull")
    

print("\n\nTESTING")

result = manipulate(test_input)
print(" Output:\n",result)
if(test_output == result):
    print(" Testing Successfull")
