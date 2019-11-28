# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 23:06:57 2019

@author: Sampritha H M
"""
import json
import matplotlib.pyplot as plt

def json_reader(file_name):
    json_file = open(file_name)
    data = json.load(json_file)
    print(data)
    train_inputs = [data['train'][i]['input'] for i in range(len(data['train']))]
    train_outputs = [data['train'][i]['output'] for i in range(len(data['train']))]
    test_inputs = [data['test'][i]['input'] for i in range(len(data['test']))]
    test_outputs = [data['test'][i]['output'] for i in range(len(data['test']))]
    return train_inputs,train_outputs,test_inputs,test_outputs


def visualize(input):
    """
    Function to plot grids and emulate testing interface.
    input = A list of test input and computed output. 
    """
    for i in range(len(input)):
        plt.matshow(input[i])
        plt.show()