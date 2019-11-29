# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:33:23 2019

@author: Sampritha H M
"""

import numpy as np
import os
import sys
import Utils.common_utilities as utils

os.chdir("../")
os.chdir("data")
file = sys.argv[1]
train_input,train_output,test_input,test_output = utils.json_reader(file)


def solve(inputData):
    array = inputData
    print("\n Input: \n",array)
    output = [[]] 
    for i in range(len(array)):
        arr = np.array(array[i])
        for j in range(len(arr)):
            if arr[j] != 0:
                output[0].append(arr[j])
    return output


print("\n\nTRANING")
for i in range(len(train_input)):
    result = []
    inputData = train_input[i]
    result.append(solve(inputData))
    print(" Output:\n",result)
    trainOutput = []
    trainOutput.append(train_output[i])
    if(trainOutput == result):
        print("Training Successfull for training input ", i)
    

print("\n\nTESTING")

output = []
output.append(solve(test_input[0]))
print(" Output:\n",output)
if(test_output == output):
    print(" Testing Successfull")