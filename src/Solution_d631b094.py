# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 00:33:23 2019

@author: Sampritha H M
"""

import numpy as np
import os
import sys
from Utils import common_utilities as utils


os.chdir("../")
os.chdir("data/training")
file = sys.argv[1]
train_input,train_output,test_input,test_output = utils.json_reader(file)


def solve(inputData):
    array = inputData[0]
    print("\n Input: \n",array)
    output = [[]] 
    for i in range(len(array)):
        arr = np.array(array[i])
        for j in range(len(arr)):
            if arr[j] != 0:
                output[0].append(arr[j])
    return output


print("\n\nTRANING")

result = []
result.append(solve(train_input))
print(" Output:\n",result)
trainOutput = []
trainOutput.append(train_output[0])
if(trainOutput == result):
    print("Training Successfull")
    

print("\n\nTESTING")

output = []
output.append(solve(test_input))
print(" Output:\n",output)
if(test_output == output):
    print(" Testing Successfull")