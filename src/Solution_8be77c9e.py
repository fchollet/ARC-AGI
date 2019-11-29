# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:51:45 2019

@author: Sampritha H M
"""

import os
import sys
import Utils.common_utilities as utils

os.chdir("..")
os.chdir("data")
file = sys.argv[1]
train_input,train_output,test_input,test_output = utils.json_reader(file)


def solve(inputData):
    array = inputData
    print("\n Input: \n",array)
    reverse = array[::-1]
    output = []
    output.append(array + reverse)
    return output

print("\n\nTRANING")
for i in range(len(train_input)):
    inputData = train_input[i]
    result = solve(inputData)
    print(" Output:\n",result)
    train_Output = []
    train_Output.append(train_output[i])
    if(train_Output == result):
        print("Training Successful for training input ", i)
    

print("\n\nTESTING")

output = solve(test_input[0])
print(" Output:\n",output)
if(test_output == output):
    print(test_input[0])
    print(output[0])
    print(" Testing Successful")

#plotInput = [test_input[0], output[0]]
#utils.visualize(plotInput)