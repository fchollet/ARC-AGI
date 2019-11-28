# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:51:45 2019

@author: Sampritha H M
"""

import os
import sys
import Utils.common_utilities as utils

os.chdir("..")
os.chdir("data/training")
file = sys.argv[1]
train_input,train_output,test_input,test_output = utils.json_reader(file)


def solve(inputData):
    array = inputData[0]
    print("\n Input: \n",array)
    reverse = array[::-1]
    output = []
    output.append(array + reverse)
    return output

print("\n\nTRANING")

result = solve(train_input)
print(" Output:\n",result)
trainOutput = []
trainOutput.append(train_output[0])
if(trainOutput == result):
    print("Training Successfull")
    

print("\n\nTESTING")

output = solve(test_input)
print(" Output:\n",output)
if(test_output == output):
    print(test_input[0])
    print(output[0])
    print(" Testing Successfull")

plotInput = [test_input[0], output[0]]
utils.visualize(plotInput)