# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 13:00:30 2019

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
    output = []
    temp = []
    print("\n Input: \n",array)
    for i in range(len(array)):
        out = []
        for j in range(len(array[i])):
            a = array[i][j]
            out.append(a)
            out.append(a)
        temp.append(out)
        temp.append(out)
        output = temp 
    return output


print("\n\nTRANING")
for i in range(len(train_input)):
    inputData = train_input[i]
    result = solve(inputData)
    print(" Output:\n",result)
    if(train_output[i] == result):
        print("Training Successful for training input ", i)
    

print("\n\nTESTING")

output = solve(test_input[0])
print(" Output:\n",output)
if test_output[0] == output:
    print("Testing Successful")