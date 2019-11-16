"""
Ajinkya Sakhare
Assignment 3
Solution #2 (4c4377d9.json)
"""
import json as js
import numpy as np

def solve(inputmatrix):
    y = np.array(inputmatrix)
    y_copy=y
    y=y[[2,1,0], :]
    return((np.concatenate((y, y_copy))).tolist())

with open("D:\\ARC1\\data\\training\\4c4377d9.json") as json_file:
    data = js.load(json_file)
print("training...")
for input in (data['train']):
    inputmatrix=(input['input'])
    out=solve(inputmatrix)
    print("derived output")
    print(out)
    outputmatrix = (input['output'])
    print("Expected output")
    print(outputmatrix)
print("testing...")
for input in (data['test']):
    inputmatrix = (input['input'])
    out = solve(inputmatrix)
    print("derived output")
    print(out)
    outputmatrix = (input['output'])
    print("Expected output")
    print(outputmatrix)