# This file will have the solution for the task a699fb00
from ioOps import read_file, print_grid
import numpy as np
import json

def solve(data):
    for input in data:
        for row in input:
            for index in range(len(row) - 2):
                if row[index] == 1 and row[index + 2] == 1:
                    row[index + 1] = 2
    return np.asarray(data)

inputFilePath = "data/training/a699fb00.json"
data = read_file(inputFilePath)
grid = solve(data)
print_grid(grid)
