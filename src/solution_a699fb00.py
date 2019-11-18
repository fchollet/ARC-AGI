# This file will have the solution for the task a699fb00
from ioOps import read_file, print_grid, get_file_path
import numpy as np
import json
import sys

def solve(data):
    for input in data:
        for row in input:
            for index in range(len(row) - 2):
                if row[index] == 1 and row[index + 2] == 1:
                    row[index + 1] = 2
    return np.asarray(data)

if __name__ == "__main__":
    inputFilePath = get_file_path(sys.argv)
    data = read_file(inputFilePath)
    grid = solve(data)
    print_grid(grid)
