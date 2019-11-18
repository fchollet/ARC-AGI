#This file has the solution to the task 29c11459.json

from ioOps import read_file, get_file_path, print_grid
import numpy as np
import json
import sys

def solve(data):
    grey = 5
    for input in data:
        for row in input:
            if row[0] != 0:
                length = len(row)
                midPoint = length // 2
                startCode = row[0]
                endCode = row[length - 1]
                row[midPoint] = grey
                row[1: midPoint] = startCode
                row[midPoint + 1: ] = endCode
    return data

if __name__ == "__main__":
    inputFilePath = get_file_path(sys.argv)
    data = read_file(inputFilePath)
    grid = solve(data)
    print_grid(grid)