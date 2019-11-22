"""NUI Galway CT5132/CT5148 Programming and Tools for AI (James McDermott)

Solution for Assignment 3: File c1d99e64.json

Student name(s): Ian Matthews
Student ID(s):   12100610

"""
import numpy as np
import json
import sys
from itertools import chain

from numpy.core._multiarray_umath import ndarray


def load_file(filename):
    """
    do I ned to be able to handle the three input types?
    :param filename:
    :return:
    >>> load_file('../data/training/d037b0a7.json')
    [[[0, 0, 6], [0, 4, 0], [3, 0, 0]], [[0, 2, 0], [7, 0, 8], [0, 0, 0]], [[4, 0, 0], [0, 2, 0], [0, 0, 0]], [[4, 0, 8], [0, 0, 0], [0, 7, 0]]]
    """
    X = json.load(open(filename))
    train = X["train"]
    test = X["test"]
    inputs = [i["input"] for i in chain(train, test)]
    return inputs


def print_grid(grid):
    """
    Print the output grids as a grid of integers with a space a separator.  There should be
    a space between grids so adding one here after the output
    :param grid:
    :return:
    """
    for row in grid:
        print(" ".join(map(str, row)))
    print(end="\n")


def solve(input_grid):
    """
    Given the input grid from any training or evaluation pair in the input json file,
    solve should return the correct output grid in the same format.
    Allowed formats are : 1. a JSON string containing a list of lists; or 2. a Python list of lists;
    or 3. a Numpy 2D array of type int
    :param input_grid:
    :return: output_grid
    >>> X = json.load(open("../data/training/c1d99e64.json"))
    >>> ia = np.asarray(X["test"][0]["input"])
    >>> solve(ia)
    array([[4, 0, 4, 0, 4, 4, 2, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 2,
            4, 0, 0],
           [4, 4, 4, 0, 0, 4, 2, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 2,
            4, 0, 0],
           [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2],
           [4, 0, 4, 4, 4, 0, 2, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 2,
            4, 4, 0],
           [4, 4, 0, 4, 4, 4, 2, 0, 0, 0, 4, 4, 4, 4, 0, 4, 4, 4, 0, 4, 4, 2,
            4, 4, 4],
           [4, 4, 4, 0, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 0, 4, 2,
            4, 0, 4],
           [4, 0, 0, 4, 0, 4, 2, 4, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 4, 0, 4, 2,
            4, 4, 4],
           [4, 4, 4, 4, 4, 0, 2, 4, 0, 4, 0, 0, 4, 4, 0, 0, 4, 4, 4, 0, 0, 2,
            0, 4, 0],
           [0, 4, 4, 0, 4, 4, 2, 4, 4, 0, 4, 4, 0, 4, 4, 0, 0, 4, 0, 4, 0, 2,
            4, 0, 4],
           [4, 4, 4, 0, 4, 4, 2, 0, 4, 4, 4, 4, 4, 0, 0, 4, 0, 4, 4, 4, 0, 2,
            4, 4, 4],
           [4, 0, 4, 4, 4, 0, 2, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 2,
            0, 0, 4],
           [4, 4, 0, 4, 0, 0, 2, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 2,
            4, 4, 4],
           [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2],
           [0, 4, 4, 0, 0, 0, 2, 0, 4, 4, 4, 4, 0, 4, 4, 0, 0, 4, 4, 4, 4, 2,
            0, 4, 4],
           [4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 0, 4, 0, 4, 4, 0, 4, 4, 4, 4, 2,
            4, 4, 4],
           [4, 4, 4, 4, 4, 0, 2, 4, 0, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 0, 2,
            4, 0, 4],
           [0, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 0, 4, 4, 4, 2,
            4, 4, 0],
           [0, 4, 4, 4, 4, 0, 2, 4, 4, 4, 0, 4, 0, 4, 0, 4, 4, 4, 4, 4, 4, 2,
            0, 4, 4],
           [4, 4, 4, 0, 4, 4, 2, 0, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2,
            0, 0, 0],
           [4, 4, 0, 4, 4, 4, 2, 4, 4, 0, 4, 4, 4, 0, 4, 4, 4, 0, 4, 4, 0, 2,
            0, 4, 4],
           [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2],
           [4, 4, 4, 4, 0, 4, 2, 4, 0, 4, 4, 4, 0, 0, 0, 0, 4, 0, 4, 4, 4, 2,
            4, 4, 4],
           [0, 4, 4, 4, 4, 4, 2, 4, 0, 4, 0, 4, 4, 0, 4, 4, 0, 4, 4, 0, 4, 2,
            4, 4, 4],
           [4, 4, 4, 4, 4, 4, 2, 4, 4, 0, 0, 0, 0, 4, 4, 4, 0, 0, 4, 4, 4, 2,
            4, 4, 0],
           [4, 0, 4, 0, 4, 4, 2, 4, 0, 0, 0, 4, 4, 4, 4, 4, 0, 4, 0, 4, 4, 2,
            0, 4, 0],
           [4, 4, 0, 4, 0, 4, 2, 0, 4, 0, 4, 4, 0, 4, 4, 0, 0, 0, 4, 0, 4, 2,
            4, 4, 4],
           [4, 0, 0, 4, 4, 4, 2, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 0, 2,
            4, 4, 4]])
    """
    grid: ndarray = np.asarray(input_grid)
    # find the rows and which are all dark
    rows = []
    for i in range(grid.shape[0]):
        if np.all(grid[i, :] == 0):
            rows.append(i)
    # colour in the cols
    for j in range(grid.shape[1]):
        if np.all(grid[:, j] == 0):
            grid[:, j] = 2

    # colour in rows
    for row in range(len(rows)):
        grid[rows[row], :] = 2

    return grid


def main():
    try:
        inputs = load_file(sys.argv[1])
        for grid in inputs:
            output = solve(grid)
            print_grid(output)
    except IndexError:
        print("please enter input file")


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
