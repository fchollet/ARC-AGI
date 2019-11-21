"""NUI Galway CT5132/CT5148 Programming and Tools for AI (James McDermott)

Solution for Assignment 3: File ed36ccf7.json

Student name(s): Ian Matthews
Student ID(s):   12100610

"""
import numpy as np
import json
import sys
from itertools import chain


def load_file(filename):
    """
    do I ned to be able to handle the three input types?
    :param filename:
    :return:
    >>> load_file('../data/training/ed36ccf7.json')
    [[[9, 0, 0], [9, 9, 9], [9, 9, 9]], [[6, 6, 6], [0, 0, 0], [6, 6, 0]], [[0, 0, 9], [0, 0, 9], [9, 9, 9]], [[2, 0, 2], [0, 0, 2], [0, 2, 2]], [[0, 0, 0], [5, 0, 0], [0, 5, 5]]]
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
    >>> ig =  [[0, 0, 0], [5, 0, 0], [0, 5, 5]]
    >>> solve(ig)
    array([[0, 0, 5],
           [0, 0, 5],
           [0, 5, 0]])
    """
    return np.rot90(input_grid)


def main():
    inputs = load_file(sys.argv[1])
    for grid in inputs:
        output = solve(grid)
        print_grid(output)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    main()
