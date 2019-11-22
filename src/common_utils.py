"""NUI Galway CT5132/CT5148 Programming and Tools for AI (James McDermott)

Common funtions for Assignment 3

Student name(s): Ian Matthews
Student ID(s):   12100610
"""

import numpy as np
import json
from itertools import chain


def load_file(filename):
    """
    Read in a json file from the data/training folder in the ARC project
    Parse the file and return a list of input grids in the order of training inputs
    testing inputs.
    :param filename:
    :return: a list of input grids
    >>> load_file('../data/training/d037b0a7.json')
    [[[0, 0, 6], [0, 4, 0], [3, 0, 0]], [[0, 2, 0], [7, 0, 8], [0, 0, 0]], [[4, 0, 0], [0, 2, 0], [0, 0, 0]], [[4, 0, 8], [0, 0, 0], [0, 7, 0]]]
    >>> load_file('invalid.json')
    please enter a valid json file
    """
    try:
        X = json.load(open(filename))
        train = X["train"]
        test = X["test"]
        inputs = [i["input"] for i in chain(train, test)]
        return inputs
    except FileNotFoundError:
        print('please enter a valid json file')


def print_grid(grid):
    """
    Print the output grids as a grid of integers with a space a separator.  There should be
    a space between grids so adding one here after the output
    :param grid: a 2 D numpy array
    >>> arr = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> print_grid(arr)
    1 2 3
    4 5 6
    7 8 9
    <BLANKLINE>
    """
    for row in grid:
        print(" ".join(map(str, row)))
    print(end="\n")


if __name__ == "__main__":
    import doctest
    doctest.testmod()
