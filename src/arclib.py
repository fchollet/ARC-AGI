"""NUI Galway CT5132/CT5148 Programming and Tools for AI (James McDermott)

Common functions for Assignment 3
By writing my name below and submitting this file, I/we declare that
all additions to the provided skeleton file are my/our own work, and that
I/we have not seen any work on this assignment by another student/group.
Student name(s): Keith Bines
Student ID(s): 19234297
"""
import json


class Task:
    """
    Class for the problem task
    """

    # Private Methods

    # Constructor must have a file
    def __init__(self, task_file):
        if task_file[-5] != '.json':
            self.__task_file = task_file + '.json'
        else:
            self.__task_file = task_file
        self.__task_name = self.__task_file[self.__task_file.rindex('/') + 1: -5]
        self.__json = self.__get_json()

    # get the json input
    def __get_json(self):
        with open(self.__task_file) as json_file:
            return json.load(json_file)
