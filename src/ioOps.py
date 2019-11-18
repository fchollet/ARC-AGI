import numpy as np
import json
import itertools

def read_file(pathToFile):
    lists = []
    with open(pathToFile, "r+") as jsonFile:
        data = json.load(jsonFile)
        train = data["train"]
        test = data["test"]

        for i in itertools.chain(train, test):
            lists.append(np.asarray(i["input"]))
        return np.asarray(lists)

def print_grid(data):
    for input in data:
        counter = 0
        dimention = len(input)
        arr = np.asarray(input).flatten()
        for i in arr:
            print(i, end=" ")
            counter += 1
            if (counter % dimention == 0):
                print()
        print("\n")