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