"""
This is designed to get a given problem in the arc dataset.

To make it snappy it currently is only fetching from the data folder

"""
import os
import json
import random


class ProblemFetcher:

    def __init__(self, evaluation_fdr=os.path.join('..', '..', 'data', 'evaluation'),
                 training_fdr=os.path.join('..', '..', 'data', 'training')):
        self.evaluation_folder = evaluation_fdr
        self.training_folder = training_fdr

    def get_specific_training_problem(self, filename):
        """
        @param filename: the filename in the folder to get
        :return:
        The loaded json object from the file
        """
        return self._get_problem(self.training_folder, filename)

    def get_specific_evaluation_problem(self, filename):
        return self._get_problem(self.evaluation_folder, filename)

    def get_random_training(self):
        files = os.listdir(self.training_folder)
        r = random.SystemRandom()
        val = r.randint(0, len(files) - 1)
        return self._get_problem(self.training_folder, files[val])

    @staticmethod
    def _get_problem(folder, filename):
        path = os.path.join(folder, filename)
        with open(path, 'r') as f:
            return json.loads(f.read())
