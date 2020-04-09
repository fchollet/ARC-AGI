"""
This is designed to get a given problem in the arc dataset.

"""
import os
import json
import random
from pathlib import Path


class ProblemFetcher:

    def __init__(self, data_path):
        self.training_folder = None
        self.test_folder = None
        self.evaluation_folder = None
        self.data_path = Path(data_path)
        self.set_all_data_paths()

    def get_specific_training_problem(self, filename):
        """
        @param filename: the filename in the folder to get
        :return:
        The loaded json object from the file
        """
        return self.get_problem(self.training_folder, filename)

    def get_specific_test_problem(self, filename):
        return self.get_problem(self.test_folder, filename)

    def get_random_training(self):
        files = os.listdir(self.training_folder)
        r = random.SystemRandom()
        val = r.randint(0, len(files) - 1)
        return self.get_problem(self.training_folder, files[val])

    @staticmethod
    def get_problem(folder, filename):
        path = os.path.join(folder, filename)
        with open(path, 'r') as f:
            return json.loads(f.read())

    def set_all_data_paths(self):
        data_path = self.data_path
        self.training_folder = data_path / 'training'
        self.test_folder = data_path / 'test'
        self.evaluation_folder = data_path / 'evaluation'
        if not os.path.exists(self.evaluation_folder):
            self.evaluation_folder = None

    def get_all_data_file_names(self):
        training_tasks = sorted(os.listdir(self.training_folder))
        evaluation_tasks = []
        if self.evaluation_folder:
            evaluation_tasks = sorted(os.listdir(self.evaluation_folder))
        test_tasks = sorted(os.listdir(self.test_folder))
        return training_tasks, evaluation_tasks, test_tasks
