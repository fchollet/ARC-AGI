"""
This will be used to run the kaggle competition version of this solver.
Everything should be modular enough that it should be easy to hot swap this starter and visualizer with the web based
version of this app.
"""
from src.models.problem_fetcher import ProblemFetcher
from src.models.SingleProblemSolver import SingleProblemSolver
import numpy as np


def main():
    path = ''
    problem_fetcher = ProblemFetcher('../../data')
    training, test = problem_fetcher.get_all_data_file_names()
    train_predictions = []
    # problem_f_name = '0b148d64.json'
    # started with problem 13 Because it looked like a good place to start and lucky numbers...
    # problem = problem_fetcher.get_specific_training_problem(problem_f_name)
    p_num = 0
    correct = 0
    for problem_f_name in training:
        p_num += 1
        problem = problem_fetcher.get_specific_training_problem(problem_f_name)
        solution = SingleProblemSolver(problem).solve()
        train_predictions.append((str(problem_f_name), solution))
        if np.array_equal(np.array(solution), np.array(problem['test'][0]['output'])):
            print('got correct solution for: ' + problem_f_name)
            correct += 1

    print('{0}/{1} correct'.format(correct, p_num))


if __name__ == "__main__":
    # execute only if run as a script
    main()