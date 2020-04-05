"""
This will be used to run the kaggle competition version of this solver.
Everything should be modular enough that it should be easy to hot swap this starter and visualizer with the web based
version of this app.
"""
from src.models.problem_fetcher import ProblemFetcher
from src.models.SingleProblemSolver import SingleProblemSolver
import pandas as pd

def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


def main():
    kaggle_path = '../input/abstraction-and-reasoning-challenge/'
    problem_fetcher = ProblemFetcher('../data')
    training, test = problem_fetcher.get_all_data_file_names()
    test_predictions = []
    # problem_f_name = '0b148d64.json'
    # started with problem 13 Because it looked like a good place to start and lucky numbers...
    # problem = ProblemFetcher().get_specific_training_problem(problem_f_name)
    p_num = 0
    for problem_f_name in test:
        print(p_num)
        p_num += 1
        problem = problem_fetcher.get_specific_test_problem(problem_f_name)
        solution = SingleProblemSolver(problem).solve()
        test_predictions.append((str(problem_f_name), solution))

    # test_predictions = [[list(pred) for pred in test_pred] for test_pred in test_predictions]
    submission = pd.read_csv(problem_fetcher.submission_path / 'sample_submission.csv', index_col='output_id')

    for pred in test_predictions:
        name = pred[0].replace('.json', '_0')
        flattened = flattener(pred[1].tolist())
        submission.loc[name, 'output'] = flattened

    print(submission.head())
    submission.to_csv(problem_fetcher.submission_path / 'submission.csv')
    print('done')


if __name__ == "__main__":
    # execute only if run as a script
    main()