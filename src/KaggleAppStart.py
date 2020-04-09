"""
This will be used to run the kaggle competition version of this solver.
Everything should be modular enough that it should be easy to hot swap this starter and visualizer with the web based
version of this app.
"""
from src.models.problem_fetcher import ProblemFetcher
from src.models.SingleProblemSolver import SingleProblemSolver
from src.models.ZoltansColorAndCounting.ColorAndCountingModuloQ import Recolor, Create
from src.models.XGBoostBullshit import do_bullshit
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import math


def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


def plot_one(ax, title, input_matrix):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title)


def plot_problem_with_gridlines(f_name, task, attempted_solutions):
    """
    Plots the first train, test pairs of a specified task, and the attempted solutions to that task
    using same color scheme as the ARC app

    :param f_name: The file name of the problem
    :param task: The task as it is defined in json in the file defined by f_name
    :param attempted_solutions: A list of attempted solutions by various algorithms.
    :return: Nothin just plots shite
    """
    print(f_name)
    num_train = len(task['train'])
    num_test = len(task['test'])
    num_solutions = len(attempted_solutions) * len(attempted_solutions[0][1])
    fig, axs = plt.subplots(2, num_train + num_test + num_solutions, figsize=(4 * num_train, 3 * 2))
    for i in range(num_train):
        plot_one(axs[0, i], 'train input', task['train'][i]['input'])
        plot_one(axs[1, i], 'train output', task['train'][i]['output'])
    # plt.tight_layout()
    # plt.show()

    # fig, axs = plt.subplots(2, num_test, figsize=(3 * num_test, 3 * 2))
    if num_test == 1:
        plot_one(axs[0, num_train], 'test input', task['test'][0]['input'])
        if 'output' in task['test'][0]:
            plot_one(axs[1, num_train], 'test output', task['test'][0]['output'])
    else:
        for i in range(num_test):
            plot_one(axs[0, num_train + i], 'test input', task['test'][i]['input'])
            if 'output' in task['test'][0]:
                plot_one(axs[1, num_train + i], 'test output', task['test'][i]['output'])

    # fig, axs = plt.subplots(2, num_solutions, figsize=(3 * num_test, 3 * 2))
    if num_solutions == 1:
        plot_one(axs[0, num_train + num_test],
                 attempted_solutions[0][0], attempted_solutions[0][1][0])
    else:
        for i in range(len(attempted_solutions)):
            for j in range(len(attempted_solutions[i][1])):
                plot_one(axs[i % 2, math.floor(num_train + num_test + i + j)],
                         attempted_solutions[i][0], attempted_solutions[i][1][j])
    plt.tight_layout()
    plt.show()


def percentage_correct(expected, calculated):
    expected = np.array(expected)
    calculated = np.array(calculated)
    if expected.shape != calculated.shape:
        return 0.0
    num_correct = 0
    total = 0
    for i in range(len(expected)):
        for j in range(len(expected[0])):
            if expected[i][j] == calculated[i][j]:
                num_correct += 1
            total += 1
    return num_correct / total * 100


def solve_problems(problem_fetcher, input_data, is_eval=False, is_test=False):
    # problem_f_name = '0b148d64.json'
    # started with problem 13 Because it looked like a good place to start and lucky numbers...
    # problem = ProblemFetcher(data_path).get_specific_training_problem(problem_f_name)
    if is_eval:
        print('DOING EVALUATION PROBLEMS')
    p_num = 0
    sp_correct = []
    zoltan_correct = []
    xbg_correct = []
    folder = problem_fetcher.training_folder
    test_predictions = []
    if is_eval:
        folder = problem_fetcher.evaluation_folder
    for problem_f_name in input_data:
        if p_num % 25 != 0:
            print(p_num)
        p_num += 1
        problem = problem_fetcher.get_problem(folder, problem_f_name)
        sp_solutions = SingleProblemSolver(problem).solve()
        if not is_test:
            sp_percent = percentage_correct(problem['test'][0]['output'], sp_solutions[0])
        zoltan_problem = Create(problem)
        zoltan_solutions = [Recolor(zoltan_problem)]
        xgb_solutions = do_bullshit(problem, problem_f_name)
        solutions = [('Delete by Color', sp_solutions)]
        if zoltan_solutions[0] != -1:
            if not is_test:
                zoltan_percent = percentage_correct(problem['test'][0]['output'], zoltan_solutions[0])
            solutions.append(('Zoltan', np.array(zoltan_solutions)))
        elif not is_test:
            zoltan_percent = 0.0
        if not isinstance(xgb_solutions, int):
            if not is_test:
                xbg_percent = percentage_correct(problem['test'][0]['output'], xgb_solutions)
            solutions.append(('XBG Ensemble', xgb_solutions))
        elif not is_test:
            xbg_percent = 0
        if is_test:
            test_predictions.append(solutions)
        elif sp_percent > 99 or zoltan_percent > 99 or xbg_percent > 99:
            if sp_percent == 100:
                sp_correct.append(problem_f_name)
            if zoltan_percent == 100:
                zoltan_correct.append(problem_f_name)
            if xbg_percent == 100:
                xbg_correct.append(problem_f_name)

            plot_problem_with_gridlines(problem_f_name, problem, solutions)

    if is_eval:
        print('DOING EVALUATION PROBLEMS NOT TRAINING')
    if not is_test:
        print('{0}/{1} Training Problems correct'.format(len(zoltan_correct) + len(sp_correct) + len(xbg_correct), p_num))
        print('{0}/{1} Training Problems correct with single problem solver'.format(len(sp_correct), p_num))
        print('{0}/{1} Training Problems correct with zoltan problem solver'.format(len(zoltan_correct), p_num))
        print('{0}/{1} Training Problems correct with xbg problem solver'.format(len(xbg_correct), p_num))
    else:
        create_submission(problem_fetcher, test_predictions)


def create_submission(problem_fetcher, test_predictions):
    submission = pd.read_csv(problem_fetcher.data_path / 'sample_submission.csv', index_col='output_id')

    names_seen = set()

    for pred in test_predictions:
        for solver in range(len(pred[1])):
            for solution_by_solver in range(len(pred[1][solver])):
                name = pred[0].replace('.json', '_{}'.format(solution_by_solver))
                output = pred[1][solver][solution_by_solver]
                if isinstance(output, np.ndarray):
                    output.tolist()
                flattened = flattener(output)
                if name in names_seen:
                    submission.loc[name, 'output'] = flattened + ' ' + submission.loc[name, 'output']
                else:
                    submission.loc[name, 'output'] = flattened
                    names_seen.add(name)

    print(submission.head())
    submission.to_csv('submission.csv')
    print('done')


def main():
    # data_path = '../input/abstraction-and-reasoning-challenge/' # kaggle path
    data_path = '../data'  # my local path
    problem_fetcher = ProblemFetcher(data_path)
    training, evaluation, test = problem_fetcher.get_all_data_file_names()
    solve_problems(problem_fetcher, training)
    solve_problems(problem_fetcher, evaluation, is_eval=True)
    solve_problems(problem_fetcher, test, is_test=True)


if __name__ == "__main__":
    # execute only if run as a script
    main()
