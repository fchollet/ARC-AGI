"""
This will be used to run the kaggle competition version of this solver.
Everything should be modular enough that it should be easy to hot swap this starter and visualizer with the web based
version of this app.
"""
from src.models.problem_fetcher import ProblemFetcher
from src.models.SingleProblemSolver import SingleProblemSolver


def main():
    # training, evaluation = ProblemFetcher().all_data_paths()
    # for problem in training:
    problem = ProblemFetcher().get_specific_training_problem('0b148d64.json')
    solution = SingleProblemSolver(problem).solve()
    print('done')


if __name__ == "__main__":
    # execute only if run as a script
    main()