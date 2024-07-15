import argparse

import tasks as tk
from utils import convert_to_json


def gen_data(tasks, num_samples):
    for task in tasks:
        print(f"generating for task{task}")
        task_cls = getattr(tk, f"Task{task}")
        data = task_cls().run(num_samples)
        r = convert_to_json(data)
        with open(f"data/extended/training/task{task}.json", "w+") as f:
            f.write(r)
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--tasks", nargs="+", help="List of tasks to generate", required=True)
    parser.add_argument("-n", "--num_samples", type=int, default=10)
    # args = parser.parse_args()
    # data = gen_data(args.tasks, args.num_samples)
    task_ids = [1, 2, 5, 372, "x1", "x2", "x3", "x4", "x5"]
    data = gen_data(task_ids, 10)
