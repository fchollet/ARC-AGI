"""Utility functions"""

import json
import os
import pathlib

import numpy as np

THIS_SCRIPT = pathlib.Path(__file__).parent.resolve()
TRAIN_SUBSET_DIR = os.path.join(THIS_SCRIPT, "../../data/training")
TEST_SUBSET_DIR = os.path.join(THIS_SCRIPT, "../../data/evaluation")

TRAINING_FILES = {
    f: os.path.join(TRAIN_SUBSET_DIR, f) for f in os.listdir(TRAIN_SUBSET_DIR)
}
EVALUATION_FILES = {
    f: os.path.join(TEST_SUBSET_DIR, f) for f in os.listdir(TEST_SUBSET_DIR)
}
ALL_FILES = EVALUATION_FILES | TRAINING_FILES


def get_sample(file_path: str):
    """Parse a JSON file and return its content as a dictionary."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    return content


def get_train_samples(file_path: str):
    """Parse a JSON file and return its content as a dictionary."""
    content = get_sample(file_path)
    return content["train"]


def get_test_sample(file_path: str):
    """Parse a JSON file and return its content as a dictionary."""
    content = get_sample(file_path)
    return content["test"]


def grid_edit_distance(
    x: np.ndarray,
    y: np.ndarray,
    normalize: bool = False,
):
    """Calculate the grid edit distance between two arrays."""
    num_edits = 0
    print("TYPE", type(x), type(y))

    if x.shape != y.shape:
        num_edits = max(np.prod(x.shape), np.prod(y.shape))
        if normalize:
            num_edits = 1.0
    else:
        num_edits = np.sum(x != y)
        if normalize:
            num_edits /= np.prod(x.shape)

    return num_edits


def evaluate(solutions: dict[str, np.ndarray]):
    """Evaluate the solutions."""

    scores = {}
    total_score = 0
    total_edits = 0
    for solution_filename, solution_grid in solutions.items():
        if solution_filename not in list(ALL_FILES.keys()):
            raise ValueError(f"Invalid filename: {solution_filename}")

        # Load the ground truth
        gt = get_test_sample(ALL_FILES[solution_filename])[0]["output"]

        # Convert the grids to numpy arrays
        gt = np.array(gt)
        solution_grid = np.array(solution_grid)

        # Calculate the grid edit distance
        normalized_score = grid_edit_distance(
            gt,
            solution_grid,
            normalize=True,
        )
        edits = grid_edit_distance(
            gt,
            solution_grid,
            normalize=False,
        )

        # Save the scores
        scores[solution_filename] = {
            "normalized": normalized_score,
            "edits": edits,
        }
        total_score += normalized_score
        total_edits += edits

    return scores | {
        "avg_score": total_score / len(solutions),
        "avg_edits": total_edits / len(solutions),
    }


if __name__ == "__main__":
    scores = evaluate({"0a938d79.json": np.array([[1, 2, 3], [1, 2, 3]])})
    print(scores)
