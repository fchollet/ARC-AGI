import itertools as it
import json
from typing import Any, List, Tuple

import numpy as np


def dfs(x, y, input, visited, island) -> bool:
    r, c = input.shape
    if x < 0 or x >= r or y < 0 or y >= c:
        return False
    elif visited[x, y] or input[x, y] != 0:
        return True
    else:
        visited[x, y] = True
        island.append((x, y))

    is_island = True
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        is_isle = dfs(x + dx, y + dy, input, visited, island)
        is_island = is_island and is_isle

    return is_island


def find_islands(input: np.ndarray) -> List[Any]:
    islands = []
    visited = np.zeros_like(input, dtype=bool)
    r, c = input.shape
    for i in range(r):
        for j in range(c):
            if input[i][j] == 0 and visited[i, j] == False:
                island = []
                is_island = dfs(i, j, input, visited, island)
                if is_island:
                    islands.append(island)
    return islands


def get_element_with_highest_frequency(input: np.ndarray, is_unique=False):
    unique, counts = np.unique(input, return_counts=True)
    if is_unique:
        max_count = np.max(counts)
        max_count_elements = np.sum(counts == max_count)
        return unique[np.argmax(counts)] if max_count_elements == 1 else None

    return unique[np.argmax(counts)]


def convert_to_json(data: List[Tuple[np.ndarray, np.ndarray]]) -> List[dict]:
    train = list(map(lambda x: {"input": x[0].tolist(), "output": x[1].tolist()}, data[:-1]))
    test = {"input": data[-1][0].tolist(), "output": data[-1][1].tolist()}
    return json.dumps({"train": train, "test": [test]})
