"""Define custom types used throughout the codebase."""
from typing import Callable, TypeAlias, TypedDict

from arc.object import Object

# For the input data, we have a 3-type hierarchy
BoardData: TypeAlias = list[list[int]]


class SceneData(TypedDict):
    input: BoardData
    output: BoardData


class TaskData(TypedDict):
    train: list[SceneData]
    test: list[SceneData]
