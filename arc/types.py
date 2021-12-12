"""Define custom types used throughout the codebase."""
from typing import TypeAlias, TypedDict


BoardData: TypeAlias = list[list[int]]


class SceneData(TypedDict):
    input: BoardData
    output: BoardData


class TaskData(TypedDict):
    train: list[SceneData]
    test: list[SceneData]
