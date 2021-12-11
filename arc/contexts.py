"""WIP"""
from typing import Any
import numpy as np

from arc.definitions import Constants as cst
from arc.util import logger
from arc.board_methods import intersect
from arc.object import Object

log = logger.fancy_logger("Context", level=30)


class Context:
    def __init__(self):
        pass


class TaskContext(Context):
    def __init__(self):
        # noise tracks which colors pollute a tiling
        self.noise = np.zeros(cst.N_COLORS)

    def learn(self):
        pass

    def statics(self, scenes: list[Any]):
        common_in_px = intersect([item.input.raw for item in scenes])
        common_out_px = intersect([item.output.raw for item in scenes])
        common_all = intersect([common_in_px, common_out_px])
        self.static = Object(grid=common_all)
        self.stc = np.sum(common_all != cst.MARKED_COLOR)


class SceneContext:
    def __init__(self):
        # TODO Consider supporting inventory through the context
        self.inventory = []
