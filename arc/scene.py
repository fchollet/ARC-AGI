from typing import Any, TypeAlias

from arc.util import logger
from arc.board import Board
from arc.contexts import SceneContext
from arc.definitions import Constants as cst
from arc.object import ObjectDelta, find_closest

log = logger.fancy_logger("Scene", level=30)

SceneData: TypeAlias = dict[str, Any]


class Scene:
    """One pair of Boards defining a 'case' or a 'test' for the Task.

    Attributes:
      idx: Numerical index of the scene as presented in the data.
      input: the input Board.
      output: the output Board.
      context: A learned set of variables that might influence operations.
    """

    def __init__(self, data: SceneData, idx: int = 0):
        self.idx = idx
        self.input = Board(data["input"], name=f"Input {idx}")
        self.output = Board(data["output"], name=f"Output {idx}")

        # Context is built between input/output, and might influence a redo
        self.context = SceneContext()

        # Initially, we start at shallow representations and proceed outward
        self._dist = -1

    @property
    def props(self) -> int:
        """Sum of total properties used to define the input and output boards."""
        return self.input.rep.props + self.output.rep.props

    @property
    def ppp(self) -> float:
        """Properties per Point: a measure of representation compactness."""
        return self.props / (self.input.rep.size + self.output.rep.size)

    @property
    def dist(self) -> float:
        """Transformational distance measured between input and output"""
        return self._dist

    def reduce(self, batch: int = cst.BATCH, max_iter: int = cst.MAX_ITER) -> None:
        """Determine a compact representation of the input and output Boards."""
        self.input.reduce(batch=batch, max_iter=max_iter)
        log.info(f"Input reduction at {self.input.rep.props}")
        if self.output:
            self.output.reduce(batch=batch, max_iter=max_iter, source=self.input)
            log.info(f"Output reduction at {self.output.rep.props}")

    # TODO Below needs review/updating
    def match(self):
        """Identify the minimal transformation set needed from input -> output Board."""
        self._dist, self.path = self.recreate(self.output.rep, self.input.inv())
        log.info(f"Minimal distance transformation ({self.dist}):")
        for delta in self.path:
            obj1, obj2, trans = delta.right, delta.left, delta.transform
            log.info(f"Tr {trans} | {obj1._name} -> {obj2._name}")

    def recreate(self, obj, inventory) -> tuple[int, list[ObjectDelta]]:
        """Recursively tries to most easily create the given object"""
        delta = find_closest(obj, inventory)
        if delta is None:
            return (-1, [])
        result = (delta.dist, [delta])
        if not obj.children:
            return result

        all_dist = 0
        all_deltas = []
        for kid in obj.children:
            kid_dist, kid_deltas = self.recreate(kid, inventory)
            all_dist += kid_dist
            all_deltas.extend(kid_deltas)
        if delta.dist <= all_dist:
            return result
        else:
            return (all_dist, all_deltas)
