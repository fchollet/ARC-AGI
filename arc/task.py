from typing import Any
from matplotlib.figure import Figure

import numpy as np

from arc.util import logger
from arc.concepts import Act
from arc.contexts import TaskContext
from arc.definitions import Constants as cst
from arc.scene import Scene
from arc.selector import group_inputs, create_selectors, base_describe, describe, select
from arc.object import Object
from arc.transforms import const_map, t2t_map
from arc.viz import plot_scenes

log = logger.fancy_logger("Task", level=20)

TaskData = dict[str, Any]


class Task:
    """One 'problem' within the ARC dataset: contains and operates on Scenes.

    Attributes:
        raw: The unprocessed data from the input JSON. Used for resetting state.
        idx: The rank of the Task when sorted alphabetically by filename.
        uid: The stem of the filename (e.g. 'path/to/{uid}.json')
        cases: All 'training' scenes for the task.
        tests: All 'test' scenes for the task.
        solution: The process to transform the input to the output.
        context: Additional information that might come from Task-level study.
        traits: Single-word descriptors of the Task. Used for analytics, grouping.
    """

    def __init__(self, task_data: TaskData, idx: int = 0, uid: str = ""):
        self.raw: TaskData = task_data
        self.idx: int = idx
        self.uid: str = uid
        self.cases: list[Scene] = []
        self.tests: list[Scene] = []

        # WIP
        self.context = TaskContext()
        self.solution = []
        self.traits: set[str] = set([])

        # Load scenes, cases ("train" data) and tests
        for scene_idx, scene_data in enumerate(task_data["train"]):
            self.cases.append(Scene(idx=scene_idx, data=scene_data))

        for scene_idx, scene_data in enumerate(task_data["test"]):
            self.tests.append(Scene(idx=scene_idx, data=scene_data))

    def __getitem__(self, arg: int | str):
        match arg:
            case int(idx):
                return self.cases[idx]
            case str(test_code):
                try:
                    return self.tests[int(test_code[1:])]
                except KeyError:
                    log.error(f"Unable to index a Task using '{test_code}'")
                    raise

    def info(self) -> None:
        """Display a set of key info about the task to the user."""
        log.info(f"Task {self.idx} UID = {self.uid} | First input board:")
        log.info(self.raw["train"][0]["input"], extra={"fmt": "bare"})

    def plot(self) -> Figure:
        return plot_scenes(
            [
                [(scene.input.rep.grid, scene.output.rep.grid) for scene in self.cases],
                [(scene.input.rep.grid, scene.output.rep.grid) for scene in self.tests],
            ],
            ["Case", "Test"],
        )

    @property
    def ppp(self) -> float:
        """Average properties-per-point across cases."""
        return np.mean([scene.ppp for scene in self.cases])

    @property
    def dist(self) -> float:
        """Average transformational distance across cases."""
        return np.mean([scene.dist for scene in self.cases])

    @property
    def n_boards(self) -> int:
        """Number of total boards in the Task."""
        return 2 * (len(self.cases) + len(self.tests))

    def complete_run(self) -> None:
        """Execute every step of the solution pipeline for the Task."""
        self.reduce()
        self.match()
        self.solve()
        self.test()

    def reduce(self, batch: int = 10, max_ct: int = 10) -> None:
        """Apply reduction across all cases, learning context and iterating."""
        # TODO apply context
        for scene in self.cases:
            log.info(f" ++ Reducing ({self.idx}, {scene.idx}) for {max_ct} rounds")
            scene.reduce(batch=batch, max_ct=max_ct)
            log.info(f"Scene PpP -> {scene.ppp:.3f}")
        log.info(f"Average PpP -> {self.ppp:.3f}")

    def match(self) -> None:
        """Match input and output objects for each case."""
        for scene in self.cases:
            log.info(f" ++ Matching ({self.idx}, {scene.idx})")
            scene.match()
            log.info(f"Scene distance -> {scene.dist}")
        log.info(f"Average Distance -> {self.dist:.1f}")

    # TODO Below needs review/updating
    def select(self):
        fails, variant, selectors = 1, 0, {}
        while fails and variant < 2:
            log.info(f" ++ Selecting {self.idx} with variant {variant}")
            self.groups, self.codes, self.links = group_inputs(self, variant)
            selectors, fails = create_selectors(self)
            variant += 1
        log.info(f"Average link distance -> {self.links}")
        return selectors

    def transform(self, groups, codes):
        t_maps = {}
        for g_idx, group in groups.items():
            if g_idx not in codes:
                log.warning(f"No transform codes for group {g_idx}")
                continue
            code = codes[g_idx]
            if code is None:
                t_maps[g_idx] = None
                continue
            for map_func in [const_map, t2t_map]:
                log.info(f"Trying map_func: {map_func.__name__}")
                curr_map = map_func(group, code)
                if curr_map:
                    t_maps[g_idx] = curr_map
                    break
        return t_maps

    def solve(self):
        selectors = self.select()
        transforms = self.transform(self.groups, self.codes)
        self.solution = [(sel, transforms.get(idx)) for idx, sel in selectors.items()]

    def generate(self, test_case: int = 0):
        soln_input = self.tests[test_case].input

        soln_input.reduce()
        output = self._generate_out(soln_input.rep, self.solution)
        return output

    def _generate_out(self, board, solution):
        out_children = []
        # Then, apply our selection criteria
        for selector, trans in solution:
            inputs = board.inventory()
            base_describe(inputs)
            describe(inputs)
            selected = select(inputs, selector)
            log.debug(f"Selected\n{selected}")
            if not trans:
                out_children.extend(selected)
                continue
            code, trait, tmap = trans
            log.info(f"Apply {code} based on {trait} via {tmap}")
            for obj in selected:
                # TODO Figure out a smoother way to do this?
                # Objects should start from an absolute, so we'll use anchor
                obj.adult()
                if trait is None:
                    out_obj = Act()[code](obj, tmap)
                else:
                    out_obj = Act()[code](obj, tmap[obj.traits[trait]])
                out_children.append(out_obj)

        result = Object(children=out_children, name="Solution")
        return result

    def test(self):
        success = 0
        log.info("Testing:")
        for idx, scene in enumerate(self.tests):
            if scene.output.rep == self.generate(idx):
                log.info("  Passed")
                success += 1
            else:
                log.info("  Failed")
        if success == len(self.tests):
            self.traits.add("passed")
