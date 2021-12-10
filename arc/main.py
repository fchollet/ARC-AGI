import glob
import json
import logging
import pickle
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from arc.util import logger
from arc.definitions import Constants as cst
from arc.task import Task

log = logger.fancy_logger("ARC", level=20)


class ARC:
    """Load and operate on a collection of Tasks.

    Tasks are given an integer index based on the sorted input filenames.

    Attributes:
        N: The number of loaded tasks in the instance.
        selection: The Task indices to consider for operations.
        tasks: The mapping of Task index to Task objects.
    """

    def __init__(
        self,
        N: int = cst.N_TRAIN,
        idxs: set[int] = set(),
        folder: str = cst.FOLDER_TRAIN,
    ):
        if not idxs:
            idxs = set(range(N))
        self.N: int = len(idxs)
        self.selection: set[int] = idxs

        self.tasks: dict[int, Task] = {}
        self.load_tasks(idxs=idxs, folder=folder)

        # TODO find a way to incorporate using blacklist coherently
        self.blacklist: set[int] = set([])

    @staticmethod
    def load(pid: str | int) -> "ARC":
        """Create an ARC instance from a pickled checkpoint."""
        with open(f"{pid}.pkl", "rb") as fh:
            return pickle.load(fh)

    def dump(self, pid: str | int) -> None:
        """Pickle the current state of the ARC instance."""
        with open(f"{pid}.pkl", "wb") as fh:
            pickle.dump(self, fh)

    def __getitem__(self, arg: int | tuple[int, int] | tuple[int, int, str]) -> Any:
        """Convenience method so the user has easy access to ARC elements."""
        match arg:
            case int(task_idx):
                return self.tasks[task_idx]
            case (task_idx, scene_idx):
                return self.tasks[task_idx][scene_idx]
            case (task_idx, scene_idx, attribute):
                return getattr(self.tasks[task_idx][scene_idx], attribute).rep

    def load_tasks(self, idxs: set[int] = set(), folder: str = ".") -> None:
        """Load indicated task(s) from the ARC dataset.

        Supplying 'idxs' will load specific tasks, while 'N' will load the first 'N'.
        """
        curr_idx, boards, tests = 0, 0, 0
        for filename in sorted(glob.glob(f"{folder}/*.json")):
            if curr_idx in idxs:
                with open(filename, "r") as fh:
                    task = Task(json.load(fh), idx=curr_idx, uid=Path(filename).stem)
                    self.tasks[curr_idx] = task
                    boards += task.n_boards
                    tests += len(task.tests)
            curr_idx += 1
        log.info(
            f"Loaded {len(self.tasks)} Tasks, with {boards} boards and {tests} tests."
        )

    def set_log(self, arg: int | dict[str, int] = None) -> None:
        """Set the logging level for ARC, or any named logger.

        Supply {"logger_name": <level int>, ...} as a convenient way to alter log content
        for your use case.
        """
        match arg:
            case int(level):
                for logname in ["Task", "Scene", "Board", "Object"]:
                    logging.getLogger(logname).setLevel(level)
            case {**levels}:
                for name, loglevel in levels.items():
                    logging.getLogger(name).setLevel(loglevel)

    def select(self, selector: set[str] = None, selection: set[int] = None) -> None:
        """Choose which tasks will be active, by direct selection or by a set of traits."""
        if selector is None and selection is None:
            self.selection = set(range(self.N))
        if selection is not None:
            self.selection = set(sorted(selection))
        if selector is not None:
            self.selection = set(sorted(self._select(selector)))

    def _select(self, selector: set[str]) -> set[int]:
        selection = set([])
        for idx, task in self.tasks.items():
            if selector.issubset(task.traits):
                selection.add(idx)
        remove = selection & self.blacklist
        if remove:
            log.info(f"Removing {len(remove)} tasks based on blacklist")
            selection -= remove
        log.info(f"Selected {len(selection)} based on Selector: {selector}")
        return selection

    def solve_tasks(self, N: int = None) -> None:
        """TODO needs updating"""
        N = N or self.N
        for idx in self.selection:
            if idx >= N:
                break
            log.info(f"Solving Task {idx}")
            try:
                self.tasks[idx].complete_run()
            except Exception as exc:
                msg = f"{type(exc).__name__} {exc}"
                log.warning(f"Failed solve of task {idx} {msg}")
