from typing import Any
import asciitree
import collections

from matplotlib.figure import Figure
from arc.comparisons import get_color_diff, get_order_diff, get_translation
from arc.layouts import tree_layout

from arc.util import logger
from arc.object import Object, ObjectDelta
from arc.processes import Process, MakeBase, ConnectObjects, SeparateColor
from arc.types import BoardData
from arc.viz import plot_layout

log = logger.fancy_logger("Board", level=20)


class Board:
    """One 2D set of colored squares--the base unit of the ARC dataset.

    We use the recursive Object class to build a hierarchical representation
    of the Board that minimizes total properties used.

    Attributes:
        rep: The current representation of the Board via Objects.
        name: An auto-generated, descriptive name for the board.
        proc_q: A priority queue for holding decomposition candidates.
        bank: Any decompositions with no further possible operations.

    """

    def __init__(
        self, data: BoardData, name: str = "", processes: list[Process] = None
    ):
        self.name = name
        self.rep = Object(grid=data)
        self.processes = processes or [MakeBase(), ConnectObjects(), SeparateColor()]

        # Used during decomposition process
        self.proc_q = collections.deque([self.rep])
        self.bank = []

        # TODO remove?
        self.inventory = []

    def plot_tree(self, **kwargs) -> Figure:
        return plot_layout(tree_layout(self.rep), show_axis=False, **kwargs)

    def tree(self, obj: Object, ind: int = 0, level: int = 10) -> None:
        """Log the Board as an hierarchy of named Objects."""
        # Quit early if we can't print the output
        if log.level > level:
            return
        obj = obj or self.rep
        if obj.decomposed:
            header = f"[{obj.decomposed}]({obj.props})"
        else:
            header = f"{obj._id}({obj.props})"
        nodes = {header: self._walk_tree(obj)}
        output = asciitree.LeftAligned()(nodes).split("\n")
        indent_str = "  " * ind
        for line in output:
            log.info(f"{indent_str}{line}")

    def _walk_tree(self, base: Object) -> dict[str, Any]:
        nodes = {}
        for kid in base.children:
            if kid.category == "Dot":
                return {}
            if kid.decomposed:
                header = f"[{kid.decomposed}]({kid.props})"
            else:
                header = f"{kid._id}({kid.props})"
            nodes[header] = self._walk_tree(kid)
        return nodes

    # TODO Redo
    def inv(self, max_dots=10):
        return self.rep.inventory(max_dots=max_dots)

    def decompose(self, batch: int = 10, max_iter: int = 10, source=None) -> None:
        """Determine the optimal representation of the Board.

        Args:
            batch: Number of candidates to keep each round. E.g. if batch=1, only the best
              candidate is retained.
            max_iter: Maximum number of iterations of decomposition.
        """
        self.inventory = source.inv() if source else []
        ct = 0
        while ct < max_iter:
            ct += 1
            self.batch_decomposition(batch=batch)
            log.debug(f"== Decomposition at {self.rep.props}p after {ct} rounds")
            if not self.proc_q:
                log.debug("===Ending decomposition due to empty processing queue")
                break
        final = sorted(self.bank + list(self.proc_q), key=lambda x: x.props)[0]
        self.rep = final.flatten()[0]
        self.rep.ppt("info")

    def batch_decomposition(self, batch: int = 10) -> None:
        """Decompose the top 'batch' candidates."""
        ct = 0
        while self.proc_q and ct < batch:
            obj = self.proc_q.popleft()
            self.tree(obj)
            added = self._decomposition(obj)
            if not added:
                self.bank.append(obj)
                log.debug("  # All leaves decomposed")
            self.proc_q.extend(added)
            log.debug(f" - Finished decomposition")
            ct += 1

    def create_decomposition(
        self, old_o: Object, new_args: dict[str, Any], **kwargs
    ) -> Object:
        # First generate any children of the main object
        children = []
        for kid_args in new_args.pop("children", []):
            children.append(Object(**kid_args))
        new_args["children"] = children

        if "color" in new_args:
            parent = Object(old_o.row, old_o.col, parent=old_o.parent, **new_args)
        else:
            parent = Object(*old_o.seed, parent=old_o.parent, **new_args)
        parent.occ = old_o.occ.copy()
        return parent

    def _decomposition(self, obj: Object) -> list[Object]:
        """Attempts to find a more canonical or condensed way to represent the object"""
        # No children means nothing to simplify
        if len(obj.children) == 0:
            return []
        # Search for the first object that's not decomposed and apply decomposition
        elif obj.decomposed:
            all_rev = []
            curr_occ = obj.occ.copy()
            for r_idx, child in enumerate(obj.children[::-1]):
                idx = len(obj.children) - 1 - r_idx
                child.occ = curr_occ.copy()
                reviewed = self._decomposition(child)
                if not reviewed:
                    curr_occ |= set([(pt[0], pt[1]) for pt in child.pts])
                    continue
                for rev in reviewed:
                    children = [kid.spawn() for kid in obj.children]
                    children[idx] = rev
                    red = obj.spawn(children=children)
                    all_rev.append(red)
                break
            return all_rev

        # Begin decomposition process:  check for existing context representations
        if check := find_closest(obj, self.inventory, threshold=0.75):
            banked = check.right.spawn()
            # TODO We should figure out the right way to assign all this
            banked.adult()
            banked.row = obj.row
            banked.col = obj.col
            banked.parent = obj.parent
            banked.decomposed = "Scene"
            return [banked]

        candidates = self.generate_candidates(obj)
        reviewed = self.check_candidates(obj, candidates)
        out = [(res.name, res.props) for res in reviewed]
        log.debug(f" + {len(reviewed)} candidates for {obj._id}: {obj.history}")
        log.debug(f"  {out}")
        for res in reviewed:
            self.tree(res)
        return reviewed

    def generate_candidates(self, obj: Object) -> list[Object]:
        obj.decomposed = "Save"
        candidates = []
        for process in self.processes:
            if process.test(obj):
                candidates.append(process.run(obj))

        # TODO This seems to be bugged based on number of args to CTile or RTile
        # Test for any form of (r, c) tiling, but only once during lifetime
        # if "Tile" not in obj.history:
        #     candidates.append(Pr.tiling(obj, **kwargs))

        results = [self.create_decomposition(obj, cand) for cand in candidates if cand]
        results.append(obj)
        return results

    def check_candidates(self, obj: Object, candidates: list[Object]) -> list[Object]:
        reviewed = []
        for cand in candidates:
            # Either check equality of absolute points, or subset
            if not set(obj.pts).issubset(cand.pts):
                log.debug(f"Repairing: {cand}")
                missing_pts = list(set(obj.pts) - set(cand.pts))
                patch = Object(pts=missing_pts, name="Patch")
                cand = Object(children=[cand, patch], name=cand.name + "Rep")
            remainder = set([(pt[0], pt[1]) for pt in set(cand.pts) - set(obj.pts)])
            if not remainder.issubset(obj.occ):
                log.debug(f"Failed Occl: {cand}")
                log.debug(f"Remainder {remainder}")
                log.debug(f"Occlusion {obj.occ}")
            else:
                reviewed.append(cand)

        return sorted(reviewed, key=lambda x: x.props)


default_comparisons = [get_order_diff, get_color_diff, get_translation]


def find_closest(
    obj: Object, inventory: list[Object], threshold: float = None
) -> ObjectDelta | None:
    if not inventory:
        return None
    match = ObjectDelta(obj, inventory[0], comparisons=default_comparisons)
    for source in inventory[1:]:
        delta = ObjectDelta(obj, source, comparisons=default_comparisons)
        if delta.dist < match.dist:
            match = delta
    if threshold is not None and match.dist > threshold:
        log.debug(f"{obj} No matches meeting threshold (best {match.dist})")
        return None
    log.info(f"{obj} Match! Distance: {match.dist}")
    return match
