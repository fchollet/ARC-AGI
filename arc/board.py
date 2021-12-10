from typing import TypeAlias
import asciitree
import collections

from matplotlib.figure import Figure

from arc.util import logger
from arc.object import Object, find_closest
from arc.processes import Processes as Pr
from arc.viz import plot_grid

log = logger.fancy_logger("Board", level=20)

BoardData: TypeAlias = list[list[int]]


class Board:
    """One 2D set of colored squares--the base unit of the ARC dataset.

    We use the recursive Object class to build a hierarchical representation
    of the Board that minimizes total properties used.

    Attributes:
        rep: The current representation of the Board via Objects.
        name: An auto-generated, descriptive name for the board.
        proc_q: A priority queue for holding reduction candidates.
        bank: Any reductions with no further possible operations.

    """

    def __init__(self, data: BoardData, name: str = ""):
        self.name = name
        self.rep = Object(grid=data)

        # Used during reduction process
        self.proc_q = collections.deque([self.rep])
        self.bank = []

        # TODO remove?
        self.inventory = []

        self._cplot = None

    @property
    def cplot(self) -> Figure:
        if not self._cplot:
            self._cplot = plot_grid(self.rep.grid)
        return self._cplot

    def tree(self, obj=None, ind=0, level=10) -> None:
        """Log the Board as an hierarchy of named Objects."""
        # Quit early if we can't print the output
        if log.level > level:
            return
        obj = obj or self.rep
        if obj.reduced:
            header = f"[{obj.reduced}]({obj.props})"
        else:
            header = f"{obj._id}({obj.props})"
        nodes = {header: self._walk_tree(obj)}
        output = asciitree.LeftAligned()(nodes).split("\n")
        ind = "  " * ind
        for line in output:
            log.info(f"{ind}{line}")

    def _walk_tree(self, base):
        nodes = {}
        for kid in base.children:
            if kid.is_dot():
                return {}
            if kid.reduced:
                header = f"[{kid.reduced}]({kid.props})"
            else:
                header = f"{kid._name}({kid.props})"
            nodes[header] = self._walk_tree(kid)
        return nodes

    def inv(self, max_dots=10):
        return self.rep.inventory(max_dots=max_dots)

    def reduce(self, batch=10, max_ct=10, source=None):
        """Determine the optimal representation of the Board."""
        self.inventory = source.inv() if source else []
        ct = 0
        while ct < max_ct:
            ct += 1
            self.batch_reduction(batch=batch)
            log.debug(f"== Reduction at {self.rep.props}p after {ct} rounds")
            if not self.proc_q:
                log.debug("===Ending reduction due to empty processing queue")
                break
        final = sorted(self.bank + list(self.proc_q), key=lambda x: x.props)[0]
        self.rep = final.flatten()[0]
        self.rep.ppt("info")

    def batch_reduction(self, batch=10):
        ct = 0
        while self.proc_q and ct < batch:
            obj = self.proc_q.popleft()
            self.tree(obj)
            added = self._reduction(obj)
            if not added:
                self.bank.append(obj)
                log.debug("  # All leaves reduced")
            self.proc_q.extend(added)
            log.debug(f" - Finished reduction for {obj}")
            ct += 1

    def create_reduction(self, old_o, new_args, **kwargs):
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

    def _reduction(self, obj):
        """Attempts to find a more canonical or condensed way to represent the object"""
        # No children means nothing to simplify
        if len(obj.children) == 0:
            return []
        # Search for the first object that's not reduced and apply reduction
        elif obj.reduced:
            all_rev = []
            curr_occ = obj.occ.copy()
            for r_idx, child in enumerate(obj.children[::-1]):
                idx = len(obj.children) - 1 - r_idx
                child.occ = curr_occ.copy()
                reviewed = self._reduction(child)
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

        # Begin reduction process:  check for existing context representations
        if check := find_closest(obj, self.inventory, threshold=0.75):
            banked = check.right.spawn()
            # TODO We should figure out the right way to assign all this
            banked.adult()
            banked.row = obj.row
            banked.col = obj.col
            banked.parent = obj.parent
            banked.reduced = "Scene"
            return [banked]

        candidates = self.generate_candidates(obj)
        reviewed = self.check_candidates(obj, candidates)
        out = [(res.name, res.props) for res in reviewed]
        log.debug(f" + {len(reviewed)} candidates for {obj._id}: {obj.history}")
        log.debug(f"  {out}")
        for res in reviewed:
            self.tree(res)
        return reviewed

    def generate_candidates(self, obj):
        obj.reduced = "Save"
        kwargs = {}
        candidates = []

        # Always try simplifying into a rectangle
        log.debug("Setting a base layer")
        candidates.append(Pr.make_base(obj, **kwargs))

        # Try separating the children into clustered groups
        log.debug("Connecting by color")
        candidates.append(Pr.connect_objs(obj, **kwargs))

        # TODO This seems to be bugged based on number of args to CTile or RTile
        # Test for any form of (r, c) tiling, but only once during lifetime
        # if "Tile" not in obj.history:
        #     candidates.append(Pr.tiling(obj, **kwargs))

        # Try separating by color
        if len(obj.c_rank) > 1:
            log.debug("Separating by color")
            candidates.append(Pr.sep_color(obj, **kwargs))
        results = [self.create_reduction(obj, cand) for cand in candidates if cand]
        results.append(obj)
        return results

    def check_candidates(self, obj, candidates):
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
