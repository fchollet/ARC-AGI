from functools import cached_property

import numpy as np

from arc.util import dictutil, logger
from arc.board_methods import layer_pts, norm_pts, norm_children, translational_order
from arc.concepts import Gen
from arc.definitions import Constants as cst

log = logger.fancy_logger("Object", level=30)


class Object:
    def __init__(
        self,
        row: int = 0,
        col: int = 0,
        color: int = cst.NULL_COLOR,
        parent: "Object" = None,
        bound: tuple[int, int] = None,
        name: str = "",
        reduced: str = "",
        gens: list[Gen] = None,
        children: list["Object"] = None,
        grid=None,
        pos=None,
        pts=None,
    ):
        self.row = row
        self.col = col
        self.color = color
        self.parent = parent
        self.bound = bound
        self.gens = gens or []
        self.children = children or []
        self.name = name

        # Used during reduction process
        self.reduced = reduced
        self.occ = set([])

        # Used during selection process
        self.traits = {}

        # Internal variables for properties
        self._grid = None if grid is None else self._grid2dots(grid)
        self._pts = None if pts is None else self._pts2dots(pts)
        self._pos = None if pos is None else self._pts2dots(pos)
        self._props = None
        self._c_rank = None
        self._order = None

        # Special initializations based on inputs
        if children:
            self.adopt()

    # @property
    # def loc(self) -> tuple[int, int]:
    #     return (self.row, self.col)

    @property
    def seed(self) -> tuple[int, int, int]:
        """The *local* position and color information of the Object."""
        return (self.row, self.col, self.color)

    @property
    def anchor(self) -> tuple[int, int, int]:
        """The *global* position and color information of the Object."""
        if self.parent is None:
            return self.seed
        return (
            self.row + self.parent.anchor[0],
            self.col + self.parent.anchor[1],
            self.color if self.color != cst.NULL_COLOR else self.parent.color,
        )

    @property
    def head(self):
        pass
        # TODO eliminate?
        # if self.color == cst.NULL_COLOR:
        #     return self.loc
        # else:
        #     return self.seed

    @property
    def category(self) -> str:
        """A single-word description of the Object."""
        if self.is_dot():
            return "Dot"
        elif not self.children:
            if len(self.gens) == 1:
                return "Line"
            elif len(self.gens) == 2:
                return "Rect"
            else:
                return "Compound"
        else:
            if len(self.gens) > 0:
                return "Tile"
            elif all([kid.is_dot() for kid in self.children]):
                return "Cluster"
            else:
                return "Obj"

    @cached_property
    def shape(self) -> tuple[int, int]:
        """The bounding dimensions of the Object."""
        if self.is_dot():
            return (1, 1)
        maxrow = max([pt[0] for pt in self.pos])
        maxcol = max([pt[1] for pt in self.pos])
        return (maxrow + 1, maxcol + 1)

    @property
    def center(self):
        # TODO Unused, eliminate?
        row = self.seed[0] + (self.shape[0] - 1) / 2
        col = self.seed[1] + (self.shape[1] - 1) / 2
        return (row, col)

    @property
    def size(self):
        return len(self.pts)

    @property
    def _id(self):
        """A concise, (nearly) unique description of the Object."""
        if self.is_dot():
            shape = ""
        else:
            shape = f"({self.shape[0]}x{self.shape[1]})"
        link = "*" if self.reduced == "Scene" else ""
        return f"{link}{self.category}{shape}@{self.anchor}"

    def __repr__(self):
        """One line description of what the object is"""
        if self.is_dot():
            info = ""
        else:
            info = f"({len(self.children)}ch, {self.size}pts, {self.props}p)"
        header = f"{self._id}{info}"
        return header

    def info(self, tab=0, cond=10):
        """Detailed info on object and its children"""
        ind = "  " * tab
        output = [ind + self.__repr__()]
        for idx, child in enumerate(self.children):
            output.append(f"{child.info(tab=tab+1)}")
        if self.gens:
            output.append(f"{ind}  Gen{self.gens}")
        # Condense output to just cond lines
        if len(output) > cond and "..." not in output:
            output = output[:cond] + ["..."]
        return "\n".join(output)

    def ppt(self, level="info"):
        for line in self.info().split("\n"):
            getattr(log, level)(line)

    def __eq__(self, other: "Object") -> bool:
        return self.pts == other.pts

    def __lt__(self, other: "Object") -> bool:
        """Compare Objects based on their size (total points), shape, and location.

        This primarily is used for providing operational determinism via sorting.
        """
        if self.size != other.size:
            return self.size < other.size
        elif self.shape != other.shape:
            return self.shape < other.shape
        elif self.anchor != other.anchor:
            return self.anchor < other.anchor
        else:
            return False

    # def __sub__(self, other: "Object") -> "ObjectDelta":
    #     """Object differences are handled separately."""
    #     return ObjectDelta(self, other)

    def __getitem__(self, key: int) -> "Object":
        return self.children[key]

    def spawn(self, *args, **kwargs):
        if self.is_dot():
            # A dot will never have kwargs
            return Object(*(args + self.seed[len(args) :]))
        base_args = ["row", "col", "color", "parent", "bound", "name", "reduced"]
        new_args = {arg: getattr(self, arg) for arg in base_args}
        for key, val in zip(base_args, args):
            new_args[key] = val
        if self.gens and "gens" not in kwargs:
            new_args["gens"] = self.gens.copy()
        if self.children and "children" not in kwargs:
            new_args["children"] = [kid.spawn() for kid in self.children]
        new_args.update(kwargs)
        new_obj = Object(**new_args)

        # Tracking variables that need to be carried over
        new_obj.occ = self.occ.copy()
        return new_obj

    def reset(self):
        self._grid = None
        self._pts = None
        self._pos = None
        self._shape = None
        self._props = None
        self._c_rank = None
        self._order = None
        for child in self.children:
            child.reset()

    @property
    def history(self):
        result = [self.reduced] if self.reduced else []
        if self.parent:
            result = self.parent.history + result
        return result

    def adopt(self):
        """Ensures parental connection, normed seed, and attributes are reset"""
        minrow, mincol = norm_children(self.children)
        if minrow or mincol:
            log.debug(f"Kid norming: {self.name}")
            self.row += minrow
            self.col += mincol
        for child in self.children:
            child.parent = self
        self.reset()

    def flatten(self):
        """Returns an object optimized by reducing unnecessary levels"""
        # Having generators or not having children is a dead end
        if self.gens or not self.children or self.category == "Cluster":
            return [self]
        new_children = []
        for kid in self.children:
            new_children.extend(kid.flatten())
        if self.parent and self.color == cst.NULL_COLOR:
            log.debug(f"Flattening {self}")
            uplevel = []
            for kid in new_children:
                row, col = kid.row + self.row, kid.col + self.col
                uplevel.append(kid.spawn(row=row, col=col))
            return uplevel
        else:
            return [self.spawn(children=new_children)]

    def adult(self):
        """Removes parent and sets seed to anchor"""
        self.row, self.col, self.color = self.anchor
        self.parent = None

    def is_dot(self):
        return not self.children and not self.gens

    def issubset(self, other):
        return set(self.pts).issubset(other.pts)

    def sim(self, other):
        """Tests if objects are same up to translation"""
        if not self.shape == other.shape:
            return False
        return (self.grid == other.grid).all()

    def sil(self, other):
        """Tests if objects have the same 'outline', i.e. ignore color"""
        shape_match = self.shape == other.shape
        local_pos_match = self.pos == other.pos
        return shape_match and local_pos_match

    def overlap(self, other):
        ct = np.sum(self.grid == other.grid)
        return ct / self.grid.size, ct

    def _grid2dots(self, grid):
        """If we defined the object via a grid, this will supply the Dot children"""
        self._grid = np.array(grid, dtype=int)
        self.children = []
        M, N = self.grid.shape
        for i in range(M):
            for j in range(N):
                self.children.append(Object(i, j, self.grid[i, j], parent=self))

    def _pts2dots(self, pts):
        if len(pts) == 1:
            if len(pts[0]) == 3:
                self.row, self.col, self.color = pts[0]
            else:
                self.row, self.col = pts[0]
        else:
            seed, normed = norm_pts(pts)
            self.row, self.col = seed
            for pt in normed:
                self.children.append(Object(*pt, parent=self))

    @property
    def grid(self):
        """2D grid, inheriting color but not position"""
        if self._grid is not None:
            return self._grid
        if self.is_dot():
            self._grid = np.array([[self.pts[0][2]]], dtype=int)
            return self._grid
        self._grid = np.full(self.shape, cst.NULL_COLOR, dtype=int)
        if self.color != cst.NULL_COLOR:
            for pt in self.pos:
                self._grid[pt] = self.color
        else:
            for pt in self.pts:
                brow, bcol, _ = self.anchor
                self._grid[pt[0] - brow, pt[1] - bcol] = pt[2]
        return self._grid

    @property
    def pts(self):
        """All points in the absolute coordinates and color"""
        if self._pts is not None:
            return self._pts
        if self.is_dot():
            self._pts = [self.anchor]
            return self._pts

        # Generators are applied to either the implied object (Dot at self coords)
        # Or else we apply the gens to each child
        objs = self.children or [Object(*self.anchor)]
        # TODO gen refactor
        log.debug(self.gens)
        for gen in self.gens:
            spawned = []
            for obj in objs:
                spawned.extend(Gen(gen).create(obj))
            objs = spawned
        # This gets all pts from children and layers them in order
        # Only retrieve points within the defined bound
        if self.bound is not None:
            bound = (self.anchor[0] + self.bound[0], self.anchor[1] + self.bound[1])
            self._pts = layer_pts(objs, bound)
        else:
            self._pts = layer_pts(objs)
        return self._pts

    @property
    def pos(self):
        """Contains list of coordinates relative to the object anchor point"""
        if self._pos is not None:
            return self._pos
        if self.is_dot():
            self._pos = [(0, 0)]
            return self._pos
        a_row, a_col, _ = self.anchor
        self._pos = [(pt[0] - a_row, pt[1] - a_col) for pt in self.pts]
        return self._pos

    @property
    def props(self):
        if self._props is not None:
            return self._props
        # When we have a contextual reference, we already "know" the object
        if self.reduced == "Scene":
            self._props = 1
            return self._props

        # Calculate local information used (self existence, positions, and color)
        from_pos = int(self.row != 0) + int(self.col != 0)
        own_props = from_pos + int(self.color != cst.NULL_COLOR)
        self._props = 1 + own_props

        if self.is_dot():
            return self._props

        # Add up total actions on gens?
        # TODO Gen refactor
        of_gens = sum([len(str(gen)) for gen in self.gens])
        of_children = sum([item.props for item in self.children])
        # Include a constant amount for being something non-Dot
        self._props += 1 + of_gens + of_children
        return self._props

    @property
    def c_rank(self) -> list[tuple[int, int]]:
        """Get the counts for each color on the grid, starting with most prevalent"""
        if self._c_rank is not None:
            return self._c_rank
        # np.unique is an indeterminate type due to "return_counts"
        cts: list[tuple[int, int]] = list(zip(*np.unique(self.grid, return_counts=True)))  # type: ignore
        sorted_cts = sorted(cts, key=lambda x: x[1], reverse=True)
        self._c_rank = [
            (color, ct) for color, ct in sorted_cts if color != cst.NULL_COLOR
        ]
        return self._c_rank

    @property
    def order(self):
        if self._order is not None:
            return self._order
        if self.is_dot():
            self._order = (1, 1, 1)
            return self._order
        # Get the most-ordered stride for each axis
        row_o = translational_order(self.grid, row_axis=True)[0]
        col_o = translational_order(self.grid, row_axis=False)[0]
        # TODO The product of individual dimension order fraction is almost certainly wrong...
        # Also, the "default" order should be the full size of the dimension, not 1
        self._order = (row_o[0], col_o[0], row_o[1] * col_o[1])
        return self._order

    def inventory(self, leaf_only=False, depth=0, max_dots=10):
        if self.is_dot():
            return [self]
        elif self.category == "Cluster":
            # If we have a cluster, only accept child dots if there aren't many
            if len(self.children) <= max_dots:
                return [self] + self.children
            else:
                return [self]
        res = [self]
        if leaf_only and self.children:
            res = []
        for kid in self.children:
            add = kid.inventory(leaf_only=leaf_only, depth=depth + 1, max_dots=max_dots)
            res.extend(add)
        return res


class ObjectDelta:
    def __init__(self, obj1, obj2):
        self.dist = 0
        self.transform = {}
        self.left = obj1
        self.right = obj2
        if obj1 == obj2:
            return
        self.get_translation(obj1, obj2)
        self.get_color_diff(obj1, obj2)
        self.get_order_diff(obj1, obj2)
        if not self.transform:
            self.dist = 1000

    @property
    def _name(self):
        header = f"Delta({self.dist}): "
        trans = ""
        for item in self.transform:
            trans += f"{item}"
        return header + f"[{trans}]"

    def __repr__(self):
        return f"{self._name}: {self.right._id} -> {self.left._id}"

    def __lt__(self, other):
        return self.dist < other.dist

    def __sub__(self, other):
        """Returns a distance between transforms, used for selection grouping"""
        # First is the distance between base objects (uses ObjectDelta)
        dist = (self.right - other.right).dist

        # Then, add in the difference in transforms
        d_xor = dictutil.dict_xor(self.transform, other.transform)
        dist += len(d_xor)
        return dist

    def get_order_diff(self, obj1, obj2):
        """Checks for differences in the arrangement of points"""
        if obj1.sil(obj2):
            return
        o1o = obj1.order
        o2o = obj2.order
        # Without a matching silhouette, only an ordered transformation works here
        # NOTE Including flooding and similar ops will change this
        if o1o[2] != 1 or o2o[2] != 1:
            self.dist = 1000
        else:
            # There could exist one or more generators to create the other object
            for axis, code in [(0, "R"), (1, "C")]:
                if obj1.shape[axis] != obj2.shape[axis]:
                    ct = obj1.shape[axis]
                    # ct = max(obj1.shape[axis] // obj2.shape[axis],
                    #          obj2.shape[axis] // obj1.shape[axis])
                    scaler = "f" if code == "R" else "p"
                    self.transform[scaler] = ct - 1
                    self.dist += 1

    def get_color_diff(self, obj1, obj2):
        c1 = set([item[0] for item in obj1.c_rank])
        c2 = set([item[0] for item in obj2.c_rank])
        if c1 != c2:
            # Color remapping is a basic transform
            if len(c1) == 1 and len(c2) == 1:
                self.transform["c"] = list(c1)[0]
                self.dist += 1
            # However, partial or multiple remapping is not
            else:
                self.dist = 1000

    def get_translation(self, obj1, obj2):
        r1, c1, _ = obj1.anchor
        r2, c2, _ = obj2.anchor
        # NOTE Consider using center vs corner for this measure
        if r1 == r2 and c1 == c2:
            return
        # Check for zeroing, which is special
        if r1 == 0 and c1 == 0:
            self.dist += 0.25
            self.transform["z"] = 0
            return
        if r2 != r1:
            # Justifying a single dimension is also special
            if r1 == 0:
                self.dist += 0.5
                self.transform["j"] = 0
            else:
                self.dist += 1
                self.transform["w"] = r1 - r2
        if c2 != c1:
            if c1 == 0:
                self.dist += 0.5
                self.transform["j"] = 1
            else:
                self.dist += 1
                self.transform["s"] = c1 - c2


def find_closest(
    obj: Object, inventory: list[Object], threshold: float = None
) -> ObjectDelta | None:
    if not inventory:
        return None
    match = ObjectDelta(obj, inventory[0])
    for source in inventory[1:]:
        delta = ObjectDelta(obj, source)
        if delta.dist < match.dist:
            match = delta
    if threshold is not None and match.dist > threshold:
        log.debug(f"{obj} No matches meeting threshold (best {match.dist})")
        return None
    log.info(f"{obj} Match! Distance: {match.dist}")
    return match
