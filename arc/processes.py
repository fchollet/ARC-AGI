from typing import Any
import numpy as np
from arc.contexts import Context
from abc import ABC, abstractmethod

from arc.util import logger
from arc.board_methods import color_connect, grid_filter
from arc.definitions import Constants as cst
from arc.object import Object

log = logger.fancy_logger("Processes", level=30)


class Process(ABC):
    def test(self, obj: Object) -> bool:
        """Check whether we believe we should run this process."""
        return True

    @abstractmethod
    def run(self, obj: Object) -> Object:
        pass

    def info(self, obj: Object) -> None:
        log.debug(f"Running {self.__class__.__name__}")


class SeparateColor(Process):
    def __init__(self):
        pass

    def test(self, obj: Object) -> bool:
        return len(obj.c_rank) > 1

    def run(self, obj: Object) -> dict[str, Any]:
        """Improves representation by combining points of like colors"""
        self.info(obj)
        # TODO Add in Context handling
        color = obj.c_rank[0][0]

        match_pos, other_pts = grid_filter(obj.grid, color)
        oArgs1 = {"color": color, "pos": match_pos, "name": f"Color|{cst.cname[color]}"}
        oArgs2 = {"pts": other_pts, "name": "Remainder|"}

        name = f"Split|{cst.cname[color]}"
        parent = {"name": name, "decomposed": "Split", "children": [oArgs1, oArgs2]}
        return parent


class MakeBase(Process):
    def __init__(self):
        pass

    def test(self, obj: Object) -> bool:
        return True

    def run(self, obj: Object) -> dict[str, Any]:
        # Select a color to use, based on area covered by the color
        # TODO Add context interaction
        # color = getattr(context, "base_color", None)
        self.info(obj)
        if 0 in [item[0] for item in obj.c_rank]:
            color = 0
        else:
            color = obj.c_rank[0][0]
        gens = []
        rows, cols = obj.grid.shape
        if cols > 1:
            gens.append(f"C{cols - 1}")
        if rows > 1:
            gens.append(f"R{rows - 1}")
        if len(obj.c_rank) > 1:
            oArgs1 = {"color": color, "gens": gens, "decomposed": "Base"}
            oArgs1["name"] = f"Rect({rows},{cols})|{cst.cname[color]}"
            _, other_pts = grid_filter(obj.grid, color)
            oArgs2 = {"pts": other_pts, "name": "BaseRemainder"}
            name = f"MakeBase|{cst.cname[color]}"
            parent = dict(children=[oArgs1, oArgs2], name=name, decomposed="Base")
        else:
            name = f"Rect({rows},{cols})|{cst.cname[color]}"
            parent = dict(color=color, name=name, decomposed="Base", gens=gens)

        return parent


class ConnectObjects(Process):
    def __init__(self):
        pass

    def test(self, obj: Object) -> bool:
        return True

    def run(self, obj: Object) -> dict[str, Any] | None:
        self.info(obj)
        marked = obj.grid.copy()
        off_colors = [cst.NULL_COLOR]
        for color in off_colors:
            marked[marked == color] = cst.MARKED_COLOR
        obj_pts, fail = color_connect(marked)
        if fail:
            log.debug("Failed Connect")
            return None
        children = []
        for idx, pts in enumerate(obj_pts):
            name = f"Conn{idx}"
            children.append(dict(pts=pts, name=name))
        parent = {
            "children": children,
            "name": f"Cnxn{len(children)}",
            "decomposed": "Conn",
        }
        return parent


class Tiling(Process):
    def __init__(self):
        pass

    def run(self, obj: Object) -> dict[str, Any] | None:
        R, C, _ = obj.order
        # If there's no tiling order, try making a base layer
        if R == 1 and C == 1:
            return None
        # Check for a uniaxial tiling, indicated by a "1" for one of the axes
        elif R == 1:
            # NOTE this just needs to check R to switch the default axis, as above
            R = obj.grid.shape[0]
        elif C == 1:
            C = obj.grid.shape[1]

        # Track the points in the repeated block, also which colors cause noise
        tile_pts, noise = [], np.zeros(cst.N_COLORS)
        # For each i,j in the repeated block, figure out the most likely color
        for i in range(R):
            for j in range(C):
                # Count how many times each color shows up in the sub-mesh
                active_mesh = obj.grid[i::R, j::C]
                cts = np.zeros(cst.N_COLORS)
                for row in active_mesh:
                    cts += np.bincount(row, minlength=cst.N_COLORS)
                # Eliminate colors from consideration based on context
                # if task and hasattr(task.context, "noise_colors"):
                #     for noise_color in task.context.noise_colors:
                #         cts[noise_color] = 0
                color = np.argmax(cts)
                cts[color] = 0
                noise += cts
                tile_pts.append((i, j, color))
        r_ct = np.ceil(obj.shape[0] / R)
        c_ct = np.ceil(obj.shape[1] / C)
        bound = None
        if obj.shape[0] % R or obj.shape[1] % C:
            bound = obj.shape
        args = dict(
            gens=[f"R{r_ct - 1}", f"C{c_ct - 1}"],
            children=[dict(pts=tile_pts, name=f"TBlock({R},{C})")],
            bound=bound,
            name=f"Tiling({R},{C})",
            decomposed="Tile",
        )
        # if task:
        #     task.context.noise += noise
        return args
