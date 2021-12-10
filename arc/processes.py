import numpy as np
from arc.contexts import Context

from arc.util import logger
from arc.board_methods import color_connect, grid_filter
from arc.definitions import Constants as cst

log = logger.fancy_logger("Processes", level=30)


class Processes:
    @staticmethod
    def sep_color(obj, scene=None, task=None):
        """Improves representation by combining tiles of like colors"""
        # Attempt with most prevalent color, unless scene has a suggestion
        if hasattr(scene, "sep_color"):
            # The contextual scene can set a priority
            color = scene.sep_color
        else:
            color = obj.c_rank[0][0]

        if len(obj.c_rank) == 2:
            color2 = obj.c_rank[1][0]
            obj1, obj2 = grid_filter(obj.grid, [color, color2])
            obj2["name"] = f"Color|{cst.cname[color2]}"
        else:
            obj1, obj2 = grid_filter(obj.grid, [color])
            obj2["name"] = f"Other"
        obj1["name"] = f"Color|{cst.cname[color]}"
        name = f"Split|{cst.cname[color]}"
        parent = {"name": name, "reduced": "Split", "children": [obj1, obj2]}
        return parent

    @staticmethod
    def make_base(obj, context: Context = None):
        # Select a color to use, based on area covered by the color
        # TODO Add context interaction
        # color = getattr(context, "base_color", None)
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
            obj1 = {"color": color, "gens": gens, "reduced": "Base"}
            obj1["name"] = f"Rect({rows},{cols})|{cst.cname[color]}"
            _, obj2 = grid_filter(obj.grid, [color])
            obj2["name"] = "BaseRemainder"
            name = f"MakeBase|{cst.cname[color]}"
            parent = dict(children=[obj1, obj2], name=name, reduced="Base")
        else:
            name = f"Rect({rows},{cols})|{cst.cname[color]}"
            parent = dict(color=color, name=name, reduced="Base", gens=gens)

        return parent

    @staticmethod
    def connect_objs(obj, scene=None, task=None):
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
            "reduced": "Conn",
        }
        return parent

    @staticmethod
    def tiling(obj, scene=None, task=None):
        R, C, _ = obj.order
        n_colors = cst.NULL_COLOR + 1
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
        tile_pts, noise = [], np.zeros(cst.NULL_COLOR + 1)
        # For each i,j in the repeated block, figure out the most likely color
        for i in range(R):
            for j in range(C):
                # Count how many times each color shows up in the sub-mesh
                active_mesh = obj.grid[i::R, j::C]
                cts = np.zeros(n_colors)
                for row in active_mesh:
                    cts += np.bincount(row, minlength=n_colors)
                # Eliminate colors from consideration based on context
                if task and hasattr(task.context, "noise_colors"):
                    for noise_color in task.context.noise_colors:
                        cts[noise_color] = 0
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
            reduced="Tile",
        )
        if task:
            task.context.noise += noise
        return args
