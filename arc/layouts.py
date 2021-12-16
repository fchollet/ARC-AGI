import numpy as np

from arc.viz import Layout, PlotDef
from arc.object import Object


def tree_layout(obj: Object) -> Layout:
    objs = [obj]
    layout: Layout = []
    while objs:
        layout.append([{"grid": obj.grid, "name": obj._id} for obj in objs])
        objs = [
            kid for obj in objs for kid in obj.children if not kid.category == "Dot"
        ]

    # Pad the layout with blanks so each row has a fixed size
    max_width = max([len(row) for row in layout])
    for row in layout:
        for i in range(max_width - len(row)):
            row.append(None)
    return layout


# def scene_layout(scene: Any):
#     g1, shape1 = hier_layout(scene.input.rep)
#     g2, shape2 = hier_layout(scene.output.rep)
#     M, N = max(shape1[0], shape2[0]), shape1[1] + shape2[1] + 1
#     output = [[0] * N for i in range(M)]
#     for i in range(shape1[0]):
#         for j in range(shape1[1]):
#             output[i][j] = g1[i][j]
#     for i in range(shape2[0]):
#         for j in range(shape2[1]):
#             output[i][shape1[1] + 1 + j] = g2[i][j]
#     return output


# def leaf_layout(inp, out=None):
#     in_row = []
#     for obj in inp.inventory(leaf_only=True):
#         in_row.append({"grid": obj.grid, "name": obj._name})
#     out_row = []
#     for obj in out.inventory(leaf_only=True):
#         out_row.append({"grid": obj.grid, "name": obj._name})
#     pad_check = sorted([in_row, out_row], key=lambda x: len(x))
#     pad_check[0] += [0] * (len(pad_check[1]) - len(pad_check[0]))
#     return [in_row, out_row]


# def match_layout(scene):
#     grids = []
#     for delta in scene.path:
#         inp, out, trans = delta.right, delta.left, delta.transform
#         left = {"grid": inp.grid, "name": inp.category}
#         right = {"grid": out.grid, "name": trans}
#         grids.append([left, right])
#     return grids
