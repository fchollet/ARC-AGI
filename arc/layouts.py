from arc.viz import Layout, PlotDef
from arc.object import Object, ObjectDelta


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


def match_layout(path: list[ObjectDelta]) -> Layout:
    layout: Layout = []
    for delta in path:
        inp, out, trans = delta.right, delta.left, delta.transform
        left: PlotDef = {"grid": inp.grid, "name": inp.category}
        right: PlotDef = {"grid": out.grid, "name": str(trans.items())}
        layout.append([left, right])
    return layout


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
