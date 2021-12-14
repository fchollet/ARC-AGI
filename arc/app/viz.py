from io import BytesIO
from matplotlib.figure import Figure
from matplotlib import pyplot
import streamlit as st

from arc.viz import Layout, plot_grid, plot_layout
from arc.object import Object


@st.cache(allow_output_mutation=True, ttl=None)
def cached_plot(plot_idx: int | tuple[int, int], plot_type: str = None) -> BytesIO:
    _arc = st.session_state.arc
    image_buffer = BytesIO()
    if plot_type == "Tree":
        fig: Figure = plot_layout(hier_layout(_arc[plot_idx].input.rep))
    else:
        match plot_idx:
            case int(task_idx):
                fig: Figure = _arc[task_idx].plot()
            case (task_idx, scene_idx):
                fig: Figure = plot_grid(_arc[(task_idx, scene_idx)].input.rep.grid)
            case _:
                return image_buffer

    fig.savefig(image_buffer, format="png")
    pyplot.close(fig)
    return image_buffer


def hier_layout(obj: Object) -> Layout:
    objs = [obj]
    layout = []
    while objs:
        layout.append([{"grid": obj.grid, "name": obj._id} for obj in objs])
        objs = [
            kid for obj in objs for kid in obj.children if not kid.category == "Dot"
        ]

    # Pad the layout with blanks so each row has a fixed size
    max_width = max([len(row) for row in layout])
    for row in layout:
        row += [0] * (max_width - len(row))
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
