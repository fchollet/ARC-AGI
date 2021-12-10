from typing import Any
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

cmap = matplotlib.colors.ListedColormap(  # type: ignore
    [
        "#555555",
        "#000000",
        "#0074D9",
        "#FF2222",
        "#2ECC40",
        "#FFDC00",
        "#AAAAAA",
        "#F012BE",
        "#FF8C00",
        "#7FDBFF",
        "#870C25",
        "#555555",
    ]
)
norm = matplotlib.colors.Normalize(vmin=-1, vmax=10)  # type: ignore


def plot_cmap():
    # -1, 10: dark grey (transparent for purposes of a board)
    # 0:black, 1:blue, 2:red, 3:greed, 4:yellow,
    # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
    fig = plt.figure(figsize=(3, 1), dpi=200)
    plt.imshow([list(range(11))], cmap=cmap, norm=norm)
    plt.xticks(list(range(11)))
    plt.yticks([])
    return fig


def hier_layout(obj):
    objs = [obj]
    grids = []
    while objs:
        grids.append([{"grid": obj.grid, "name": obj._name} for obj in objs])
        objs = [kid for obj in objs for kid in obj.children if not kid.is_dot()]
    M, N = len(grids), max([len(row) for row in grids])
    for row in grids:
        row += [0] * (N - len(row))
    return grids, (M, N)


def scene_layout(scene: Any):
    g1, shape1 = hier_layout(scene.input.rep)
    g2, shape2 = hier_layout(scene.output.rep)
    M, N = max(shape1[0], shape2[0]), shape1[1] + shape2[1] + 1
    output = [[0] * N for i in range(M)]
    for i in range(shape1[0]):
        for j in range(shape1[1]):
            output[i][j] = g1[i][j]
    for i in range(shape2[0]):
        for j in range(shape2[1]):
            output[i][shape1[1] + 1 + j] = g2[i][j]
    return output


def leaf_layout(inp, out=None):
    in_row = []
    for obj in inp.inventory(leaf_only=True):
        in_row.append({"grid": obj.grid, "name": obj._name})
    out_row = []
    for obj in out.inventory(leaf_only=True):
        out_row.append({"grid": obj.grid, "name": obj._name})
    pad_check = sorted([in_row, out_row], key=lambda x: len(x))
    pad_check[0] += [0] * (len(pad_check[1]) - len(pad_check[0]))
    return [in_row, out_row]


def match_layout(scene):
    grids = []
    for delta in scene.path:
        inp, out, trans = delta.right, delta.left, delta.transform
        left = {"grid": inp.grid, "name": inp.category}
        right = {"grid": out.grid, "name": trans}
        grids.append([left, right])
    return grids


def plot_layout(grids):
    M, N = len(grids), max([len(row) for row in grids])
    fig, axs = plt.subplots(M, N, figsize=(5, 5), dpi=150)
    for r, row in enumerate(grids):
        for c, args in enumerate(row):
            if M == 1 and N == 1:
                curr = axs
            elif M == 1:
                curr = axs[c]
            elif N == 1:
                curr = axs[r]
            else:
                curr = axs[r][c]
            curr.axis("off")
            if isinstance(args, int):
                continue
            grid = args["grid"]
            curr.set_title(args["name"], {"fontsize": 6})
            curr.imshow(grid, cmap=cmap, norm=norm)
            curr.set_yticks(list(range(grid.shape[0])))
            curr.set_xticks(list(range(grid.shape[1])))
    plt.tight_layout()
    return fig


def plot_grid(grid: np.ndarray):
    fig, axs = plt.subplots(1, 1, figsize=(4, 4), dpi=50)
    axs.imshow(grid, cmap=cmap, norm=norm)
    axs.set_yticks(list(range(grid.shape[0])))
    axs.set_xticks(list(range(grid.shape[1])))
    # plt.tight_layout()
    return fig


def plot_scenes(
    scene_lists: list[list[tuple[np.ndarray, np.ndarray]]], groups: list[str]
):
    n = sum([len(scenes) for scenes in scene_lists])
    fig, axs = plt.subplots(2, n, figsize=(4 * n, 8), dpi=50)
    plt.subplots_adjust(wspace=0, hspace=0)
    f_idx = 0
    for group, scenes in zip(groups, scene_lists):
        for scene_idx, (grid_in, grid_out) in enumerate(scenes):
            axs[0][f_idx].imshow(grid_in, cmap=cmap, norm=norm)
            axs[0][f_idx].set_title(f"{group}-{scene_idx} in")
            axs[0][f_idx].set_yticks(list(range(grid_in.shape[0])))
            axs[0][f_idx].set_xticks(list(range(grid_in.shape[1])))
            axs[1][f_idx].imshow(grid_out, cmap=cmap, norm=norm)
            axs[1][f_idx].set_title(f"{group}-{scene_idx} out")
            axs[1][f_idx].set_yticks(list(range(grid_out.shape[0])))
            axs[1][f_idx].set_xticks(list(range(grid_out.shape[1])))
            f_idx += 1
    plt.tight_layout()
    return fig


def plot_raw_task(task):
    n = len(task["train"]) + len(task["test"])
    fname = task["fname"]
    print(f"Filename: {fname}.json")
    fig, axs = plt.subplots(2, n, figsize=(4 * n, 8), dpi=50)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f"Train-{i} in")
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f"Train-{i} out")
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f"Test-{i} in")
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f"Test-{i} out")
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1

    plt.tight_layout()
    return fig
