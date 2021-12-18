from typing import TypeAlias, TypedDict
import matplotlib
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np

from arc.types import TaskData


class PlotDef(TypedDict):
    grid: np.ndarray
    name: str


Layout: TypeAlias = list[list[PlotDef | None]]

color_map = matplotlib.colors.ListedColormap(  # type: ignore
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


def plot_color_map() -> Figure:
    # -1, 10: dark grey (transparent for purposes of a board)
    # 0:black, 1:blue, 2:red, 3:greed, 4:yellow,
    # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
    fig = plt.figure(figsize=(3, 1), dpi=200)
    plt.imshow([list(range(11))], cmap=color_map, norm=norm)
    plt.xticks(list(range(11)))
    plt.yticks([])
    return fig


def plot_layout(layout: Layout, scale: float = 1.0, show_axis: bool = True) -> Figure:
    M, N = len(layout), max([len(row) for row in layout])
    fig, axs = plt.subplots(M, N, figsize=(10 * scale, 10 * scale), dpi=100)
    for r, row in enumerate(layout):
        for c, args in enumerate(row):
            if M == 1 and N == 1:
                curr = axs
            elif M == 1:
                curr = axs[c]
            elif N == 1:
                curr = axs[r]
            else:
                curr = axs[r][c]
            if args is None:
                curr.axis("off")
                continue
            if not show_axis:
                curr.axis("off")
            grid = args["grid"]
            curr.set_title(args["name"], {"fontsize": 6})
            curr.imshow(grid, cmap=color_map, norm=norm)
            curr.set_yticks(list(range(grid.shape[0])))
            curr.set_xticks(list(range(grid.shape[1])))
    plt.tight_layout()
    return fig


def plot_grid(grid: np.ndarray) -> Figure:
    fig, axs = plt.subplots(1, 1, figsize=(4, 4), dpi=50)
    axs.imshow(grid, cmap=color_map, norm=norm)
    axs.set_yticks(list(range(grid.shape[0])))
    axs.set_xticks(list(range(grid.shape[1])))
    # plt.tight_layout()
    return fig
