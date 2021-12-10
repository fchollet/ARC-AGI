from typing import Any
from matplotlib.figure import Figure
import numpy as np
import streamlit as st

from arc.main import ARC
from arc.viz import plot_grid
from arc.app.settings import Settings
from arc.app.util import timed


def explorer():
    _arc = st.session_state.arc
    pages = []
    grid = [[] for i in range(Settings.grid_width)]
    for i in _arc.selection:
        grid[i % Settings.grid_width].append(i)
        if len(grid[-1]) >= Settings.grid_height:
            pages.append(grid)
            grid = [[] for i in range(Settings.grid_width)]

    pages.append(grid)
    idx = 0
    title_col, slider_col, _ = st.columns([3, 1, 1])
    with title_col:
        st.title(f"Explore each input to the first scene")
    if len(pages) > 1:
        with slider_col:
            idx = st.select_slider(
                label="Page", options=[str(i) for i in range(len(pages))]
            )

    grid = pages[int(idx)]
    columns = st.columns(Settings.grid_width)
    for column, task_idxs in zip(columns, grid):
        with column:
            for task_idx in task_idxs:

                def on_click(_idx: int):
                    def action():
                        st.session_state.idx = _idx

                    return action

                st.button(str(task_idx), on_click=on_click(task_idx))
                st.pyplot(cached_plot((task_idx, 0)))


@st.cache(allow_output_mutation=True, ttl=None)
def cached_plot(plot_idx: tuple[int, int]) -> Figure:
    _arc = st.session_state.arc
    return plot_grid(_arc[plot_idx].input.rep.grid)


def single(task_idx: int):
    _arc = st.session_state.arc
    with st.expander(f"Visual overview of Task {task_idx}", expanded=True):
        st.pyplot(fig=_arc[task_idx].plot())
