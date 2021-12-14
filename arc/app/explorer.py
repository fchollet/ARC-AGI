import streamlit as st

from arc.app.settings import Settings
from arc.app.viz import cached_plot


def explorer():
    _arc = st.session_state.arc
    pages = []
    grid = [[] for i in range(Settings.grid_width)]
    curr_col = 0
    for i in _arc.selection:
        grid[curr_col].append(i)
        curr_col = (curr_col + 1) % Settings.grid_width
        if len(grid[-1]) >= Settings.grid_height:
            pages.append(grid)
            grid = [[] for i in range(Settings.grid_width)]

    pages.append(grid)
    page_idx = 0
    title_col, slider_col, _ = st.columns([3, 1, 1])
    with title_col:
        st.title(f"Explore each input to the first scene")
    if len(pages) > 1:
        with slider_col:
            page_idx = st.select_slider(
                label="Page", options=[str(i) for i in range(len(pages))]
            )

    grid = pages[int(page_idx)]
    columns = st.columns(Settings.grid_width)
    for column, task_idxs in zip(columns, grid):
        with column:
            for task_idx in task_idxs:

                def on_click(_idx: int):
                    def action():
                        st.session_state.task_idx = _idx

                    return action

                st.button(str(task_idx), on_click=on_click(task_idx))
                st.image(cached_plot((task_idx, 0)))
