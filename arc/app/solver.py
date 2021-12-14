import streamlit as st

from arc.app.util import timed
from arc.app.viz import cached_plot


def solver(task_idx: int):
    title = "Choose step to stop after"
    options = ["Decomposition", "Matching"]
    process_idx = st.sidebar.selectbox(title, options, index=0)
    _arc = st.session_state.arc
    with st.expander(f"Visual overview of Task {task_idx}", expanded=True):
        st.image(cached_plot(task_idx))
    _arc[task_idx][0].input.decompose()
    with st.expander(f"Decomposition of the first Scene's input board", expanded=True):
        st.image(cached_plot((task_idx, 0), "Tree"))
