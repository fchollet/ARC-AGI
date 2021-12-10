from typing import Any, Optional

import streamlit as st

from arc.main import ARC
from arc.app.explorer import explorer, single
from arc.app.util import timed
from arc.app.settings import Settings


class UI:
    def __init__(self) -> None:
        if "idx" not in st.session_state:
            st.session_state.idx = None

        N = menu()

        _arc = init(N)

        selector(_arc)

        if st.session_state.idx is not None:
            single(task_idx=st.session_state.idx)
        else:
            explorer()


def menu() -> int:
    # st.title("Exploring and Solving the ARC challenge")

    blacklist = [8, 16, 28, 54, 60, 69, 73]  # Involves larger, tiled boards
    mode_options = ["Explore", "Demo", "Stats", "Dev"]
    mode = st.sidebar.selectbox("Select a mode", mode_options, index=3)

    if mode == "Stats":
        N = 400
    elif mode == "Dev":
        N = 48
    else:
        N = 10
    return N


def selector(_arc: ARC) -> None:
    title = "Choose a task"
    options = ["None"] + [str(i) for i in _arc.tasks]
    index = 0 if st.session_state.idx is None else st.session_state.idx + 1
    task_idx = st.sidebar.selectbox(title, options, index=index)
    if task_idx == "None":
        st.session_state.idx = None
    else:
        st.session_state.idx = int(task_idx)


def init(N: int):
    if "arc" not in st.session_state or st.session_state.arc.N != N:
        st.write(f"Loading ARC dataset ({N} tasks)")
        st.session_state.arc = ARC(N=N, folder=Settings.folder)
    return st.session_state.arc
