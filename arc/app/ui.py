import streamlit as st

from arc.main import ARC
from arc.app.explorer import explorer
from arc.app.solver import solver
from arc.app.settings import Settings


def run_ui() -> None:
    """Central UI logic governing components to display."""
    init_session()
    mode_selector()
    task_selector()

    if st.session_state.task_idx > 0:
        solver(task_idx=st.session_state.task_idx)
    else:
        explorer()


def init_session() -> None:
    if "arc" not in st.session_state:
        st.session_state.arc = None


def mode_selector() -> None:
    # st.title("Exploring and Solving the ARC challenge")

    options = ["Demo", "Dev", "Stats"]
    st.sidebar.selectbox("Select a mode", options, key="mode", index=1)

    if st.session_state.mode == "Stats":
        N = 400
    elif st.session_state.mode == "Dev":
        N = 40
    else:
        N = 2

    st.write(f"Loading ARC dataset ({N} tasks)")
    st.session_state.arc = ARC(N=N, folder=Settings.folder)


def task_selector() -> None:
    if st.session_state.arc is None:
        st.write("No ARC dataset loaded")
        return
    title = "Choose a task"
    options = [0] + list(st.session_state.arc.selection)

    def labeler(option: int) -> str:
        if option == 0:
            return "Explore"
        return str(option)

    st.sidebar.selectbox(title, options, index=0, format_func=labeler, key="task_idx")
