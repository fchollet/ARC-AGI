"""This is a very simple wrapper around the Streamlit UI code"""

import streamlit as st

from arc.app.ui import run_ui

st.set_page_config(layout="wide")

run_ui()
