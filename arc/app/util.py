import functools
import streamlit as st
import time


def timed(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        dt = time.time() - t1
        st.write(f"...finished in {dt:.3f}s")
        return result

    return wrapped
