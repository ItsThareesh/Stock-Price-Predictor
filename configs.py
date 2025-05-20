import streamlit as st


def init_session_state():
    defaults = {
        'DATE_UPDATED': False,
        'CLOSE_UPDATED': False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
