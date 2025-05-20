import streamlit as st


def init_session_state():
    defaults = {
        'DATE_UPDATED': False,
        'CLOSE_UPDATED': False,
        'CLICKED_TRAIN': False,
        'FINISHED': False,
        'RUN_PREDICT': False,
        'HISTORY': {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
