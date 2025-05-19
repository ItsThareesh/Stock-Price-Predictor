import streamlit as st


def init_session_state():
    defaults = {
        'DATE_UPDATED': False,
        'CLOSE_UPDATED': False,
        'TRAINED': False,
        'RUN_PREDICT': False
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def update_date():
    st.session_state['DATE_UPDATED'] = True


def update_close():
    st.session_state['CLOSE_UPDATED'] = True
