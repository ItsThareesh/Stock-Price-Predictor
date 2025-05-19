import streamlit as st


def init_session_state():
    if 'DATE_UPDATED' not in st.session_state:
        st.session_state['DATE_UPDATED'] = False
    if 'CLOSE_UPDATED' not in st.session_state:
        st.session_state['CLOSE_UPDATED'] = False
    if 'TRAINED' not in st.session_state:
        st.session_state['TRAINED'] = False
    if 'RUN_PREDICT' not in st.session_state:
        st.session_state['RUN_PREDICT'] = False


def update_date():
    st.session_state['DATE_UPDATED'] = True


def update_close():
    st.session_state['CLOSE_UPDATED'] = True
