import streamlit as st

from helper import load_csv, run_inference, clean_data
from widgets import upload_files_widget

# Streamlit App
st.set_page_config(page_title="Temperature Predictor", page_icon="🌡️", layout="wide")

st.title("🌡️ Temperature Predictor with LSTM + Conv1D")

st.markdown("<br><br>", unsafe_allow_html=True)

uploaded_file, model_file, feature_scaler_file, target_scaler_file = upload_files_widget()


if uploaded_file:
    df = load_csv(uploaded_file)

    st.markdown("<br>", unsafe_allow_html=True)
    st.write(df.head())

    clean_df = clean_data(df)

    st.markdown("<br>", unsafe_allow_html=True)  # Line Break
    with st.expander(label="Feature Engineered Data", expanded=True):
        st.write(clean_df.head())

    window_size = st.slider("Window Size", min_value=12, max_value=72, value=24, step=12)
    step_size = st.slider("Step Size", min_value=50, max_value=500, value=250, step=50)

    try:
        if model_file and feature_scaler_file and target_scaler_file:
            run_inference(clean_df, model_file, feature_scaler_file, target_scaler_file, window_size, step_size)

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
