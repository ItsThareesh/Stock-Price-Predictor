import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

import tempfile

from helper import build_model, prepare_train_test_datasets
from widgets import custom_progress_bar, generate_prediction_graph, generate_history_graph, sidebar
from configs import init_session_state

init_session_state()

# Streamlit App
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈")

st.title("📈 Stock Price Predictor with LSTM")

st.markdown("<br><br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("# Upload your CSV file", type=["csv"])
model_file = st.file_uploader("Upload your trained Model file (.keras, .h5)", type=["keras", "h5"])


if uploaded_file:
    df = pd.read_csv(uploaded_file)

    date_column, close_column, test_split, window_size, epochs, batch_size, show_graph = sidebar(model_file, df)

    if st.session_state['DATE_UPDATED'] and st.session_state['CLOSE_UPDATED']:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            df.sort_values(date_column, inplace=True)
            df.reset_index(drop=True, inplace=True)

            st.markdown("<br>", unsafe_allow_html=True)  # Line Break

            st.write("### Preview Data:", df.head())

        except Exception as e:
            st.error(f"Error converting date column: {e}")

        try:
            training_data_len, scaler, X_train, y_train, X_test = prepare_train_test_datasets(df, close_column, test_split, window_size)

            if model_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                    tmp.write(model_file.read())
                    loaded_model_path = tmp.name

                model = load_model(loaded_model_path)  # Load Model

                _, col2, _ = st.columns([1, 1, 1])

                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Line Break
                    placeholder = st.empty()

                    if not st.session_state.get('RUN_PREDICT', False):
                        if placeholder.button("📈 Run Predictions"):
                            placeholder.empty()  # Hide Button instantly

                            predictions = model.predict(X_test)
                            predictions = scaler.inverse_transform(predictions)

                            st.session_state['RUN_PREDICT'] = True
                            st.session_state["PREDICTIONS"] = predictions

                if st.session_state.get('RUN_PREDICT', False):
                    st.success("✅ Model Predictions Completed!")
                    generate_prediction_graph(df, plt, date_column, close_column, training_data_len)

                # Cleanup temporary files

                if os.path.exists(loaded_model_path):
                    os.remove(loaded_model_path)

            else:
                model = build_model(X_train)

                _, col2, _ = st.columns([1, 1, 1])

                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Line Break
                    placeholder = st.empty()

                    if not st.session_state.get('CLICKED_TRAIN', False):
                        if placeholder.button("🚀 Train Model", disabled=st.session_state.get('CLICKED_TRAIN', False)):
                            placeholder.empty()  # Hide Button Instantly

                            st.session_state['CLICKED_TRAIN'] = True

                if st.session_state.get('CLICKED_TRAIN', False):
                    if not st.session_state.get('FINISHED_TRAINING', False):
                        try:
                            model = custom_progress_bar(st, epochs, batch_size, X_train, y_train, model)
                            predictions = model.predict(X_test)
                            predictions = scaler.inverse_transform(predictions)

                            st.session_state['CLICKED_TRAIN'] = True
                            st.session_state["PREDICTIONS"] = predictions

                            st.success("✅ Model Trained Successfully!")
                            st.rerun()

                        except Exception as e:
                            st.error("Something went wrong... Refresh the page and Start again!")

                    else:
                        if show_graph:
                            with st.expander(label="Training Graphs"):
                                generate_history_graph(st.session_state['HISTORY'], plt)

                        generate_prediction_graph(df, plt, date_column, close_column, training_data_len)

                        _, col2, _ = st.columns([1, 1, 1])

                        with col2:
                            with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
                                saved_model_path = tmp.name

                                model.save(saved_model_path)
                                tmp.seek(0)
                                binary = tmp.read()

                            st.markdown("<br>", unsafe_allow_html=True)  # Line Break

                            st.download_button(label="Download Trained Model (.keras)", data=binary, file_name="custom_trained_model.keras")

                            # Cleanup the temporary files

                            if os.path.exists(saved_model_path):
                                os.remove(saved_model_path)

        except Exception as e:
            st.error(f"Error processing data: {e}")
