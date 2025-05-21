import os

import streamlit as st
import pandas as pd

import tempfile
import joblib
import zipfile

from helper import build_model, load_csv, prepare_train_test_datasets, load_uploaded_model
from widgets import custom_progress_bar, generate_prediction_graph, generate_history_graph, sidebar, upload_files_widget
from configs import init_session_state

init_session_state()

# Streamlit App
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈")

st.title("📈 Stock Price Predictor with LSTM")

st.markdown("<br><br>", unsafe_allow_html=True)

uploaded_file, model_file, scaler_file = upload_files_widget()


if uploaded_file:
    df = load_csv(uploaded_file)

    date_column, close_column, test_split, window_size, epochs, batch_size, lr, show_graph = sidebar(model_file, df)

    if st.session_state["DATE_UPDATED"] and st.session_state["CLOSE_UPDATED"]:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            df.sort_values(date_column, inplace=True)
            df.reset_index(drop=True, inplace=True)

            st.markdown("<br>", unsafe_allow_html=True)  # Line Break

            st.write("### Preview Data:", df.head())

        except Exception as e:
            st.error(f"Error converting date column: {e}")

        try:
            if model_file and scaler_file:
                with tempfile.TemporaryDirectory() as tmpdir:
                    loaded_model_path = os.path.join(tmpdir, "loaded_model.keras")  # Save model
                    scaler_path = os.path.join(tmpdir, "loaded_scaler.bin")  # Save scaler

                    with open(loaded_model_path, "wb") as f:
                        f.write(model_file.read())

                    with open(scaler_path, "wb") as f:
                        f.write(scaler_file.read())

                    X_test, scaler = prepare_train_test_datasets(df, close_column,
                                                                 None, window_size,
                                                                 scaler_path, True)

                    model = load_uploaded_model(loaded_model_path)

                    _, col2, _ = st.columns([1, 1, 1])

                    with col2:
                        st.markdown("<br>", unsafe_allow_html=True)  # Line Break

                        if st.button("📈 Run Predictions"):
                            try:
                                predictions = model.predict(X_test)
                                predictions = scaler.inverse_transform(predictions)
                                st.session_state["PREDICTIONS"] = predictions

                                st.session_state["RUN_PREDICT"] = True
                            except Exception as e:
                                st.error(f"⚠️ Error: {str(e)}")

                    if st.session_state.get('RUN_PREDICT', False):
                        st.success("✅ Model Predictions Completed!")
                        generate_prediction_graph(df, date_column, close_column, window_size)

                        st.session_state["RUN_PREDICT"] = False

            else:
                training_data_len, X_train, y_train, X_test, scaler = prepare_train_test_datasets(df, close_column,
                                                                                                  test_split, window_size)

                model = build_model(X_train, lr)

                _, col2, _ = st.columns([1, 1, 1])

                with col2:
                    st.markdown("<br>", unsafe_allow_html=True)  # Line Break
                    placeholder = st.empty()

                    if not st.session_state.get('CLICKED_TRAIN', False):
                        if placeholder.button("🚀 Train Model", disabled=st.session_state.get('CLICKED_TRAIN', False)):
                            placeholder.empty()  # Hide Button Instantly

                            st.session_state["CLICKED_TRAIN"] = True

                if st.session_state.get('CLICKED_TRAIN', False):
                    if not st.session_state.get('FINISHED_TRAINING', False):
                        try:
                            model = custom_progress_bar(epochs, X_train, y_train, model)

                            predictions = model.predict(X_test)
                            predictions = scaler.inverse_transform(predictions)
                            st.session_state["PREDICTIONS"] = predictions

                            st.success("✅ Model Trained Successfully!")
                            st.rerun()

                        except Exception as e:
                            st.error(f"⚠️ Error: {str(e)}")

                    else:
                        if show_graph:
                            with st.expander(label="Training Graphs"):
                                generate_history_graph(st.session_state["HISTORY"])

                        generate_prediction_graph(df, date_column, close_column, training_data_len)

                        _, col2, _ = st.columns([1, 1, 1])

                        with col2:
                            with tempfile.TemporaryDirectory() as tmpdir:
                                model_path = os.path.join(tmpdir, "model.keras")
                                scaler_path = os.path.join(tmpdir, "scaler.bin")
                                zip_path = os.path.join(tmpdir, "model_package.zip")

                                # Save model
                                model.save(model_path)

                                # Save scaler
                                joblib.dump(scaler, scaler_path)

                                # Zip both
                                with zipfile.ZipFile(zip_path, 'w') as zipf:
                                    zipf.write(model_path, arcname="model.keras")
                                    zipf.write(scaler_path, arcname="scaler.bin")

                                # Read zip to bytes
                                with open(zip_path, "rb") as f:
                                    zip_bytes = f.read()

                                st.markdown("<br>", unsafe_allow_html=True)

                                st.download_button(label="Download Model + Scaler (ZIP)", data=zip_bytes, file_name="model_and_scaler.zip")

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
