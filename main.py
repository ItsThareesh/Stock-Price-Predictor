import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model

import tempfile

from helper import build_model, prepare_train_test_datasets
from widgets import custom_progress_bar, show_graph
import configs

configs.init_session_state()

# Streamlit App
st.title("📈 Stock Price Predictor with LSTM")

st.markdown("<br><br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("# Upload your stock CSV file", type=["csv"])
model_file = st.file_uploader("Upload your trained Keras Model (.keras)", type=["keras"])


if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander(label="Model Parameters"):
        # Dataframe Paramters
        df_cols = df.columns.tolist()

        date_column = st.selectbox("Select the Date column", df_cols, index=None, key='date_column', on_change=configs.update_date)
        close_column = st.selectbox("Select the Close Price column", df_cols, index=None, key='close_column', on_change=configs.update_close)

        # Model Parameters
        test_split = st.slider("Test Data Fraction", 0.05, 0.5, 0.1, step=0.05)
        window_size = st.slider("Window Size", min_value=10, max_value=200, value=60, step=5)
        epochs = st.slider("Epochs", min_value=1, max_value=100, value=25)
        batch_size = st.selectbox("Batch Size", options=[16, 32, 64, 128], index=1)

    if st.session_state['DATE_UPDATED']:
        try:
            df[date_column] = pd.to_datetime(df[date_column])
            df.sort_values(date_column, inplace=True)
            df.reset_index(drop=True, inplace=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.write("### Preview Data:", df.head())

        except Exception as e:
            st.error(f"Error converting date column: {e}")

        if st.session_state['CLOSE_UPDATED']:
            try:
                training_data_len, scaler, X_train, y_train, X_test = prepare_train_test_datasets(df, close_column, test_split, window_size)

                if model_file:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                        tmp.write(model_file.read())
                        tmp_path = tmp.name

                    model = load_model(tmp_path)

                    # Predictions
                    col1, col2, col3 = st.columns([1, 1, 1])

                    with col2:
                        if not st.session_state['RUN_PREDICT']:
                            st.markdown("<br><br>", unsafe_allow_html=True)

                            if st.button("Run Predictions"):
                                with st.spinner("Running Predictions..."):
                                    predictions = model.predict(X_test)
                                    predictions = scaler.inverse_transform(predictions)

                                st.session_state['RUN_PREDICT'] = True
                                st.session_state["PREDICTIONS"] = predictions

                    if st.session_state['RUN_PREDICT']:
                        st.success("✅ Model Predictions Completed!")
                        st.markdown("<br>", unsafe_allow_html=True)
                        show_graph(df, plt, date_column, close_column, training_data_len)

                else:
                    model = build_model(X_train)

                    col1, col2, col3 = st.columns([1, 1, 1])

                    with col2:
                        if not st.session_state['TRAINED']:
                            st.markdown("<br><br>", unsafe_allow_html=True)

                            if st.button("🚀 Train Model"):
                                with st.spinner("Training Model..."):
                                    model = custom_progress_bar(epochs, batch_size, X_train, y_train, model)
                                    predictions = model.predict(X_test)
                                    predictions = scaler.inverse_transform(predictions)

                                st.session_state['TRAINED'] = True
                                st.session_state["PREDICTIONS"] = predictions

                    if st.session_state['TRAINED']:
                        st.success("✅ Model Trained Successfully!")
                        st.markdown("<br>", unsafe_allow_html=True)
                        show_graph(df, plt, date_column, close_column, training_data_len)

            except Exception as e:
                st.error(f"Error processing data: {e}")
