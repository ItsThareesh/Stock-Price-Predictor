import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

import keras

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from typing import Tuple

import math


def upload_files_widget() -> Tuple[UploadedFile, UploadedFile, UploadedFile]:
    """
    Uploading files widget.

    Returns:
        Tuple[UploadedFile, UploadedFile, UploadedFile]:
            A tuple containing the uploaded model file, scaler file, and data CSV file.
    """

    uploaded_file = st.file_uploader("# Upload your CSV file", type=["csv"])
    model_file = st.file_uploader("Upload your trained Model file (.keras, .h5)", type=["keras", "h5"])
    scaler_file = st.file_uploader("Upload your Scaler file", type=["bin"])

    return uploaded_file, model_file, scaler_file


def sidebar(model_file: UploadedFile, df: pd.DataFrame) -> Tuple[str, str, float, int, int, int, float, bool]:
    """
        Custom Sidebar for setting Dataframe and Model parameters.

        Args:
            model_file (UplodedFile): User uploaded Model file (.keras, .h5).
            df (pd.DataFrame): DataFrame containing the user uploaded CSV file.

        Returns:
            Tuple[str, str, float, int, int, int, float, bool]

            A tuple containing:
            - Date column name
            - Close column name
            - Test split fraction
            - Window size
            - Epochs
            - Batch size
            - Learning Rate
            - Show Model Training Graphs flag
    """

    st.sidebar.markdown("## Model Parameters")

    epochs = None
    batch_size = None
    show_graph = None
    test_split = None
    lr = None

    df_cols = df.columns.tolist()

    date_column = st.sidebar.selectbox("Select the Date column", df_cols, index=None, key='date_column',
                                       on_change=lambda: st.session_state.update({'DATE_UPDATED': True}))

    close_column = st.sidebar.selectbox("Select the Close Price column", df_cols, index=None, key='close_column',
                                        on_change=lambda: st.session_state.update({'CLOSE_UPDATED': True}))
    if not model_file:
        test_split = st.sidebar.slider("Test Data Fraction", min_value=0.05, max_value=0.5, step=0.05, value=0.05)

    window_size = st.sidebar.slider("Window Size", min_value=10, max_value=200, value=60, step=5)

    if not model_file:
        lr = st.sidebar.slider("Learning Rate", min_value=1e-6, max_value=1e-2, value=1e-3, step=5e-4, format="%.6f")
        epochs = st.sidebar.slider("Epochs", min_value=5, max_value=100, value=25, step=5)
        batch_size = st.sidebar.selectbox("Batch Size", options=[16, 32, 64, 128], index=1)
        show_graph = st.sidebar.checkbox("Show Graph", value=False)

    return date_column, close_column, test_split, window_size, epochs, batch_size, lr, show_graph


def custom_progress_bar(
        epochs: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model: keras.Model,
        batch_size: int
) -> keras.Model:
    """
        Custom Progress Bar for Training the model.

        Args:
            epochs (int): Number of epochs.
            X_train & y_train (np.ndarray): Training Data.
            X_val & y_val (np.ndarray): Validation Data.
            model (keras.Model): Keras Model.
            batch_size (int): Batch size.

        Returns:
            keras.Model: Trained Keras Model.
    """

    progress_bar = st.progress(0, text="⏳ Training in progress...")

    history_all = {
        'training_loss': [],
        'root_mean_squared_error': [],
        'val_loss': [],
        'val_root_mean_squared_error': []
    }

    for epoch in range(epochs):
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, batch_size=batch_size)

        # Extract metrics
        epoch_loss = history.history['loss'][0]
        rmse = history.history['root_mean_squared_error'][0]
        val_loss = history.history['val_loss'][0]
        val_rmse = history.history['val_root_mean_squared_error'][0]

        # Log history
        history_all['training_loss'].append(epoch_loss)
        history_all['root_mean_squared_error'].append(rmse)
        history_all['val_loss'].append(val_loss)
        history_all['val_root_mean_squared_error'].append(val_rmse)

        progress = int(((epoch + 1) / epochs) * 100)
        progress_bar.progress(progress, text=f"Epoch {epoch + 1}/{epochs}")

    st.session_state['FINISHED_TRAINING'] = True
    st.session_state['HISTORY'] = history_all

    return model


def generate_prediction_graph(df: pd.DataFrame, date_column: str, close_column: str, start_index: int) -> None:
    """
        Plots Prediction Graph.

        Args:
            df (pd.DataFrame): DataFrame containing the user uploaded CSV file.
            date_column (str): Date column name.
            close_column (str): Close price column name.
            start_index (int): Length of training data.

        Returns:
            None
    """

    predictions = st.session_state.get('PREDICTIONS')

    test_df = df[start_index:]

    st.write("### 📊 Prediction vs Actual")

    plt.figure(figsize=(12, 6))
    plt.plot(test_df[date_column], test_df[close_column], label="Validation Set (Actual)", color='orange')
    plt.plot(test_df[date_column], predictions, label="Predictions", color='red', alpha=0.65)
    plt.title("Stock Predictions - Zoomed Validation Set")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()

    st.pyplot(plt)


def generate_history_graph(history: dict) -> None:
    """
        Plots History Graph.

        Args:
            history (dict): History Dictionary stored in Session State.

        Returns:
            None
    """

    metrics = list(history.keys())
    num_metrics = len(metrics)
    cols = 2
    rows = math.ceil(num_metrics / cols)

    plt.figure(figsize=(8 * cols, 6 * rows))

    for idx, metric in enumerate(metrics, 1):
        plt.subplot(rows, cols, idx)
        plt.plot(history[metric], label=metric)
        plt.title(f"{metric} over Epochs")
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()

    st.pyplot(plt)
