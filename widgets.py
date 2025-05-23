import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

import keras

import numpy as np

import matplotlib.pyplot as plt

from typing import Tuple

from sklearn.preprocessing import StandardScaler


def upload_files_widget() -> Tuple[UploadedFile, UploadedFile, UploadedFile]:
    """
    Uploading files widget.

    Returns:
        Tuple[UploadedFile, UploadedFile, UploadedFile, UploadedFile]:
            A tuple containing the uploaded CSV data file, model file, feature_scaler file, target_scaler file.
    """

    uploaded_file = st.file_uploader("# Upload your CSV file", type=["csv"])
    model_file = st.file_uploader("Upload your trained Model file (.keras, .h5)", type=["keras", "h5"])
    feature_scaler_file = st.file_uploader("Upload your Feature Scaler file", type=["bin"])
    target_scaler_file = st.file_uploader("Upload your Target Scaler file", type=["bin"])

    return uploaded_file, model_file, feature_scaler_file, target_scaler_file


def generate_prediction_graph(
    model: keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    target_scaler: StandardScaler,
    step: int = 250,
) -> Tuple[np.ndarray, np.ndarray]:
    """
        Plots Prediction Graph on the Validation/Test Dataset.
        Depends upon the is_test parameter.

        Args:
            model (keras.Model): Trained Keras Model.
            X & y (np.ndarray): Input features and target.
            target_scaler (StandardScaler): Scaler to inverse transform predictions.
            step (int): Step size for plotting. Default is 250.

        Returns:
            Optional[Tuple[np.ndarray, np.ndarray]]

            if is_test is True: Returns y_true and y_pred for model metrics
    """

    y_pred_scaled = model.predict(X)

    # 2. Inverse scale to original units
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_true = target_scaler.inverse_transform(y)

    # 3. Plot first 100 samples or full range
    plt.figure(figsize=(12, 6))
    plt.plot(y_true[::step], label='Actual Temperature')
    plt.plot(y_pred[::step], label='Predicted Temperature')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature (°C)')
    plt.title(f"Actual vs Predicted Temperature on Test Set")
    plt.legend()

    st.pyplot(plt)

    return y_true, y_pred
