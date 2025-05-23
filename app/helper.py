import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import keras
from keras import layers
from keras import models
from keras import optimizers

import joblib

from typing import Tuple

from widgets import generate_prediction_graph

import tempfile
import os


@st.cache_data
def load_csv(uploaded_file: UploadedFile) -> pd.DataFrame:
    """
        Loads the CSV file into a DataFrame.

        Args:
            uploaded_file (UploadedFile): User uploaded CSV File.

        Returns:
            pd.DataFrame: DataFrame containing the data.
    """

    df = pd.read_csv(uploaded_file)
    df = df[5::6]

    return df


@st.cache_data
def clean_data(df) -> pd.DataFrame:
    """
        Cleans the input DataFrame removing unnecessary columns and performing feature engineering.

        Args:
            df (pd.DataFrame): Input DataFrame.
        Returns:
            pd.DataFrame: Cleaned DataFrame with Engineering Features.
    """

    cleaned_df = df.copy()

    date_time = pd.to_datetime(cleaned_df.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

    # Add engineered time features
    cleaned_df['Month'] = date_time.dt.month
    cleaned_df['Week'] = date_time.dt.isocalendar().week
    cleaned_df['Year'] = date_time.dt.year

    cleaned_df.dropna(inplace=True)

    # Remove Outlier / Erroneous Values
    cleaned_df.loc[-9999.0, 'wv (m/s)'] = 0.0
    cleaned_df.loc[-9999.0, 'max. wv (m/s)'] = 0.0

    wv = cleaned_df.pop('wv (m/s)')
    max_wv = cleaned_df.pop('max. wv (m/s)')

    # Convert to radians.
    wd_rad = cleaned_df.pop('wd (deg)') * np.pi / 180

    # Calculate the wind x and y components.
    cleaned_df['Wx'] = wv * np.cos(wd_rad)
    cleaned_df['Wy'] = wv * np.sin(wd_rad)

    # Calculate the max wind x and y components.
    cleaned_df['max Wx'] = max_wv * np.cos(wd_rad)
    cleaned_df['max Wy'] = max_wv * np.sin(wd_rad)

    timestamp_s = date_time.map(pd.Timestamp.timestamp)

    day = 24*60*60
    year = (365.2425)*day

    cleaned_df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    cleaned_df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    cleaned_df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    cleaned_df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    # Drop rows with missing values
    cleaned_df.dropna(inplace=True)
    cleaned_df.isna().sum()

    return cleaned_df


def build_model(X_train: np.ndarray, learning_rate: float) -> keras.Model:
    """
    Builds and compiles the Deep Learning model.

    Args:
        X_train (np.ndarray): Training data.
        learning_rate (float): Learning Rate for the optimizer.

    Retruns:
        keras.Model: Compiled Keras model.
    """

    model = models.Sequential(name="ImprovedTempPredictor")

    model.add(layers.Input(shape=(X_train.shape[1], X_train.shape[2])))

    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())

    model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
    model.add(layers.BatchNormalization())

    model.add(layers.Bidirectional(layers.LSTM(32)))
    model.add(layers.BatchNormalization())

    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='mae',
        metrics=['root_mean_squared_error', 'r2_score']
    )

    model.summary()

    return model


def create_sliding_window(features: np.ndarray, target: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sliding windows for the given data to provide context for LSTM Prediction.

    Args:
        features (np.ndarray): Input feature values.
        target (np.ndarray): Target values.
        window_size (int): Size of the sliding window.

    Retruns:
        Tuple[np.ndarray, np.ndarray]
    """

    X, y = [], []
    total_length = len(features)

    for i in range(total_length - window_size):
        X.append(features[i:i+window_size])
        y.append(target[i + window_size])

    return (np.array(X), np.array(y))


@st.cache_resource
def load_uploaded_model(model_path: str) -> models.Model:
    """
        Loads the model from the specified path. Written only to cache the process of fetching model.

        Args:
            model_path (str): Path to the model file (.keras).

        Returns:
            StandardScaler object.
    """

    return models.load_model(model_path)


@st.cache_resource(show_spinner="Loading Scaler...")
def load_scaler(scaler_path: str) -> StandardScaler:
    """
        Loads the feature/target scaler from the specified path.

        Args:
            scaler_path (str): Path to the scaler file (.joblib).

        Returns:
            StandardScaler object.
    """

    return joblib.load(scaler_path)


def prepare_train_test_datasets(
    df: pd.DataFrame,
    window_size: int,
    feature_scaler_path: str,
    target_scaler_path: str
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare the training and validation datasets. Or prepare the test dataset.
    Depends upon the test_split parameter.

    Args:
        df (pd.DataFrame): Input dataframe.
        test_split (float, optional): Fraction of data used for validation. If None, treats input as test data.
        window_size (int): Size of the sliding window. Set by user.
        feature_scaler_path (str): Pre-fitted feature scaler (for test). Uploaded by user.
        target_scaler_path (str): Pre-fitted target scaler (for test). Uploaded by user.

    Returns:
        Tuple[np.ndarray, np.ndarray, StandardScaler]
    """

    features = df.drop(columns=['T (degC)']).values
    target = df['T (degC)'].values.reshape(-1, 1)

    feature_scaler = load_scaler(feature_scaler_path)
    target_scaler = load_scaler(target_scaler_path)

    scaled_features = feature_scaler.transform(features)
    scaled_target = target_scaler.transform(target)

    X_test, y_test = create_sliding_window(scaled_features, scaled_target, window_size)

    return X_test, y_test, target_scaler


def run_inference(
    df: pd.DataFrame,
    model_file: UploadedFile,
    feature_scaler_file: UploadedFile,
    target_scaler_file: UploadedFile,
    window_size: int,
    step_size: int
):
    """
        Run Inference on the uploaded data using the pre-trained model.

        Args:
            df (pd.DataFrame): Input Cleaned DataFrame.
            model_file (UploadedFile): Model File uploaded by user.
            feature_scaler_file (UploadedFile): Feature Scaler File uploaded by user.
            target_scaler_file (UploadedFile): Target Scaler File uploaded by user.
            window_size (int): Size of the sliding window.
            step_size (int): Step size for plotting.

        Returns:
            None
    """

    with tempfile.TemporaryDirectory() as tmpdir:
        loaded_model_path = os.path.join(tmpdir, "model.keras")  # Save model
        feature_scaler_path = os.path.join(tmpdir, "feature_scaler.bin")  # Save feature scaler
        target_scaler_path = os.path.join(tmpdir, "target_scaler.bin")  # Save target scaler

        with open(loaded_model_path, "wb") as f:
            f.write(model_file.read())

        with open(feature_scaler_path, "wb") as f:
            f.write(feature_scaler_file.read())

        with open(target_scaler_path, "wb") as f:
            f.write(target_scaler_file.read())

        n = len(df)  # No. of rows
        test_df = df[int(n*0.9):]

        X_test, y_test, target_scaler = prepare_train_test_datasets(test_df, window_size, feature_scaler_path, target_scaler_path)

        model = load_uploaded_model(loaded_model_path)

        _, col2, _ = st.columns([1, 1, 1])

        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Line Break
            placeholder = st.empty()

            if placeholder.button("📈 Run Predictions"):
                try:
                    st.session_state['CLICKED_RUN'] = True
                    placeholder.empty()

                    predictions = model.predict(X_test)
                    predictions = target_scaler.inverse_transform(predictions)

                    st.session_state["PREDICTIONS"] = predictions

                    placeholder.success("✅ Model Predictions Completed!")

                except Exception as e:
                    placeholder.error(f"⚠️ Error: {str(e)}")
                    st.session_state['CLICKED_RUN'] = False

        # Plot only if predictions exist
        if st.session_state.get('CLICKED_RUN') and "PREDICTIONS" in st.session_state:
            generate_prediction_graph(model, X_test, y_test, target_scaler, step_size)
            st.session_state["CLICKED_RUN"] = False
