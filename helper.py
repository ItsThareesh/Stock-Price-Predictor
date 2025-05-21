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

from typing import Tuple, Union, Optional


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
    return df


@st.cache_resource
def build_model(X_train: np.ndarray, learning_rate: float) -> keras.Model:
    """
    Builds and compiles the Deep Learning model.

    Args:
        X_train (np.ndarray): Training data.
        learning_rate (float): Learning Rate se

    Retruns:
        keras.Model: Compiled Keras model.
    """

    model = models.Sequential(name="StockPriceLSTM")
    model.add(layers.Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(layers.LSTM(128, return_sequences=True))
    model.add(layers.LSTM(64, return_sequences=False))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))
    model.summary()

    model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["root_mean_squared_error"])

    return model


def create_sliding_window(data: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates sliding windows for the given data to provide context for LSTM Prediction.

    Args:
        data (np.ndarray): Input data.
        window_size (int): Size of the sliding window.

    Retruns:
        Tuple[np.ndarray, np.ndarray]
    """

    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])

    return np.array(X), np.array(y).reshape(-1)


@st.cache_resource
def load_uploaded_model(model_path: str) -> models.Model:
    """
        Loads the model from the specified path. Written only to cache the process of fetching model.

        Args:
            model_path (str): Path to the model file (.keras, .h5).

        Returns:
            StandardScaler object.
    """

    return models.load_model(model_path)


@st.cache_resource(show_spinner="Loading Scaler...")
def load_scaler(scaler_path: str) -> StandardScaler:
    """
        Loads the scaler from the specified path.

        Args:
            scaler_path (str): Path to the scaler file (.bin).

        Returns:
            StandardScaler object.
    """

    return joblib.load(scaler_path)


def prepare_train_test_datasets(
    df: pd.DataFrame,
    close_column: str,
    test_split: Optional[float],
    window_size: int,
    scaler_path: Optional[str] = None,
    only_predict=False
) -> Union[
    Tuple[int, np.ndarray, np.ndarray, np.ndarray, StandardScaler],
    Tuple[int, np.ndarray, StandardScaler]
]:
    """
    Prepare the training and testing datasets.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        close_column (str): Name of the column containing close prices. Selected in sidebar by user.
        test_split (float): Fraction of data to be used for testing. Selected in sidebar by user.
        window_size (int): Size of the sliding window. Selected in sidebar by user.
        scaler_path (str, optional): Path to the scaler file. Defaults to None.
        only_predict (bool, optional): Flag to indicate if only prediction is needed. Defaults to False.

    Returns:
        Union[
            Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler],
            Tuple[np.ndarray, StandardScaler]
        ]:

            If only_predict is True: Returns training_data_length, X_test, and scaler.
            If only_predict is False: Returns training_data_length, X_train, y_train, X_val, y_val and scaler.
    """

    close_prices = df[close_column].values
    dataset = close_prices.reshape(-1, 1)

    if only_predict:
        # Load scaler and Transform entire dataset
        scaler = load_scaler(scaler_path)
        scaled_dataset = scaler.transform(dataset)
        X_test, _ = create_sliding_window(scaled_dataset, window_size)

        return X_test, scaler

    else:
        training_data_len = int(np.ceil((1 - test_split) * len(close_prices)))

        # Initialize Scaler and Fit Transform entire dataset
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(dataset)

        scaled_train = scaled_data[:training_data_len]
        X_train, y_train = create_sliding_window(scaled_train, window_size)

        scaled_test = scaled_data[training_data_len - window_size:]
        X_val, y_val = create_sliding_window(scaled_test, window_size)

        return training_data_len, X_train, y_train, X_val, y_val, scaler
