import pandas as pd
import streamlit as st

import numpy as np

from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

import joblib


@st.cache_data
def load_csv(uploaded_file):
    df = pd.read_csv(uploaded_file)
    return df


@st.cache_resource
def build_model(X_train):
    model = Sequential(name="StockPriceLSTM")
    model.add(keras.Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.summary()
    model.compile(optimizer="adam", loss="mse", metrics=["root_mean_squared_error"])

    return model


def create_sliding_window(data, window_size):
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])

    return np.array(X), np.array(y).reshape(-1)


@st.cache_resource(show_spinner="Loading Scaler...")
def load_scaler(scaler_path):
    return joblib.load(scaler_path)


def prepare_train_test_datasets(df, close_column, test_split, window_size, scaler_path=None, only_predict=False):
    close_prices = df[close_column].values
    dataset = close_prices.reshape(-1, 1)

    training_data_len = int(np.ceil((1 - test_split) * len(close_prices)))

    if only_predict:
        # Load scaler and transform entire dataset
        scaler = load_scaler(scaler_path)
        scaled_dataset = scaler.transform(dataset)

        scaled_test = scaled_dataset[training_data_len - window_size:]
        X_test, _ = create_sliding_window(scaled_test, window_size)

        return training_data_len, X_test, scaler

    else:
        # Initialize Scaler and Fit Transform entire dataset
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(dataset)

        scaled_train = scaled_data[:training_data_len]
        X_train, y_train = create_sliding_window(scaled_train, window_size)

        scaled_test = scaled_data[training_data_len - window_size:]
        X_test, _ = create_sliding_window(scaled_test, window_size)

        return training_data_len, X_train, y_train, X_test, scaler
