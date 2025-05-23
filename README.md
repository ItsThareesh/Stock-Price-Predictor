# 🌡️ Temperature Prediction using Deep Learning (LSTM + Conv1D)

This repository demonstrates a deep learning model for short-term temperature forecasting using time-series data. Refer to `temperature-predictor-lstm-conv1d.ipynb` for more information on model training, metrics, graph plots and more.

A minimal Streamlit interface is included for running predictions on test datasets.

## Project Summary

-   **Objective**: Predict temperature using recent historical readings.
-   **Model**: Bidirectional LSTM + Conv1D hybrid architecture
-   **Performance**:
    -   R² ≈ 0.99
    -   RMSE ≈ 0.68
    -   MAE ≈ 0.48

## Model Architecture

### Hybrid Conv1D + Bidirectional LSTM

-   **Conv1D layer**
-   **Bidirectional LSTM layers:**
    Process input sequences forwards and backwards, improving sensitivity to complex sequential dynamics in weather data.
-   **Regularization:**
    Normalizing activations helps avoid vanishing/exploding gradients and allowing higher learning rates.
-   **Optimization:**
    Employed **LR_Scheduler** and **Early Stopping** to dynamically adjust learning rate for faster convergence and halt training once validation loss stagnates, avoiding overfitting.

## Core Features

-   Time-based sliding windows to capture recent trends, feeding the model sufficient data for accurate temperature prediction.
-   Cyclical encoding for temporal features (Day, Month)
-   Learning rate scheduling and early stopping
-   Simple Streamlit demo for running inferences on custom inputs, i.e. window-size and steps (for plotting predicted output)

## Run Locally (Without Docker)

### 1. Clone the Repository

```bash
git clone https://github.com/ItsThareesh/Temperature-Predictor-LSTM.git
cd Temperature-Predictor-LSTM
```

### 2. Create Virtual Environment

```bash
python -m virtualenv venv
```

##### For venv or virtualenv on Linux/Mac:

```bash
source venv/bin/activate
```

##### For venv or virtualenv on Windows (Command Prompt):

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit web app

```bash
streamlit run app/app.py
```

## Run with Docker

### 1. Clone the Repository

```bash
git clone https://github.com/ItsThareesh/Temperature-Predictor-LSTM.git
cd Temperature-Predictor-LSTM
```

### 2. Build Docker Image

```bash
docker build -t temperature-predictor-lstm .
```

### 3. Run the Docker Container

```bash
docker run -p 8501:8501 temperature-predictor-lstm
```

### 4. Access the App

Open your browser and go to: `localhost:8501`

## Training your own model

If you wish to experiment more and train the model on your own, I'd strongly recommend you to use Kaggle.

[Click here to open the Kaggle Notebook and start training!](https://www.kaggle.com/code/thareeshprabakaran/temperature-predictor-lstm-conv1d)

## Future Improvements

-   **Experiment with Advanced Transformer-based Architectures**
-   **Allow users to fine-tune the model on their own datasets**

## License

This project is licensed under the MIT License.
