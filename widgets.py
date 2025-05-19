import streamlit as st


def custom_progress_bar(st, epochs, batch_size, X_train, y_train, model):
    progress_bar = st.progress(0, text="⏳ Training in progress...")
    history_all = {'loss': [], 'root_mean_squared_error': []}

    for epoch in range(epochs):
        history = model.fit(X_train, y_train, epochs=1, batch_size=batch_size)

        epoch_loss = history.history['loss'][0]
        rmse_loss = history.history['root_mean_squared_error'][0]

        history_all['loss'].append(epoch_loss)
        history_all['root_mean_squared_error'].append(rmse_loss)

        progress = int(((epoch + 1) / epochs) * 100)
        progress_bar.progress(progress, text=f"Epoch {epoch + 1}/{epochs}")

    st.session_state['HISTORY'] = history_all

    return model


def generate_prediction_graph(df, plt, date_column, close_column, training_data_len):
    predictions = st.session_state["PREDICTIONS"]
    test_df = df[training_data_len:]

    st.write("### 📊 Prediction vs Actual")

    plt.figure(figsize=(12, 6))
    plt.plot(test_df[date_column], test_df[close_column], label="Test (Actual)", color='orange')
    plt.plot(test_df[date_column], predictions, label="Predictions", color='red', alpha=0.65)
    plt.title("Stock Predictions - Zoomed Test Set")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()

    st.pyplot(plt)


def generate_history_graph(history, plt):
    metrics = list(history.keys())
    num_metrics = len(metrics)

    plt.figure(figsize=(8*num_metrics, 6))

    for idx, metric in enumerate(metrics, 1):
        plt.subplot(1, num_metrics, idx)
        plt.plot(history[metric], label=metric)
        plt.title(f"{metric} over Epochs")
        plt.xlabel('Epochs')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)

    plt.tight_layout()

    st.pyplot(plt)


def sidebar(model_file, df):
    st.sidebar.markdown("## Model Parameters")

    df_cols = df.columns.tolist()

    date_column = st.sidebar.selectbox("Select the Date column", df_cols, index=None, key='date_column',
                                       on_change=lambda: st.session_state.update({'DATE_UPDATED': True}))
    close_column = st.sidebar.selectbox("Select the Close Price column", df_cols, index=None, key='close_column',
                                        on_change=lambda: st.session_state.update({'CLOSE_UPDATED': True}))

    test_split = st.sidebar.slider("Test Data Fraction", 0.05, 0.5, 0.1, step=0.05)
    window_size = st.sidebar.slider("Window Size", min_value=10, max_value=200, value=60, step=5)

    epochs = None
    batch_size = None
    show_graph = None

    if not model_file:
        epochs = st.sidebar.slider("Epochs", min_value=5, max_value=100, value=25, step=5)
        batch_size = st.sidebar.selectbox("Batch Size", options=[16, 32, 64, 128], index=1)
        show_graph = st.sidebar.checkbox("Show Graph", value=True)

    return date_column, close_column, test_split, window_size, epochs, batch_size, show_graph
