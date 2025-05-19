import streamlit as st


def custom_progress_bar(epochs, batch_size, X_train, y_train, model):
    progress_bar = st.progress(0, text="⏳ Training in progress...")

    for epoch in range(epochs):
        model.fit(X_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
        progress = int(((epoch + 1) / epochs) * 100)
        progress_bar.progress(progress, text=f"Epoch {epoch + 1}/{epochs}")

    return model


def show_graph(df, plt, date_column, close_column, training_data_len):
    predictions = st.session_state["PREDICTIONS"]
    test_df = df[training_data_len:]

    st.write("### 📊 Prediction vs Actual")
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(test_df[date_column], test_df[close_column], label="Actual", color="orange")
    ax1.plot(test_df[date_column], predictions.flatten(), label="Predicted", color="red")
    ax1.set_title("Stock Price Prediction")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price")
    ax1.legend()
    st.pyplot(fig1)


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

    if not model_file:
        epochs = st.sidebar.slider("Epochs", min_value=5, max_value=100, value=25, step=5)
        batch_size = st.sidebar.selectbox("Batch Size", options=[16, 32, 64, 128], index=1)

    return date_column, close_column, test_split, window_size, epochs, batch_size
