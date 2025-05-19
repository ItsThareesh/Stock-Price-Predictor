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
