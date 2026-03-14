import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from model_loader import model

def predict_future(data):
    # Assume data is DataFrame with Date and features
    if 'Date' in data.columns:
        data = data.set_index('Date')

    # Select numeric columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found.")

    df_num = data[numeric_cols].astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_num.values)

    # Assume time_steps=30, and predict next 7 days
    time_steps = 30
    if scaled.shape[0] < time_steps:
        raise ValueError("Not enough data for prediction.")

    last_seq = scaled[-time_steps:]
    current_sequence = last_seq.copy()

    future_predictions = []
    n_features = scaled.shape[1]

    for _ in range(7):
        x = current_sequence.reshape(1, time_steps, n_features)
        pred = model.predict(x, verbose=0)
        pred_val = float(np.array(pred).ravel()[0])

        # Inverse scale for BOD (assume first column is BOD)
        dummy = np.zeros((1, n_features))
        dummy[0, 0] = pred_val
        inv = scaler.inverse_transform(dummy)
        predicted_value = float(inv[0, 0])

        future_predictions.append(predicted_value)

        # Update sequence
        new_row = current_sequence[-1].copy()
        new_row[0] = pred_val
        current_sequence = np.vstack([current_sequence[1:], new_row])

    return future_predictions