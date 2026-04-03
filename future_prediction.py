import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.model_compat import load_compatible_model


def find_target_column(df, target_name):
    if target_name in df.columns:
        return target_name
    target_lower = target_name.strip().lower()
    for c in df.columns:
        col_lower = c.strip().lower()
        if col_lower == target_lower:
            return c
        if target_lower in col_lower or col_lower in target_lower:
            return c
    normalized_aliases = {
        "bod": ["bod"],
        "cod": ["cod"],
        "ph": ["ph", "ph value", "power of hydrogen"],
        "do": ["do", "dissolved oxygen"],
    }
    for col in df.columns:
        col_lower = col.lower()
        for canonical, aliases in normalized_aliases.items():
            if target_lower in aliases and canonical in col_lower:
                return col
    raise ValueError(f"Target column '{target_name}' not found in dataset columns: {list(df.columns)}")


def load_data(data_path):
    # resolve paths relative to script location when given a relative path
    if not os.path.isabs(data_path):
        base = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base, data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    df = pd.read_csv(data_path, parse_dates=["Date"], dayfirst=False)
    if "Date" in df.columns:
        df.set_index("Date", inplace=True)
    return df


def prepare_scaler_and_data(df):
    # use numeric columns only for scaling
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in the dataset to scale.")
    df_num = df[numeric_cols].astype(float)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_num.values)
    return scaler, scaled, df_num, numeric_cols


def predict_future(model, scaler, scaled_data, target_index, future_days, time_steps):
    n_features = scaled_data.shape[1]
    if scaled_data.shape[0] < time_steps:
        raise ValueError("Not enough history to form the required time window.")

    last_seq = scaled_data[-time_steps:].copy()
    current_sequence = last_seq.copy()

    future_predictions = []

    for _ in range(future_days):
        x = current_sequence.reshape(1, time_steps, n_features)
        pred = model.predict(x, verbose=0)
        # convert to scalar
        pred_val = float(np.array(pred).ravel()[0])

        # inverse scale only for the target column by creating dummy row
        dummy = np.zeros((1, n_features), dtype=float)
        dummy[0, target_index] = pred_val
        inv = scaler.inverse_transform(dummy)
        predicted_value = float(inv[0, target_index])

        future_predictions.append((pred_val, predicted_value))

        # update sequence: drop first row, append new row copying last row and replacing target
        new_row = current_sequence[-1].copy()
        new_row[target_index] = pred_val
        current_sequence = np.vstack([current_sequence[1:], new_row])

    return future_predictions


def save_results(output_path, future_dates, future_predictions, target_name):
    # future_predictions: list of tuples (scaled_pred, inv_pred)
    preds_scaled = [p[0] for p in future_predictions]
    preds = [p[1] for p in future_predictions]

    df_out = pd.DataFrame({
        "Date": future_dates,
        f"Predicted {target_name} (raw)": np.round(preds, 3),
        f"Model Output (scaled)": np.round(preds_scaled, 5),
    })

    # summary
    summary = {
        "Last Prediction Date": [future_dates[-1].strftime("%Y-%m-%d")],
        "Days Predicted": [len(future_dates)],
        "Mean Predicted": [float(np.mean(preds))],
        "Min Predicted": [float(np.min(preds))],
        "Max Predicted": [float(np.max(preds))],
    }
    df_summary = pd.DataFrame(summary)

    # attempt to write output, removing previous file if locked
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
    except Exception:
        # ignore removal errors; the ExcelWriter may still raise
        pass
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_out.to_excel(writer, sheet_name="Predictions", index=False)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
    except PermissionError:
        # output file probably open in another app; try alternate name
        alt = output_path.replace(".xlsx", "_new.xlsx")
        with pd.ExcelWriter(alt, engine="openpyxl") as writer:
            df_out.to_excel(writer, sheet_name="Predictions", index=False)
            df_summary.to_excel(writer, sheet_name="Summary", index=False)
        output_path = alt

    return df_out, df_summary


def main():
    parser = argparse.ArgumentParser(description="Predict future water-quality values using a trained LSTM model")
    parser.add_argument("--data", default="data/Rajahmundry_STP_Daily_Synthetic_2020_2023.csv", help="Path to CSV data file")
    parser.add_argument("--model", default=None, help="Path to trained Keras model (.h5). Defaults to a target-specific model path.")
    parser.add_argument("--days", type=int, default=7, help="Number of future days to predict")
    parser.add_argument("--time_steps", type=int, default=30, help="History window (days) to use for prediction")
    parser.add_argument("--target", default="BOD (mg/L)", help="Target column name, for example 'COD (mg/L)', 'PH', or 'DO (mg/L)'")
    parser.add_argument("--output", default=None, help="Excel output path. Defaults to a target-specific file name.")

    args = parser.parse_args()

    df = load_data(args.data)

    target_col = find_target_column(df, args.target)
    target_slug = (
        target_col.lower()
        .replace("(mg/l)", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(" ", "_")
        .strip("_")
    )

    scaler, scaled, df_num, numeric_cols = prepare_scaler_and_data(df)

    # determine target index within numeric columns
    if target_col not in numeric_cols:
        raise ValueError(f"Target column '{target_col}' is not numeric.")
    target_index = numeric_cols.index(target_col)

    # ensure model path is absolute or relative to script
    base = os.path.dirname(os.path.abspath(__file__))
    if args.model is None:
        keras_path = os.path.join(base, "models", f"{target_slug}_cnn_lstm_attention_model.keras")
        h5_path = os.path.join(base, "models", f"{target_slug}_cnn_lstm_attention_model.h5")
        args.model = keras_path if os.path.exists(keras_path) else h5_path
    elif not os.path.isabs(args.model):
        args.model = os.path.join(base, args.model)
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    if args.output is None:
        args.output = os.path.join(base, f"Future_{target_slug.upper()}_Predictions.xlsx")

    model = load_compatible_model(args.model)

    future_days = args.days
    time_steps = args.time_steps

    preds = predict_future(model, scaler, scaled, target_index, future_days, time_steps)

    # future dates
    if isinstance(df.index, pd.DatetimeIndex):
        last_date = df.index[-1]
    else:
        # fallback: assume last row's name can be parsed
        last_date = pd.to_datetime(df.index[-1])
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

    df_out, df_summary = save_results(args.output, future_dates, preds, target_col)

    print(f"Saved {target_col} predictions to {args.output}")
    print(df_out)
    print('\nSummary:')
    print(df_summary.to_string(index=False))


if __name__ == "__main__":
    main()
