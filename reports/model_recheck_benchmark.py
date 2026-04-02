import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Add,
    Attention,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    LSTM,
    LayerNormalization,
    MaxPooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from src.cnn_lstm_attention import build_cnn_lstm_attention_model
from src.preprocessing import create_sequences, scale_data, split_data


np.random.seed(42)
tf.random.set_seed(42)


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def inverse_target(values_scaled, scaler, target_index):
    dummy = np.zeros((values_scaled.shape[0], scaler.n_features_in_))
    dummy[:, target_index] = values_scaled.reshape(-1)
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_index]


def build_bilstm_residual_attention(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.15)(x)

    lstm1 = Bidirectional(LSTM(64, return_sequences=True))(x)
    lstm1 = LayerNormalization()(lstm1)
    lstm1 = Dropout(0.2)(lstm1)

    lstm2 = Bidirectional(LSTM(32, return_sequences=True))(lstm1)
    lstm2 = LayerNormalization()(lstm2)
    lstm2 = Dropout(0.2)(lstm2)

    # Simple residual merge between sequence blocks
    proj = Conv1D(lstm2.shape[-1], kernel_size=1, padding="same")(lstm1)
    merged = Add()([lstm2, proj])

    attn = Attention()([merged, merged])
    pooled = GlobalAveragePooling1D()(attn)

    dense = Dense(64, activation="relu")(pooled)
    dense = Dropout(0.25)(dense)
    dense = Dense(32, activation="relu")(dense)
    output = Dense(1, activation="linear")(dense)

    model = Model(inputs=inputs, outputs=output, name="BiLSTM_Residual_Attention")
    model.compile(optimizer=Adam(learning_rate=8e-4), loss="mse", metrics=["mae"])
    return model


def run_experiment(model_name, model_builder, X_train, y_train, X_test, y_test, scaler, target_index):
    model = model_builder((X_train.shape[1], X_train.shape[2]))

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=0),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=0,
    )

    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = inverse_target(y_pred_scaled, scaler, target_index)
    y_true = inverse_target(y_test.reshape(-1, 1), scaler, target_index)

    result = {
        "model": model_name,
        "epochs_ran": len(history.history["loss"]),
        "RMSE": rmse(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape(y_true, y_pred),
    }
    return result


def main():
    data_path = Path("data/Rajahmundry_STP_Daily_Synthetic_2020_2023.csv")
    df = pd.read_csv(data_path)
    target_col = "BOD (mg/L)"

    # Keep all numeric columns except Date to avoid the old 5-feature mismatch.
    df = df.drop(columns=["Date"])

    train_df, test_df = split_data(df, train_ratio=0.8)
    train_scaled, test_scaled, scaler = scale_data(train_df, test_df)

    target_index = list(df.columns).index(target_col)
    X_train, y_train = create_sequences(train_scaled, target_index, window_size=30)
    X_test, y_test = create_sequences(test_scaled, target_index, window_size=30)

    experiments = []
    experiments.append(
        run_experiment(
            "CNN_LSTM_Attention_Current",
            build_cnn_lstm_attention_model,
            X_train,
            y_train,
            X_test,
            y_test,
            scaler,
            target_index,
        )
    )
    experiments.append(
        run_experiment(
            "BiLSTM_Residual_Attention_Candidate",
            build_bilstm_residual_attention,
            X_train,
            y_train,
            X_test,
            y_test,
            scaler,
            target_index,
        )
    )

    best = sorted(experiments, key=lambda x: x["RMSE"])[0]
    output = {
        "target": target_col,
        "n_train_sequences": int(X_train.shape[0]),
        "n_test_sequences": int(X_test.shape[0]),
        "experiments": experiments,
        "best_by_rmse": best,
    }

    output_path = Path("reports/model_recheck_results.json")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
