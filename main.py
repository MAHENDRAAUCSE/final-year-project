import argparse
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from src.cnn_lstm_attention import build_cnn_lstm_attention_model
from src.preprocessing import (
    calculate_mape,
    calculate_rmse,
    create_sequences,
    inverse_transform,
    load_data,
    scale_data,
    split_data,
)


np.random.seed(42)
tf.random.set_seed(42)


CONFIG = {
    "window_size": 30,
    "target_column": "BOD (mg/L)",
    "train_ratio": 0.8,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 15,
    "reduce_lr_patience": 8,
}


def make_target_slug(target_column):
    return (
        target_column.lower()
        .replace("(mg/l)", "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
        .replace(" ", "_")
        .strip("_")
    )


def build_output_paths(target_column):
    slug = make_target_slug(target_column)
    return {
        "metrics": f"reports/{slug}_metrics.json",
        "report": f"reports/{slug}_detailed_report.txt",
        "model_keras": f"models/{slug}_cnn_lstm_attention_model.keras",
        "model": f"models/{slug}_cnn_lstm_attention_model.h5",
        "weights": f"models/{slug}_cnn_lstm_attention_model.weights.h5",
        "plot_history": f"plots/{slug}_01_training_history.png",
        "plot_predictions": f"plots/{slug}_02_predictions_vs_actual.png",
        "plot_residuals": f"plots/{slug}_03_residuals.png",
        "plot_metrics": f"plots/{slug}_04_metrics_summary.png",
        "plot_scatter": f"plots/{slug}_05_actual_vs_predicted_scatter.png",
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN-LSTM-attention model for a selected water-quality target"
    )
    parser.add_argument(
        "--target",
        default=CONFIG["target_column"],
        help="Target column name, for example 'COD (mg/L)', 'PH', or 'DO (mg/L)'",
    )
    parser.add_argument(
        "--data",
        default="data/Rajahmundry_STP_Daily_Synthetic_2020_2023.csv",
        help="Input dataset CSV path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    CONFIG["target_column"] = args.target
    output_paths = build_output_paths(CONFIG["target_column"])

    print("=" * 70)
    print(f"CNN + LSTM + ATTENTION MODEL FOR {CONFIG['target_column']} PREDICTION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    print("STEP 1: Loading data...")
    if not os.path.exists(args.data):
        print(f"[!] Dataset not found: {args.data}")
        return

    df = load_data(args.data)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print()

    df = df.drop("Date", axis=1)
    if CONFIG["target_column"] not in df.columns:
        raise ValueError(f"Target column '{CONFIG['target_column']}' not found in dataset columns: {list(df.columns)}")

    print("STEP 2: Splitting data (80-20)...")
    train, test = split_data(df, train_ratio=CONFIG["train_ratio"])
    print(f"Train samples: {len(train)}")
    print(f"Test samples: {len(test)}")
    print()

    print("STEP 3: Scaling data (MinMax)...")
    train_scaled, test_scaled, scaler = scale_data(train, test)
    target_index = list(df.columns).index(CONFIG["target_column"])
    print(f"Target column index: {target_index} ({CONFIG['target_column']})")
    print()

    print("STEP 4: Creating sequences...")
    X_train, y_train = create_sequences(train_scaled, target_index, CONFIG["window_size"])
    X_test, y_test = create_sequences(test_scaled, target_index, CONFIG["window_size"])
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print()

    print("STEP 5: Building CNN-LSTM-Attention model...")
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_cnn_lstm_attention_model(input_shape)
    print(model.summary())
    print()

    print("STEP 6: Setting up callbacks...")
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=CONFIG["early_stopping_patience"],
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=CONFIG["reduce_lr_patience"],
        min_lr=1e-7,
        verbose=1,
    )
    print()

    print("STEP 7: Training model...")
    history = model.fit(
        X_train,
        y_train,
        epochs=CONFIG["epochs"],
        batch_size=CONFIG["batch_size"],
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    print("Training completed")
    print()

    print("STEP 8: Making predictions...")
    y_pred_scaled = model.predict(X_test)
    y_pred = inverse_transform(y_pred_scaled, scaler, target_index)
    y_test_original = inverse_transform(y_test.reshape(-1, 1), scaler, target_index)
    print(f"Predictions shape: {y_pred.shape}")
    print()

    print("STEP 9: Calculating metrics...")
    rmse = calculate_rmse(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    mape = calculate_mape(y_test_original, y_pred)
    metrics = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2 Score": float(r2),
        "MAPE (%)": float(mape),
    }
    print(metrics)
    print()

    print("STEP 10: Saving reports...")
    with open(output_paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    report_text = f"""
CNN + LSTM + ATTENTION MODEL - PERFORMANCE REPORT
============================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CONFIGURATION
============================================================
Window Size: {CONFIG['window_size']} days
Target Column: {CONFIG['target_column']}
Train-Test Split: {CONFIG['train_ratio']*100}% - {(1-CONFIG['train_ratio'])*100}%
Batch Size: {CONFIG['batch_size']}
Epochs: {CONFIG['epochs']}
Early Stopping Patience: {CONFIG['early_stopping_patience']}

PERFORMANCE METRICS
============================================================
RMSE: {rmse:.4f}
MAE: {mae:.4f}
R2 Score: {r2:.4f}
MAPE (%): {mape:.4f}

DATA SUMMARY
============================================================
Total Samples: {len(df)}
Train Samples: {len(train)}
Test Samples: {len(test)}
Sequences Generated: {len(X_test)}
"""
    with open(output_paths["report"], "w", encoding="utf-8") as f:
        f.write(report_text.strip() + "\n")
    print(f"Metrics saved to {output_paths['metrics']}")
    print(f"Detailed report saved to {output_paths['report']}")
    print()

    print("STEP 11: Saving model...")
    model.save(output_paths["model_keras"])
    model.save(output_paths["model"])
    model.save_weights(output_paths["weights"])
    print(f"Portable model saved to {output_paths['model_keras']}")
    print(f"Model saved to {output_paths['model']}")
    print(f"Weights saved to {output_paths['weights']}")
    print()

    print("STEP 12: Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[0].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Model Loss During Training")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["mae"], label="Train MAE", linewidth=2)
    axes[1].plot(history.history["val_mae"], label="Val MAE", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("Model MAE During Training")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_paths["plot_history"], dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(y_test_original, label=f"Actual {CONFIG['target_column']}", linewidth=2, alpha=0.7)
    ax.plot(y_pred, label=f"Predicted {CONFIG['target_column']}", linewidth=2, alpha=0.7)
    ax.fill_between(range(len(y_test_original)), y_test_original, y_pred, alpha=0.2)
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel(CONFIG["target_column"])
    ax.set_title(f"{CONFIG['target_column']} Predictions vs Actual Values")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_paths["plot_predictions"], dpi=300, bbox_inches="tight")
    plt.close()

    residuals = y_test_original - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.6, s=50)
    axes[0].axhline(y=0, color="r", linestyle="--", linewidth=2)
    axes[0].set_xlabel(f"Predicted {CONFIG['target_column']}")
    axes[0].set_ylabel("Residuals")
    axes[0].set_title("Residual Plot")
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(residuals, bins=30, edgecolor="black", alpha=0.7)
    axes[1].set_xlabel("Residuals")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution")
    axes[1].grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_paths["plot_residuals"], dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    metric_names = ["RMSE", "MAE", "R2", "MAPE"]
    metric_values = [rmse, mae, r2, mape]
    ax.bar(metric_names, metric_values, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"], edgecolor="black", linewidth=2)
    for bar, val in zip(ax.patches, metric_values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.4f}", ha="center", va="bottom")
    ax.set_ylabel("Value")
    ax.set_title("Performance Metrics Summary")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_paths["plot_metrics"], dpi=300, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test_original, y_pred, alpha=0.6, s=50, edgecolors="black")
    min_val = min(y_test_original.min(), y_pred.min())
    max_val = max(y_test_original.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel(f"Actual {CONFIG['target_column']}")
    ax.set_ylabel(f"Predicted {CONFIG['target_column']}")
    ax.set_title(f"Actual vs Predicted (R2 = {r2:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(output_paths["plot_scatter"], dpi=300, bbox_inches="tight")
    plt.close()

    print("Training complete")


if __name__ == "__main__":
    main()
