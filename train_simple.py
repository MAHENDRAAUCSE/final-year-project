import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib before importing pyplot
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import json
from datetime import datetime

from src.preprocessing import *
from src.cnn_lstm_attention import build_cnn_lstm_attention_model

print("="*70)
print("CNN + LSTM + ATTENTION MODEL - SIMPLE TRAINING")
print("="*70)

try:
    # Load data
    print("\n[1/7] Loading data...")
    df = load_data("data/Rajahmundry_STP_Daily_Synthetic_2020_2023.csv")
    df = df.drop('Date', axis=1)
    print(f"    Shape: {df.shape}, Columns: {list(df.columns)}")
    
    # Split
    print("\n[2/7] Splitting data (80-20)...")
    train, test = split_data(df, train_ratio=0.8)
    print(f"    Train: {len(train)}, Test: {len(test)}")
    
    # Scale
    print("\n[3/7] Scaling data...")
    train_scaled, test_scaled, scaler = scale_data(train, test)
    target_index = 0  # BOD is first column
    
    # Sequences
    print("\n[4/7] Creating sequences (window=30)...")
    X_train, y_train = create_sequences(train_scaled, target_index, window_size=30)
    X_test, y_test = create_sequences(test_scaled, target_index, window_size=30)
    print(f"    X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    # Build model
    print("\n[5/7] Building model...")
    model = build_cnn_lstm_attention_model((X_train.shape[1], X_train.shape[2]))
    print(f"    Total params: {model.count_params()}")
    
    # Train
    print("\n[6/7] Training (max 100 epochs)...")
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=0)
    
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Predictions
    print("\n[7/7] Evaluating...")
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = inverse_transform(y_pred_scaled, scaler, target_index)
    y_test_original = inverse_transform(y_test.reshape(-1, 1), scaler, target_index)
    
    # Metrics
    rmse = calculate_rmse(y_test_original, y_pred)
    mae = mean_absolute_error(y_test_original, y_pred)
    r2 = r2_score(y_test_original, y_pred)
    mape = calculate_mape(y_test_original, y_pred)
    
    print("\n" + "="*70)
    print("RESULTS:")
    print(f"  RMSE: {rmse:.4f} mg/L")
    print(f"  MAE:  {mae:.4f} mg/L")
    print(f"  R2:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print("="*70 + "\n")
    
    # Save metrics
    metrics = {'RMSE': float(rmse), 'MAE': float(mae), 'R2_Score': float(r2), 'MAPE_%': float(mape)}
    with open('reports/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✓ Metrics saved")
    
    # Save model
    model.save('models/cnn_lstm_attention_model.h5')
    print("✓ Model saved")
    
    # Plot 1: Training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Val', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training History - Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train', linewidth=2)
    plt.plot(history.history['val_mae'], label='Val', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training History - MAE')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/01_training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Plot 1: Training history")
    
    # Plot 2: Predictions
    plt.figure(figsize=(14, 5))
    plt.plot(y_test_original, label='Actual', linewidth=2, marker='o', markersize=3, alpha=0.7)
    plt.plot(y_pred, label='Predicted', linewidth=2, marker='s', markersize=3, alpha=0.7)
    plt.xlabel('Sample')
    plt.ylabel('BOD (mg/L)')
    plt.title('BOD Predictions vs Actual')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/02_predictions_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Plot 2: Predictions")
    
    # Plot 3: Scatter
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test_original, y_pred, alpha=0.6, s=50)
    min_val = min(y_test_original.min(), y_pred.min())
    max_val = max(y_test_original.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    plt.xlabel('Actual BOD (mg/L)')
    plt.ylabel('Predicted BOD (mg/L)')
    plt.title(f'Actual vs Predicted (R² = {r2:.4f})')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/03_scatter_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Plot 3: Scatter plot")
    
    # Plot 4: Metrics bar
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics_names = ['RMSE', 'MAE', 'R2', 'MAPE']
    metrics_values = [rmse, mae, r2, mape]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    ax.bar(metrics_names, metrics_values, color=colors, edgecolor='black', linewidth=1.5)
    for i, (name, val) in enumerate(zip(metrics_names, metrics_values)):
        ax.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    ax.set_ylabel('Value')
    ax.set_title('Performance Metrics')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('plots/04_metrics_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Plot 4: Metrics bar")
    
    print("\n✓ All outputs saved successfully!")
    print(f"  - Reports: {os.listdir('reports')}")
    print(f"  - Plots: {os.listdir('plots')}")
    print(f"  - Models: {os.listdir('models')}")

except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
