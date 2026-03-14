import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(filepath):
    """Load dataset from CSV"""
    df = pd.read_csv(filepath)
    return df

def split_data(df, train_ratio=0.8):
    """Split data into train and test sets (80-20)"""
    split_idx = int(len(df) * train_ratio)
    train = df[:split_idx].reset_index(drop=True)
    test = df[split_idx:].reset_index(drop=True)
    return train, test

def scale_data(train, test, feature_range=(0, 1)):
    """Scale data using MinMaxScaler"""
    scaler = MinMaxScaler(feature_range=feature_range)
    
    # Fit on training data
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    
    return train_scaled, test_scaled, scaler

def create_sequences(data, target_index, window_size=30):
    """
    Create sequences for LSTM training
    
    Args:
        data: Scaled data array (T, F) where T=timesteps, F=features
        target_index: Index of target column (BOD)
        window_size: Number of timesteps to look back
    
    Returns:
        X: Input sequences (N, window_size, F)
        y: Target values (N,)
    """
    X, y = [], []
    
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, target_index])
    
    return np.array(X), np.array(y)

def inverse_transform(predictions, scaler, target_index):
    """
    Inverse transform predictions to original scale
    
    Args:
        predictions: Model predictions (N, 1)
        scaler: Fitted MinMaxScaler object
        target_index: Index of target column
    
    Returns:
        Original scale predictions
    """
    # Create a dummy array with same shape as original data
    dummy = np.zeros((predictions.shape[0], scaler.n_features_in_))
    dummy[:, target_index] = predictions.flatten()
    
    # Inverse transform
    result = scaler.inverse_transform(dummy)
    return result[:, target_index]

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
