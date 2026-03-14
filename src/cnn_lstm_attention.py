import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, BatchNormalization,
    LSTM, Dense, Dropout, Attention, GlobalAveragePooling1D,
    Concatenate, Reshape
)

def build_cnn_lstm_attention_model(input_shape):
    """
    Build CNN + LSTM + Attention hybrid model for time series prediction
    
    Architecture:
    - 1D CNN: Extract short-term temporal patterns
    - LSTM: Capture long-term dependencies
    - Attention: Focus on important timesteps
    - Dense layers: Make final prediction
    
    Args:
        input_shape: Tuple (window_size, n_features)
    
    Returns:
        Compiled Keras model
    """
    
    inputs = Input(shape=input_shape, name='input_layer')
    
    # ========== CNN Branch ==========
    cnn = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    cnn = BatchNormalization()(cnn)
    cnn = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(cnn)
    cnn = MaxPooling1D(pool_size=2, padding='same')(cnn)
    cnn = Dropout(0.2)(cnn)
    
    # ========== LSTM Branch ==========
    lstm = LSTM(64, return_sequences=True, activation='relu')(inputs)
    lstm = Dropout(0.2)(lstm)
    lstm = LSTM(32, return_sequences=True, activation='relu')(lstm)
    lstm = Dropout(0.2)(lstm)
    
    # ========== Attention Layer ==========
    # Apply attention on LSTM outputs
    attention = Attention()([lstm, lstm])
    attention = GlobalAveragePooling1D()(attention)
    
    # ========== Merge CNN and Attention ==========
    merged = Concatenate()([cnn[:, -1, :], attention])
    
    # ========== Dense Layers ==========
    x = Dense(64, activation='relu')(merged)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    x = Dense(16, activation='relu')(x)
    
    # Output layer
    outputs = Dense(1, activation='linear')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name='CNN_LSTM_Attention')
    
    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def build_simple_lstm_model(input_shape):
    """Baseline LSTM model for comparison"""
    model = tf.keras.Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model
