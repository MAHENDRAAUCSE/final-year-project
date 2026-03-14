import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_wwtp_data(start_date='2020-01-01', end_date='2023-12-31', output_path=None):
    """
    Generate synthetic WWTP data for 2020-2023
    with realistic patterns for BOD, COD, TSS, TN, TP, pH, DO
    """
    
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_samples = len(dates)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Base values and patterns
    bod_base = 150
    cod_base = 300
    tss_base = 200
    tn_base = 40
    tp_base = 5
    ph_base = 7.2
    do_base = 4.8
    
    # Create synthetic data with seasonal patterns
    day_of_year = np.array([d.dayofyear for d in dates])
    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
    
    # Add trend
    trend = np.linspace(0, 1, n_samples)
    
    # Generate parameters with noise and seasonal patterns
    bod = bod_base * seasonal_factor * (1 + 0.2 * trend) + np.random.normal(0, 15, n_samples)
    cod = cod_base * seasonal_factor * (1 + 0.15 * trend) + np.random.normal(0, 30, n_samples)
    tss = tss_base * seasonal_factor * (1 + 0.1 * trend) + np.random.normal(0, 25, n_samples)
    tn = tn_base * seasonal_factor + np.random.normal(0, 5, n_samples)
    tp = tp_base * seasonal_factor + np.random.normal(0, 0.5, n_samples)
    ph = ph_base + 0.15 * np.sin(2 * np.pi * day_of_year / 365 + np.pi / 4) + np.random.normal(0, 0.08, n_samples)
    do = do_base - 0.9 * np.sin(2 * np.pi * day_of_year / 365) - 0.002 * (cod - cod_base) + np.random.normal(0, 0.35, n_samples)
    
    # Add autocorrelation (values depend on previous day)
    for i in range(1, n_samples):
        bod[i] = 0.7 * bod[i] + 0.3 * bod[i-1]
        cod[i] = 0.7 * cod[i] + 0.3 * cod[i-1]
        tss[i] = 0.7 * tss[i] + 0.3 * tss[i-1]
        tn[i] = 0.7 * tn[i] + 0.3 * tn[i-1]
        tp[i] = 0.7 * tp[i] + 0.3 * tp[i-1]
        ph[i] = 0.85 * ph[i] + 0.15 * ph[i-1]
        do[i] = 0.75 * do[i] + 0.25 * do[i-1]
    
    # Ensure all values are positive
    bod = np.maximum(bod, 10)
    cod = np.maximum(cod, 20)
    tss = np.maximum(tss, 15)
    tn = np.maximum(tn, 5)
    tp = np.maximum(tp, 0.5)
    ph = np.clip(ph, 6.5, 8.5)
    do = np.clip(do, 1.0, 8.5)
    
    # Create DataFrame with correct columns
    df = pd.DataFrame({
        'Date': dates,
        'BOD (mg/L)': np.round(bod, 2),
        'COD (mg/L)': np.round(cod, 2),
        'TSS (mg/L)': np.round(tss, 2),
        'TN (mg/L)': np.round(tn, 2),
        'TP (mg/L)': np.round(tp, 2),
        'PH': np.round(ph, 2),
        'DO (mg/L)': np.round(do, 2)
    })
    
    # Save if output path provided
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")
        print(f"Shape: {df.shape}")
        print(f"\nFirst few rows:\n{df.head()}")
        print(f"\nLast few rows:\n{df.tail()}")
        print(f"\nStatistics:\n{df.describe()}")
    
    return df

if __name__ == "__main__":
    # Generate data
    df = generate_synthetic_wwtp_data(
        start_date='2020-01-01',
        end_date='2023-12-31',
        output_path='data/Rajahmundry_STP_Daily_Synthetic_2020_2023.csv'
    )
