import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


def load_and_process_data(data_path, window_size=5, test_size=72):
    """Load and process GHI data"""
    # Load data
    data = pd.read_csv(Path(__file__).parent.parent / data_path)

    # Process dates
    data["Date"] = data["Date"].apply(lambda x: x.split("/")[0])
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    # Normalize
    scaler = MinMaxScaler()
    ghi_scaled = scaler.fit_transform(data["GHI"].values.reshape(-1, 1))

    # Create sequences
    X, y = [], []
    for i in range(len(ghi_scaled) - window_size):
        X.append(ghi_scaled[i : i + window_size])
        y.append(ghi_scaled[i + window_size])

    # Split data
    split_idx = -test_size
    X_train, X_test = np.array(X[:split_idx]), np.array(X[split_idx:])
    y_train, y_test = np.array(y[:split_idx]), np.array(y[split_idx:])

    return (X_train, y_train), (X_test, y_test), scaler
