import pytest
from src.data_loader import load_and_process_data


def test_data_loading():
    # Test data loading with a small dataset
    data_path = "data/GHI_data.csv"
    window_size = 5
    test_size = 72

    # Load data
    (X_train, y_train), (X_test, y_test), scaler = load_and_process_data(
        data_path, window_size, test_size
    )

    # Check shapes
    assert X_train.shape[1] == window_size, (
        "X_train should have the correct window size"
    )
    assert len(y_train) == len(X_train), "y_train should match X_train length"
    assert len(y_test) == test_size, "y_test should match test_size"

    # Check scaler
    assert scaler is not None, "Scaler should be initialized"
