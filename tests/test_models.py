import pytest
from src.models import build_transformer, build_lstm


def test_transformer_build():
    # Test Transformer model building
    model = build_transformer(
        window_size=5, d_model=64, num_heads=8, dff=256, num_layers=4
    )
    assert model is not None, "Transformer model should be built successfully"
    assert model.input_shape == (None, 5, 1), (
        "Transformer input shape should match (window_size, 1)"
    )


def test_lstm_build():
    # Test LSTM model building
    model = build_lstm(window_size=5)
    assert model is not None, "LSTM model should be built successfully"
    assert model.input_shape == (None, 5, 1), (
        "LSTM input shape should match (window_size, 1)"
    )
