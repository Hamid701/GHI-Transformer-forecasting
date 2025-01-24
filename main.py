from pathlib import Path
import numpy as np
import joblib
import tensorflow as tf
from src.data_loader import load_and_process_data
from src.models import build_transformer, build_lstm
from src.evaluate import print_metrics, save_metrics
from src.visualize import save_plot

# Configuration
SETTINGS = {
    "data_path": "data/GHI_data.csv",
    "window_size": 5,
    "test_size": 72,
    "batch_size": 64,
    "epochs": 50,
}

# Best hyperparameters for Transformer (from tuning)
BEST_TRANSFORMER_PARAMS = {
    "d_model": 98,
    "num_heads": 7,
    "dff": 199,
    "num_layers": 4,
    "dropout_rate": 0.2935,
}

# Best hyperparameters for LSTM (from tuning)
BEST_LSTM_PARAMS = {
    "units": 128,
    "dropout_rate": 0.1702,
    "learning_rate": 0.0164,
}


def save_processed_data(data, save_dir, scaler):
    """Save processed data and scaler to disk"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

    # Save data
    np.save(save_dir / "X_train.npy", data[0][0])
    np.save(save_dir / "y_train.npy", data[0][1])
    np.save(save_dir / "X_test.npy", data[1][0])
    np.save(save_dir / "y_test.npy", data[1][1])

    # Save scaler
    joblib.dump(scaler, save_dir / "scaler.pkl")
    print(f"Processed data saved to: {save_dir}")


def main():
    # Load and process data
    (X_train, y_train), (X_test, y_test), scaler = load_and_process_data(
        SETTINGS["data_path"], SETTINGS["window_size"], SETTINGS["test_size"]
    )

    # Save processed data
    save_processed_data(
        ((X_train, y_train), (X_test, y_test)),
        "data/processed",  # Save directory
        scaler,
    )

    # Build and train Transformer with best hyperparameters
    transformer = build_transformer(
        window_size=SETTINGS["window_size"],
        **BEST_TRANSFORMER_PARAMS,
    )
    print("\nTraining Transformer with best parameters...")
    transformer.fit(
        X_train.reshape(-1, SETTINGS["window_size"], 1),
        y_train,
        batch_size=SETTINGS["batch_size"],
        epochs=SETTINGS["epochs"],
        validation_split=0.1,
        verbose=1,
    )
    transformer.save("models/transformer_model")

    # Build and train LSTM with best hyperparameters
    lstm = build_lstm(
        window_size=SETTINGS["window_size"],
        **BEST_LSTM_PARAMS,
    )
    print("\nTraining LSTM with best parameters...")
    lstm.fit(
        X_train.reshape(-1, SETTINGS["window_size"], 1),
        y_train,
        batch_size=SETTINGS["batch_size"],
        epochs=SETTINGS["epochs"],
        validation_split=0.1,
        verbose=1,
    )
    lstm.save("models/lstm_model")

    print("\nTraining completed and models saved!")

    # Evaluate models
    print("\nEvaluating models on test set...")

    # Generate predictions
    transformer_pred = scaler.inverse_transform(transformer.predict(X_test))
    lstm_pred = scaler.inverse_transform(
        lstm.predict(X_test.reshape(-1, SETTINGS["window_size"], 1))
    )
    y_true = scaler.inverse_transform(y_test)

    # Print metrics to console
    print_metrics(y_true, transformer_pred, lstm_pred)

    # Save metrics to file
    save_metrics(y_true, transformer_pred, lstm_pred, "results/metrics.txt")

    # Save plot
    save_plot(y_true, transformer_pred, lstm_pred, "visualization/forecast_plot.html")


if __name__ == "__main__":
    main()
