from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    CSVLogger,
    TensorBoard,
)
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


def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = ["models", "data/processed", "results", "visualization", "logs"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Checked/Created directory: {directory}")


def create_callbacks(model_name, timestamp):
    """Create callbacks for training monitoring"""
    return [
        ModelCheckpoint(
            f"models/{model_name}_best_{timestamp}.h5",
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        ),
        CSVLogger(
            f"logs/{model_name}_training_{timestamp}.csv", separator=",", append=False
        ),
        TensorBoard(
            log_dir=f"logs/{model_name}_{timestamp}",
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
        ),
    ]


def save_processed_data(data, save_dir, scaler):
    """Save processed data and scaler to disk"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    np.save(save_dir / "X_train.npy", data[0][0])
    np.save(save_dir / "y_train.npy", data[0][1])
    np.save(save_dir / "X_test.npy", data[1][0])
    np.save(save_dir / "y_test.npy", data[1][1])

    # Save scaler
    joblib.dump(scaler, save_dir / "scaler.pkl")
    print(f"Processed data saved to: {save_dir}")


def plot_training_history(history, model_name, timestamp):
    """Plot and save training metrics history"""
    metrics_to_plot = [
        ("loss", "Loss"),
        ("mae", "Mean Absolute Error"),
        ("rmse", "Root Mean Squared Error"),
    ]

    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 15))
    fig.suptitle(f"{model_name} Training History", fontsize=16)

    for idx, (metric, title) in enumerate(metrics_to_plot):
        axes[idx].plot(history.history[metric], label=f"Training {title}")
        axes[idx].plot(history.history[f"val_{metric}"], label=f"Validation {title}")
        axes[idx].set_title(title)
        axes[idx].set_xlabel("Epoch")
        axes[idx].set_ylabel(title)
        axes[idx].legend()
        axes[idx].grid(True)

    plt.tight_layout()
    save_path = f"visualization/{model_name}_training_history_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Training history plot saved to: {save_path}")

    # Save training history data
    history_df = pd.DataFrame(history.history)
    csv_path = f"results/{model_name}_training_history_{timestamp}.csv"
    history_df.to_csv(csv_path, index=False)
    print(f"Training history data saved to: {csv_path}")


def main():
    # Create necessary directories
    ensure_directories()

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load and process data
    print("\nLoading and processing data...")
    (X_train, y_train), (X_test, y_test), scaler = load_and_process_data(
        SETTINGS["data_path"], SETTINGS["window_size"], SETTINGS["test_size"]
    )

    # Save processed data
    save_processed_data(
        ((X_train, y_train), (X_test, y_test)),
        "data/processed",
        scaler,
    )

    # Build and train Transformer
    print("\nTraining Transformer with best parameters...")
    transformer = build_transformer(
        window_size=SETTINGS["window_size"],
        **BEST_TRANSFORMER_PARAMS,
    )

    transformer_history = transformer.fit(
        X_train.reshape(-1, SETTINGS["window_size"], 1),
        y_train,
        batch_size=SETTINGS["batch_size"],
        epochs=SETTINGS["epochs"],
        validation_split=0.1,
        verbose=1,
        callbacks=create_callbacks("transformer", timestamp),
    )

    # Plot and save Transformer training history
    plot_training_history(transformer_history, "Transformer", timestamp)

    # Save Transformer model
    transformer_save_path = f"models/transformer_model_{timestamp}"
    transformer.save(transformer_save_path)
    transformer.save_weights(f"models/transformer_weights_{timestamp}.h5")
    print(f"Transformer model saved to: {transformer_save_path}")

    # Build and train LSTM
    print("\nTraining LSTM with best parameters...")
    lstm = build_lstm(
        window_size=SETTINGS["window_size"],
        **BEST_LSTM_PARAMS,
    )

    lstm_history = lstm.fit(
        X_train.reshape(-1, SETTINGS["window_size"], 1),
        y_train,
        batch_size=SETTINGS["batch_size"],
        epochs=SETTINGS["epochs"],
        validation_split=0.1,
        verbose=1,
        callbacks=create_callbacks("lstm", timestamp),
    )

    # Plot and save LSTM training history
    plot_training_history(lstm_history, "LSTM", timestamp)

    # Save LSTM model
    lstm_save_path = f"models/lstm_model_{timestamp}"
    lstm.save(lstm_save_path)
    lstm.save_weights(f"models/lstm_weights_{timestamp}.h5")
    print(f"LSTM model saved to: {lstm_save_path}")

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
    save_metrics(
        y_true, transformer_pred, lstm_pred, f"results/metrics_{timestamp}.txt"
    )

    # Save plot
    save_plot(
        y_true,
        transformer_pred,
        lstm_pred,
        f"visualization/forecast_plot_{timestamp}.html",
    )

    print(f"\nAll results saved with timestamp: {timestamp}")
    print("To view training progress, run: tensorboard --logdir=logs")


if __name__ == "__main__":
    main()
