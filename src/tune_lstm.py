import sys
from pathlib import Path
import optuna
import logging
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
)

# =============================================
# Create directories ONCE at script startup
# =============================================
required_dirs = ["logs", "models", "results"]
for dir_name in required_dirs:
    Path(dir_name).mkdir(parents=True, exist_ok=True)

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models import build_lstm
from src.data_loader import load_and_process_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/lstm_tuning.log"), logging.StreamHandler()],
)

# Load data
(X_train, y_train), (X_test, y_test), scaler = load_and_process_data(
    data_path="data/GHI_data.csv", window_size=5, test_size=72
)


def lr_scheduler(epoch, lr):
    """Learning rate scheduler"""
    return lr * 0.95 if epoch > 10 else lr


def objective_lstm(trial):
    """Enhanced LSTM tuning with learning rate scheduling"""
    params = {
        "units": trial.suggest_int("units", 64, 256, step=32),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.05),
        "recurrent_dropout": trial.suggest_float(
            "recurrent_dropout", 0.0, 0.3, step=0.05
        ),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    }

    logging.info(f"\n🚀 Starting Trial {trial.number} with params: {params}")

    try:
        # Build model
        model = build_lstm(window_size=5, **params)

        # Callbacks
        early_stop = EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        )
        checkpoint = ModelCheckpoint(
            f"models/lstm_trial_{trial.number}.h5",
            save_best_only=True,
            monitor="val_loss",
        )
        lr_callback = LearningRateScheduler(lr_scheduler)

        # Train
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=params["batch_size"],
            callbacks=[early_stop, checkpoint, lr_callback],
            verbose=0,
        )

        # Load best weights
        model.load_weights(f"models/lstm_trial_{trial.number}.h5")

        # Evaluate
        val_loss = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"✅ Trial {trial.number} completed | Val Loss: {val_loss:.4f}")

        return val_loss

    except Exception as e:
        logging.error(f"❌ Trial {trial.number} failed: {str(e)}")
        return float("inf")


def tune_lstm():
    """Run optimization with enhanced configuration"""
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(n_startup_trials=10)
    )
    study.optimize(objective_lstm, n_trials=50, timeout=3600 * 4)  # 4 hour timeout

    # Save results
    best_params = study.best_params
    logging.info(f"\n🏆 Best Params: {best_params}")
    logging.info(f"🏆 Best Val Loss: {study.best_value:.4f}")

    # Save best model
    best_model = build_lstm(window_size=5, **best_params)
    best_model.save("models/best_lstm.h5")

    return best_params


if __name__ == "__main__":
    tune_lstm()
