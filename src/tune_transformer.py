import sys
import optuna
import logging
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# =============================================
# Create directories ONCE at script startup
# =============================================
required_dirs = ["logs", "models", "results"]
for dir_name in required_dirs:
    Path(dir_name).mkdir(parents=True, exist_ok=True)

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models import build_transformer
from src.data_loader import load_and_process_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/transformer_tuning.log"),
        logging.StreamHandler(),
    ],
)

# Load data
(X_train, y_train), (X_test, y_test), scaler = load_and_process_data(
    data_path="data/GHI_data.csv", window_size=5, test_size=72
)


def objective_transformer(trial):
    """Enhanced Transformer tuning with early stopping and checkpointing"""
    params = {
        "d_model": trial.suggest_int("d_model", 32, 256, step=32),
        "num_heads": trial.suggest_categorical("num_heads", [4, 8, 16]),
        "dff": trial.suggest_int("dff", 128, 512, step=64),
        "num_layers": trial.suggest_int("num_layers", 2, 6),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.4, step=0.05),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
    }

    logging.info(f"\n🚀 Starting Trial {trial.number} with params: {params}")

    try:
        # Build model
        model = build_transformer(window_size=5, **params)

        # Callbacks
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        checkpoint = ModelCheckpoint(
            f"models/transformer_trial_{trial.number}.h5",
            save_best_only=True,
            monitor="val_loss",
        )

        # Train with validation split
        history = model.fit(
            X_train,
            y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=params["batch_size"],
            callbacks=[early_stop, checkpoint],
            verbose=0,
        )

        # Load best weights from checkpoint
        model.load_weights(f"models/transformer_trial_{trial.number}.h5")

        # Evaluate
        val_loss = model.evaluate(X_test, y_test, verbose=0)
        logging.info(f"✅ Trial {trial.number} completed | Val Loss: {val_loss:.4f}")

        return val_loss

    except Exception as e:
        logging.error(f"❌ Trial {trial.number} failed: {str(e)}")
        return float("inf")


def tune_transformer():
    """Run optimization with pruning"""
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    study.optimize(
        objective_transformer, n_trials=50, timeout=3600 * 6
    )  # 6 hour timeout

    # Save results
    best_params = study.best_params
    logging.info(f"\n🏆 Best Params: {best_params}")
    logging.info(f"🏆 Best Val Loss: {study.best_value:.4f}")

    # Save best model
    best_model = build_transformer(window_size=5, **best_params)
    best_model.save("models/best_transformer.h5")

    return best_params


if __name__ == "__main__":
    tune_transformer()
