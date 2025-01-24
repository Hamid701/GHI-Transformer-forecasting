import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def calculate_metrics(y_true, y_pred):
    """Calculate RMSE, nRMSE, MAE, and MASE"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    nrmse = rmse / np.mean(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    mase = calculate_mase(y_true, y_pred)
    return rmse, nrmse, mae, mase


def calculate_mase(y_true, y_pred):
    """Calculate Mean Absolute Scaled Error (MASE)"""
    naive_forecast = np.roll(y_true, 1)  # Naive forecast: y(t) = y(t-1)
    naive_mae = mean_absolute_error(
        y_true[1:], naive_forecast[1:]
    )  # Skip the first value
    model_mae = mean_absolute_error(y_true, y_pred)
    return model_mae / naive_mae


def print_metrics(y_true, transformer_pred, lstm_pred):
    """Print formatted metrics"""
    print("\nEvaluation Metrics:")
    print(f"{'Model':<12} | {'RMSE':<8} | {'nRMSE':<8} | {'MAE':<8} | {'MASE':<8}")
    print("-" * 55)

    for name, pred in [("Transformer", transformer_pred), ("LSTM", lstm_pred)]:
        rmse, nrmse, mae, mase = calculate_metrics(y_true, pred)
        print(f"{name:<12} | {rmse:<8.2f} | {nrmse:<8.4f} | {mae:<8.2f} | {mase:<8.2f}")


def save_metrics(y_true, transformer_pred, lstm_pred, file_path):
    """Save metrics to a file"""
    from pathlib import Path

    file_path = Path(file_path)
    file_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Create directory if it doesn't exist

    with open(file_path, "w") as f:
        f.write("Evaluation Metrics:\n")
        f.write(
            f"{'Model':<12} | {'RMSE':<8} | {'nRMSE':<8} | {'MAE':<8} | {'MASE':<8}\n"
        )
        f.write("-" * 55 + "\n")
        for name, pred in [("Transformer", transformer_pred), ("LSTM", lstm_pred)]:
            rmse, nrmse, mae, mase = calculate_metrics(y_true, pred)
            f.write(
                f"{name:<12} | {rmse:<8.2f} | {nrmse:<8.4f} | {mae:<8.2f} | {mase:<8.2f}\n"
            )
    print(f"Metrics saved to: {file_path}")
