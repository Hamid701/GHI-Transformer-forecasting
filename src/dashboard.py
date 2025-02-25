import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import tensorflow as tf
import dash
from dash import dcc, html, Input, Output, dash_table, State
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import sys
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Now you can import from src
from src.evaluate import calculate_metrics


def find_model_file(model_type):
    """Find model file with flexible naming"""
    models_dir = project_root / "models"
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found at {models_dir}")

    # Look for model directories first
    model_dirs = list(models_dir.glob(f"*{model_type}_model_*"))
    if model_dirs:
        model_dir = sorted(model_dirs)[-1]  # Get the latest directory
        return model_dir

    # If no directories found, look for direct .h5 files
    model_files = list(models_dir.glob(f"*{model_type}*.h5"))
    if model_files:
        return sorted(model_files)[-1]  # Get the latest file

    raise FileNotFoundError(f"No {model_type} model found in {models_dir}")


# Load models and data ONCE at startup
def load_models_and_data():
    """Load models, data, and scaler with better error handling"""
    try:
        # Load scaler and test data from the correct directory
        processed_dir = project_root / "data" / "processed"
        if not processed_dir.exists():
            processed_dir = project_root / "dd"  # Try alternative directory

        print(f"Looking for data in: {processed_dir}")

        scaler_path = processed_dir / "scaler.pkl"
        X_test_path = processed_dir / "X_test.npy"
        y_test_path = processed_dir / "y_test.npy"

        # Load data files with error checking
        if not all(p.exists() for p in [scaler_path, X_test_path, y_test_path]):
            raise FileNotFoundError(f"Missing data files in {processed_dir}")

        print("Loading data files...")
        scaler = joblib.load(scaler_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

        # Find and load models with custom loader
        print("Loading models...")
        transformer_path = find_model_file("transformer")
        lstm_path = find_model_file("lstm")

        print(f"Found Transformer at: {transformer_path}")
        print(f"Found LSTM at: {lstm_path}")

        # Load models with error handling
        try:
            transformer = tf.keras.models.load_model(str(transformer_path))
            lstm = tf.keras.models.load_model(str(lstm_path))
        except Exception as e:
            print(f"Error loading models: {e}")
            # Try loading with custom_objects if needed
            custom_objects = {
                "RootMeanSquaredError": tf.keras.metrics.RootMeanSquaredError
            }
            transformer = tf.keras.models.load_model(
                str(transformer_path), custom_objects=custom_objects
            )
            lstm = tf.keras.models.load_model(
                str(lstm_path), custom_objects=custom_objects
            )

        # Generate predictions
        print("Generating predictions...")
        transformer_pred = transformer.predict(X_test)
        lstm_pred = lstm.predict(X_test.reshape(-1, 5, 1))

        # Inverse transform predictions
        transformer_pred = scaler.inverse_transform(transformer_pred)
        lstm_pred = scaler.inverse_transform(lstm_pred)
        y_true = scaler.inverse_transform(y_test)

        # Create datetime index
        dates = pd.date_range(start="2023-01-01", periods=len(y_true), freq="h")
        y_true = pd.Series(y_true.flatten(), index=dates)
        transformer_pred = pd.Series(transformer_pred.flatten(), index=dates)
        lstm_pred = pd.Series(lstm_pred.flatten(), index=dates)

        print("✓ Data loaded successfully!")
        return y_true, transformer_pred, lstm_pred

    except Exception as e:
        print(f"Error loading data: {e}")
        raise


# Load data and models
y_true, transformer_pred, lstm_pred = load_models_and_data()

# Create Dash app with Bootstrap styling
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "GHI Forecasting Dashboard"

# Layout
app.layout = dbc.Container(
    [
        # Title
        dbc.Row(
            dbc.Col(
                html.H1(
                    "Global Horizontal Irradiance (GHI) Forecasting Dashboard",
                    className="text-center",
                )
            )
        ),
        # Model selection dropdown
        dbc.Row(
            [
                dbc.Col(
                    html.Label(
                        "Select Model:",
                        title="Choose between Transformer, LSTM, or compare both models.",
                    ),
                    width=2,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="model-dropdown",
                        options=[
                            {"label": "Transformer", "value": "transformer"},
                            {"label": "LSTM", "value": "lstm"},
                            {"label": "Compare Both", "value": "compare"},
                        ],
                        value="compare",
                        clearable=False,
                    ),
                    width=4,
                ),
            ],
            className="mb-4",
        ),
        # Date range picker
        dbc.Row(
            [
                dbc.Col(
                    html.Label(
                        "Select Date Range:",
                        title="Choose a date range to focus on specific time periods.",
                    ),
                    width=2,
                ),
                dbc.Col(
                    dcc.DatePickerRange(
                        id="date-range-picker",
                        start_date=y_true.index[0],
                        end_date=y_true.index[-1],
                        display_format="YYYY-MM-DD",
                    ),
                    width=8,
                ),
            ],
            className="mb-4",
        ),
        # Graph
        dbc.Row(dbc.Col(dcc.Graph(id="forecast-plot"))),
        # Metrics table
        dbc.Row(
            dbc.Col(
                dash_table.DataTable(
                    id="metrics-table",
                    columns=[
                        {"name": "Model", "id": "model"},
                        {"name": "RMSE", "id": "rmse"},
                        {"name": "nRMSE", "id": "nrmse"},
                        {"name": "MAE", "id": "mae"},
                        {"name": "MASE", "id": "mase"},
                    ],
                    data=[],
                    style_table={"overflowX": "auto"},
                ),
                width=10,
            ),
            className="mb-4",
        ),
        # Download button
        dbc.Row(
            dbc.Col(
                html.Button(
                    "Download Predictions",
                    id="download-button",
                    className="btn btn-primary",
                ),
                width=2,
            ),
            className="mb-4",
        ),
        dcc.Download(id="download-data"),
        # Error message
        dbc.Row(dbc.Col(html.Div(id="error-message", style={"color": "red"}))),
    ],
    fluid=True,
)


# Callback for updating the plot and metrics
@app.callback(
    [
        Output("forecast-plot", "figure"),
        Output("metrics-table", "data"),
        Output("error-message", "children"),
    ],
    [
        Input("model-dropdown", "value"),
        Input("date-range-picker", "start_date"),
        Input("date-range-picker", "end_date"),
    ],
)
def update_plot(selected_model, start_date, end_date):
    try:
        # Slice data based on date range
        mask = (y_true.index >= start_date) & (y_true.index <= end_date)
        y_true_sliced = y_true[mask]
        transformer_pred_sliced = transformer_pred[mask]
        lstm_pred_sliced = lstm_pred[mask]

        # Create figure
        fig = go.Figure()

        # Add traces based on selected model
        if selected_model == "transformer" or selected_model == "compare":
            fig.add_trace(
                go.Scatter(
                    x=y_true_sliced.index,
                    y=transformer_pred_sliced,
                    mode="lines",
                    name="Transformer",
                    line=dict(color="blue"),
                )
            )

        if selected_model == "lstm" or selected_model == "compare":
            fig.add_trace(
                go.Scatter(
                    x=y_true_sliced.index,
                    y=lstm_pred_sliced,
                    mode="lines",
                    name="LSTM",
                    line=dict(color="green"),
                )
            )

        # Add actual values
        fig.add_trace(
            go.Scatter(
                x=y_true_sliced.index,
                y=y_true_sliced,
                mode="lines",
                name="Actual",
                line=dict(color="red", dash="dash"),
            )
        )

        # Update layout
        fig.update_layout(
            title="GHI Forecasting Results",
            xaxis_title="Time",
            yaxis_title="GHI (W/m²)",
            hovermode="x unified",
            height=600,
        )

        # Calculate metrics using evaluate.py
        metrics = []
        if selected_model == "transformer" or selected_model == "compare":
            rmse, nrmse, mae, mase = calculate_metrics(
                y_true_sliced, transformer_pred_sliced
            )
            metrics.append(
                {
                    "model": "Transformer",
                    "rmse": f"{rmse:.2f}",
                    "nrmse": f"{nrmse:.4f}",
                    "mae": f"{mae:.2f}",
                    "mase": f"{mase:.2f}",
                }
            )

        if selected_model == "lstm" or selected_model == "compare":
            rmse, nrmse, mae, mase = calculate_metrics(y_true_sliced, lstm_pred_sliced)
            metrics.append(
                {
                    "model": "LSTM",
                    "rmse": f"{rmse:.2f}",
                    "nrmse": f"{nrmse:.4f}",
                    "mae": f"{mae:.2f}",
                    "mase": f"{mase:.2f}",
                }
            )

        return fig, metrics, ""

    except Exception as e:
        return go.Figure(), [], f"Error: {str(e)}"


# Callback for downloading predictions
@app.callback(
    Output("download-data", "data"),
    [Input("download-button", "n_clicks")],
    [State("date-range-picker", "start_date"), State("date-range-picker", "end_date")],
    prevent_initial_call=True,
)
def download_predictions(n_clicks, start_date, end_date):
    try:
        # Slice data based on date range
        mask = (y_true.index >= start_date) & (y_true.index <= end_date)
        y_true_sliced = y_true[mask]
        transformer_pred_sliced = transformer_pred[mask]
        lstm_pred_sliced = lstm_pred[mask]

        # Create DataFrame
        df = pd.DataFrame(
            {
                "Date": y_true_sliced.index,
                "Actual": y_true_sliced,
                "Transformer": transformer_pred_sliced,
                "LSTM": lstm_pred_sliced,
            }
        )

        # Return CSV file
        return dcc.send_data_frame(df.to_csv, "predictions.csv")
    except Exception as e:
        return f"Error: {str(e)}"


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
