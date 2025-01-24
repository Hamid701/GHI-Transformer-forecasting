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


# Load models and data ONCE at startup
def load_models_and_data():
    """Load models, data, and scaler"""
    try:
        # Load scaler
        scaler_path = Path(__file__).parent.parent / "data/processed/scaler.pkl"
        print(f"Loading scaler from: {scaler_path}")
        scaler = joblib.load(scaler_path)

        # Load test data
        X_test_path = Path(__file__).parent.parent / "data/processed/X_test.npy"
        y_test_path = Path(__file__).parent.parent / "data/processed/y_test.npy"
        print(f"Loading X_test from: {X_test_path}")
        print(f"Loading y_test from: {y_test_path}")
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)

        # Load models
        transformer_path = Path(__file__).parent.parent / "models/transformer_model"
        lstm_path = Path(__file__).parent.parent / "models/lstm_model"
        print(f"Loading Transformer from: {transformer_path}")
        print(f"Loading LSTM from: {lstm_path}")
        transformer = tf.keras.models.load_model(transformer_path)
        lstm = tf.keras.models.load_model(lstm_path)

        # Generate predictions
        transformer_pred = scaler.inverse_transform(transformer.predict(X_test))
        lstm_pred = scaler.inverse_transform(lstm.predict(X_test.reshape(-1, 5, 1)))
        y_true = scaler.inverse_transform(y_test)

        # Create a datetime index for the data
        dates = pd.date_range(start="2023-01-01", periods=len(y_true), freq="h")
        y_true = pd.Series(y_true.flatten(), index=dates)
        transformer_pred = pd.Series(transformer_pred.flatten(), index=dates)
        lstm_pred = pd.Series(lstm_pred.flatten(), index=dates)

        print("Data loaded successfully!")
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
