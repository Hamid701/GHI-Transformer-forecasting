import numpy as np
import plotly.graph_objects as go
from pathlib import Path


def save_plot(y_true, transformer_pred, lstm_pred, file_path):
    """Save interactive plot to HTML file"""
    # Create directory if it doesn't exist
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Create plot
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(y_true)),
            y=y_true.flatten(),
            mode="lines",
            name="Actual",
            line=dict(color="red", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(transformer_pred)),
            y=transformer_pred.flatten(),
            mode="lines",
            name="Transformer",
            line=dict(color="blue"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.arange(len(lstm_pred)),
            y=lstm_pred.flatten(),
            mode="lines",
            name="LSTM",
            line=dict(color="green"),
        )
    )

    # Update layout
    fig.update_layout(
        title="GHI Forecasting Results",
        xaxis_title="Time Steps",
        yaxis_title="GHI (W/m²)",
        hovermode="x unified",
        height=600,
    )

    # Save plot
    fig.write_html(file_path)
    print(f"Plot saved to: {file_path}")
