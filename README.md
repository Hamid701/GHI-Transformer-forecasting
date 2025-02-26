# Revolutionizing Solar Energy Forecasting with Transformer-Based Models

## Overview
As the world grapples with the challenges of global warming, renewable energy sources like solar power are becoming increasingly critical. However, integrating solar energy into electricity grids requires **accurate forecasting** of solar resources to balance supply and demand efficiently. Traditional forecasting models, such as ARIMA and LSTM, have shown promise but face limitations in handling long-term dependencies and computational efficiency.

In this project, I introduce a **novel Transformer-based approach** for **Global Horizontal Irradiance (GHI)** forecasting. By leveraging the **multi-head attention mechanism**, this model eliminates the sequential training bottleneck of recurrent models, enabling parallel processing and significantly improving both accuracy and training speed. The results demonstrate a **20% reduction in Root Mean Squared Error (RMSE)** compared to **LSTM models**, making it a game-changer for solar energy forecasting.

### Why This Project Matters
Why This Project Matters
Solar energy is inherently intermittent due to its dependence on weather conditions. Accurate GHI forecasts enable:

- **Optimized grid management** : Better integration of solar energy into existing grids.
- **Reduced operational costs** : Minimizing reliance on backup power sources.
- **Improved energy planning** : Enhancing decision-making for energy storage and distribution.

### Objectif
This study implements advanced time series forecasting models to predict **(GHI)** using hourly
historical data. The implementation includes state-of-the-art models such as **Transformer** and **LSTM**,
which focus on parallelization and handling long-term dependencies in time series data.

---

## Publication
You can check the white preprint paper here:
- **Transformer-Based Global Horizontal Irradiance Forecasting: A Multi-Head Attention Approach for Hour-Ahead Predictions**  
  [Read the paper here](https://www.researchgate.net/publication/388350500_Transformer-Based_Global_Horizontal_Irradiance_Forecasting_A_Multi-Head_Attention_Approach_for_Hour-Ahead_Predictions)

---

## Key Features of the Project

### 1. Transformer-Based Architecture
The Transformer model uses a multi-head attention mechanism to capture both short-term and long-term dependencies in time series data. Unlike recurrent models, it processes all time steps simultaneously, enabling faster training and better generalization.

### 2. Interactive Dashboard
To make the model accessible and actionable, I developed an interactive dashboard using Dash and Plotly . Key features include:

- **Model comparison** : Compare predictions from Transformer and LSTM models.
- **Real-time metrics** : Visualize RMSE, nRMSE, MAE, and MASE for selected models.
- **Time range slider** : Focus on specific time periods for detailed analysis.
- **Download functionality** : Export prediction data as ```.csv``` files for further use.
  
### 3. Advanced Hyperparameter Optimization
Using state-of-the-art techniques like the Tree-Structured Parzen Estimator (TPE) algorithm, I optimized hyperparameters for both Transformer and LSTM models, ensuring optimal performance while minimizing computational overhead.

---

## Methodology

### Data Source
The dataset consists of **hourly GHI** from 2006 to 2015 for Rabat, Morocco (33.97°N, -6.85°E), provided by the **Copernicus Atmosphere Monitoring Service (CAMS)** . The data was preprocessed using a **sliding window approach** to generate labeled input-output pairs for supervised learning.

Below is a visualization of the hourly Global Horizontal Irradiance (GHI) data and it's hourly distribution for the site Rabat, highlighting its variability over time.
<div align="center">
  <img src="visualization\ghi_vs_time.png" alt="Time Series Visualization" width="45%">
  <img src="visualization\GHI_Distribution_by_Hour.png" alt="GHI Distribution by Hour" width="45%"/>
  <img src="visualization\Average_GHI_by_Month.png" alt="Average GHI by Month" width="45%">
  <img src="visualization\GHI_Distribution_by_Season.png" alt="GHI Distribution by Season" width="45%"/>
</div>

1. **Daily Pattern**:
   - Clear diurnal cycle with peak values during mid-day.
   - Zero values during night hours.

2. **Seasonal Variation**:
   - Higher GHI values during the summer months.
   - Lower values during winter months.
   - Significant variation in day length across seasons.
     
### Model Design

The Transformer model uses a multi-head attention mechanism to capture both short-term and long-term dependencies in the data. Below is a simplified diagram of the architecture.

- **Input Embedding**: Maps the time series into a high-dimensional space.
- **Positional Encoding**: Encodes temporal order using sinusoidal functions.
- **Encoder Layers**: Four stacked encoder layers with multi-head self-attention and feed-forward networks.
- **Linear Layer**: Projects the final representation into a scalar value for GHI prediction.

<div align="center">
  <img src="visualization\Transformer_str.png" alt="Transformer architecture for forecasting one hour ahead of global horizontal irradiance (GHI)." width="700"/>
</div>

### Training Process
- **Transformer** : Achieved rapid convergence with minimal fluctuations in loss.
- **LSTM** : Showed slower convergence and higher error rates compared to the Transformer.

The training history of the Transformer and LSTM models is shown below. Notice how the Transformer converges faster and achieves lower validation errors.

<div align="center">
  <img src="visualization\Transformer_training_history_20250211_124625.png" alt="Transformer Training History" width="45%"/>
  <img src="visualization\LSTM_training_history_20250211_124625.png" alt="LSTM Training History" width="45%"/>
</div>

---

## Data Flow Diagram

<img src="visualization/data_flow.png" alt="Data flow" width="300" height="500"/>

--- 
## Results and Performance
The Transformer model outperformed all benchmark models across key metrics:

Evaluation Metrics:

| Model                 | RMSE  | nRMSE (%) | MAE   | MASE  |
|-----------------------|-------|-----------|-------|-------|
| **Transformer**       | 12.40 | 9.02      | 7.41  | 0.16  |
| **LSTM**              | 15.54 | 11.31     | 10.04 | 0.21  |
| **ARIMA**             | 18.09 | 13.45     | 16.12 | 0.35  |
| **Exp. Smoothing**    | 45.90 | 25.12     | 35.56 | 0.75  |
| **Persistence**       | 42.85 | 24.45     | 33.07 | 0.70  | 

<div align="center">
  <img src="visualization\Model_Performance_Comparison.png" alt="Model Performance Comparison" width="70%"/>
</div>

These results highlight the Transformer's superior ability to capture complex temporal patterns, making it ideal for solar energy forecasting.

---

## Interactive Visualizations
The interactive dashboard provides an intuitive interface for exploring the model's predictions and performance metrics. Here’s a quick preview:

- **Model Comparison**: Compare predictions from Transformer and LSTM.
- **Time Range Slider**: Focus on specific time periods.
- **Metrics Display**: Real-time RMSE, nRMSE, MAE and MASE for selected models.
- **Download**: Download the predictions to your local device as .csv file.
- 
### Quick Preview
<div align="center">
  <img src="visualization/Dash-Preview.gif" alt="Dashboard Demo" width="800"/>
</div>

### Full Video Demonstration
<div align="center">
  <video width="800" controls>
    <source src="visualization/Dash-Preview.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

Alternatively, you can:
- [Download the demo video](visualization/Dash-Preview.mp4)

---

## Installation and Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/Hamid701/GHI-Transformer-forecasting.git
    cd GHI-Transformer-forecasting
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the the script:
    ```bash
    python src/main.py
    ```
    
4. Run the dashboard locally:
    ```bash
    python src/dashboard.py
    ```

Open your browser and navigate to [http://127.0.0.1:8050](http://127.0.0.1:8050) to view the interactive dashboard.

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request with a detailed description of your changes.

## Future Work

- **Add Exogenous Variables**: Incorporate additional input features such as wind speed, temperature, and humidity to improve model accuracy.
- **Experiment with Hybrid Models**: Explore the performance of hybrid models that combine the strengths of Transformers and LSTMs.
- **Integrate Real-Time Data Updates**: Enable the dashboard to fetch and display real-time data for up-to-date forecasting.

## Acknowledgments

This project builds on the groundbreaking work of Vaswani et al. in their paper **"Attention is All You Need"** . Special thanks to the **Copernicus Atmosphere Monitoring Service (CAMS)** for providing the dataset.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Contact
For questions or collaborations, feel free to reach out:

Email: ettayyebi.hamid@gmail.com
LinkedIn: linkedin.com/in/ett-hamid
