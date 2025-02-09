# Gold Price Prediction Model

## Overview
This project implements a deep learning model to predict gold prices in Indian Rupees (INR) using historical data and technical indicators. The model utilizes a bidirectional LSTM architecture with attention mechanisms to forecast gold prices for the next 7 days.

## Features
- Historical gold price analysis
- Technical indicator calculations (Moving Averages, RSI, MACD)
- Data preprocessing and normalization
- Bidirectional LSTM model with dropout layers
- 7-day price forecasting
- Visualization of predictions

## Requirements
```
pandas
numpy
matplotlib
scikit-learn
tensorflow
```

## Project Structure
- `gold_price_lstm_model.keras`: Trained model file
- `Adjusted_Daily_Gold_Rate_India.xlsx`: Input dataset containing historical gold prices

## Technical Indicators
The model incorporates several technical indicators:
- 7-day Moving Average (MA7)
- 14-day Moving Average (MA14)
- 30-day Moving Average (MA30)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)

## Model Architecture
- Input Layer: 60-day sequence with 6 features
- Bidirectional LSTM layers with ReLU activation
- Dropout layers (0.2) for regularization
- Dense layers for final prediction
- Adam optimizer with learning rate of 0.001

## Usage
1. Ensure all required dependencies are installed
2. Place the dataset file in the project directory
3. Run the script to either:
   - Load an existing trained model
   - Train a new model if no saved model exists
4. View the prediction results and visualization

## Model Training
The model includes:
- Early stopping with patience of 10 epochs
- Learning rate reduction on plateau
- Batch size of 32
- Maximum of 150 epochs

## Output
The script generates:
- A line plot showing historical prices and predicted values
- Printed predictions for the next 7 days with dates

## Data Preprocessing
- Handles missing values through interpolation
- Normalizes features using MinMaxScaler
- Creates sequences of 60 days for training
- Calculates technical indicators automatically

## Notes
- The model automatically saves after training for future use
- Predictions are denormalized to show actual INR values
- The visualization includes the last 30 days of historical data

## Contributing
To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with a detailed description of changes

## License
This project is licensed under the MIT License. See the LICENSE file for details.