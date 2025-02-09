import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
from datetime import datetime, timedelta

def calculate_technical_indicators(df):
    df['MA7'] = df['INR'].rolling(window=7).mean()
    df['MA14'] = df['INR'].rolling(window=14).mean()
    df['MA30'] = df['INR'].rolling(window=30).mean()
    df['RSI'] = calculate_rsi(df['INR'])
    df['MACD'] = calculate_macd(df['INR'])
    df.bfill(inplace=True)
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def predict_next_n_days(model, last_sequence, scaler, n_days=7, features=None):
    """
    Predict gold prices for the next n days using the trained model
    """
    predictions = []
    current_sequence = last_sequence.copy()
    
    for i in range(n_days):
        # Predict next day
        next_pred_scaled = model.predict(current_sequence.reshape(1, sequence_length, len(features)))
        
        # Convert prediction to actual price
        next_pred = scaler.inverse_transform(
            np.hstack((next_pred_scaled, np.zeros((1, len(features)-1))))
        )[0, 0]
        
        predictions.append(next_pred)
        
        # Update sequence for next prediction
        # Create a new row with technical indicators
        new_row = np.zeros(len(features))
        new_row[0] = next_pred_scaled[0, 0]  # Normalized predicted price
        
        # Remove first row and append new prediction
        current_sequence = np.vstack((current_sequence[1:], new_row))
    
    return predictions

# Load dataset
file_path = "Adjusted_Daily_Gold_Rate_India.xlsx"
df = pd.read_excel(file_path)
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.drop(columns=['Premium_Discount_USD', 'Adjusted_Gold_Price_INR'], inplace=True)
df['INR'] = df['INR'].interpolate().bfill()
df = calculate_technical_indicators(df)

# Normalize Data
scaler = MinMaxScaler()
features = ['INR', 'MA7', 'MA14', 'MA30', 'RSI', 'MACD']
df[features] = scaler.fit_transform(df[features])
data = df[features].values

sequence_length = 60
X, y = [], []
for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])
    y.append(data[i+sequence_length][0])
X, y = np.array(X), np.array(y)

# Load or train model
model_path = "gold_price_lstm_model.keras"
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully!")
else:
    model = Sequential([
        Input(shape=(sequence_length, len(features))),
        Bidirectional(LSTM(100, return_sequences=True, activation='relu')),
        Dropout(0.2),
        Bidirectional(LSTM(100, return_sequences=False, activation='relu')),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5)
    
    model.fit(X, y, epochs=150, batch_size=32, verbose=1, callbacks=[early_stopping, reduce_lr])
    
    model.save(model_path)
    print("Model saved successfully!")

# Get last sequence for prediction
last_sequence = df[features].values[-sequence_length:]

# Predict next 7 days
predictions = predict_next_n_days(model, last_sequence, scaler, n_days=7, features=features)

# Generate future dates
last_date = df.index[-1]
future_dates = [last_date + timedelta(days=x+1) for x in range(7)]

# Plot results
plt.figure(figsize=(15,7))
# Plot historical data
plt.plot(df.index[-30:], 
         df['INR'].iloc[-30:] * scaler.data_range_[0] + scaler.data_min_[0], 
         label="Historical Price")

# Plot predictions
plt.plot(future_dates, predictions, 'r--', label="Predicted Price")
plt.xlabel("Date")
plt.ylabel("Gold Price (INR)")
plt.title("Gold Price Prediction for Next 7 Days")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Print predictions
print("\nPredicted Gold Prices for Next 7 Days:")
for date, price in zip(future_dates, predictions):
    print(f"{date.strftime('%Y-%m-%d')}: INR {price:.2f}")