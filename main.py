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

# Check if model exists
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
    
    model.fit(X, y, epochs=100, batch_size=32, verbose=1, callbacks=[early_stopping, reduce_lr])
    
    model.save(model_path)
    print("Model saved successfully!")

# Predict future prices
predictions = model.predict(X)
predictions = scaler.inverse_transform(np.hstack((predictions, np.zeros((len(predictions), len(features)-1)))))[:, 0]

# Plot results
plt.figure(figsize=(12,6))
plt.plot(df.index[sequence_length:], df['INR'].iloc[sequence_length:] * scaler.data_range_[0] + scaler.data_min_[0], label="Actual Price")
plt.plot(df.index[sequence_length:], predictions, label="Predicted Price", linestyle='dashed')
plt.xlabel("Date")
plt.ylabel("Gold Price (INR)")
plt.legend()
plt.show()

# Predict next day's price
last_60_days = df[features].values[-sequence_length:].reshape(1, sequence_length, len(features))
predicted_price_scaled = model.predict(last_60_days)
predicted_price = scaler.inverse_transform(np.hstack((predicted_price_scaled, np.zeros((1, len(features)-1)))))[:, 0]
print(f"Predicted Gold Price for Tomorrow: INR {predicted_price[0]:.2f}")


