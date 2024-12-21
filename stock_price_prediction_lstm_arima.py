
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score

# Load stock data
ticker = 'TATAELXSI.NS'
data = yf.download(ticker, start='2020-01-01', end='2024-11-14', interval='1d')

# Preprocessing
data['Close'].fillna(method='ffill', inplace=True)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

# Prepare training data for LSTM
look_back = 60  # Number of days to look back
X_train, y_train = [], []
for i in range(look_back, len(scaled_data)):
    X_train.append(scaled_data[i-look_back:i, 0])
    y_train.append(scaled_data[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dense(units=1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train, y_train, epochs=10, batch_size=32)

# ARIMA model setup
arima_model = ARIMA(data['Close'], order=(5, 1, 0))  # Example ARIMA(5,1,0)
arima_result = arima_model.fit()

# Forecast using LSTM and ARIMA
predictions_lstm = lstm_model.predict(X_train)
predictions_lstm = scaler.inverse_transform(predictions_lstm)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

# Merge LSTM and ARIMA predictions
lstm_mse = mean_squared_error(y_train_actual, predictions_lstm)
arima_predictions = arima_result.predict(start=look_back, end=len(data)-1, dynamic=False)
arima_mse = mean_squared_error(data['Close'][look_back:], arima_predictions)

# Combine predictions
combined_predictions = 0.5 * predictions_lstm.flatten() + 0.5 * arima_predictions.values
combined_mse = mean_squared_error(data['Close'][look_back:], combined_predictions)

# Plot results
plt.figure(figsize=(14, 7))
plt.plot(data.index[look_back:], data['Close'][look_back:], label='Actual Prices', color='blue')
plt.plot(data.index[look_back:], predictions_lstm, label='LSTM Predictions', color='orange')
plt.plot(data.index[look_back:], arima_predictions, label='ARIMA Predictions', color='green')
plt.plot(data.index[look_back:], combined_predictions, label='Combined Predictions', color='red')
plt.title('Stock Price Prediction with LSTM and TSERIMA')
plt.xlabel('Date')
plt.ylabel('Stock Price (INR)')
plt.legend()
plt.show()

print(f"LSTM MSE: {lstm_mse:.4f}")
print(f"ARIMA MSE: {arima_mse:.4f}")
print(f"Combined MSE: {combined_mse:.4f}")

# Calculate and display R-squared
r2 = r2_score(data['Close'][look_back:], combined_predictions)
print(f"R-squared: {r2:.4f}")
