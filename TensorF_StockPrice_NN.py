#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 11:11:51 2023

@author: paviprathiraja

TensorFlow for machine learning in finance, specifically for predictive analysis.

Basic neural network to predict stock prices based on historical data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load historical stock price data
# Assume 'data.csv' contains two columns: 'Date' and 'Close'
data = pd.read_csv('data.csv')
closing_prices = data['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
closing_prices_normalized = scaler.fit_transform(closing_prices)

# Prepare data for training
seq_length = 10  # Length of sequences for each input data point
X, y = [], []

for i in range(len(closing_prices_normalized) - seq_length):
    X.append(closing_prices_normalized[i:i + seq_length])
    y.append(closing_prices_normalized[i + seq_length])

X, y = np.array(X), np.array(y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(X.shape[1], X.shape[2])),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse:.4f}')

# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(y_pred)
actual_prices = scaler.inverse_transform(y_test)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(predicted_prices, label='Predicted Prices')
plt.plot(actual_prices, label='Actual Prices')
plt.title('Stock Price Prediction using TensorFlow')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(predicted_prices, label='Predicted Prices')
plt.plot(actual_prices, label='Actual Prices')
plt.title('Stock Price Prediction using TensorFlow')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
plt.xlim (150, 200)
plt.show()
# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(predicted_prices, label='Predicted Prices')
plt.plot(actual_prices, label='Actual Prices')
plt.title('Stock Price Prediction using TensorFlow')
plt.xlabel('Time')
plt.ylabel('Closing Price')
plt.legend()
plt.xlim (0, 50)
plt.show()