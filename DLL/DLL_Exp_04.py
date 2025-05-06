"""
Google Stock Price Prediction using Recurrent Neural Networks

This script implements a time series analysis and prediction system using RNN (LSTM)
to predict Google stock prices.

Author: Madhur Jaripatke
Date: May 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import seaborn as sns
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def get_stock_data(symbol='GOOGL', period='5y'):
    """Download stock data using yfinance."""
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df

def create_sequences(data, seq_length):
    """Create sequences for time series prediction."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def create_model(sequence_length):
    """Create a stacked LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_training_history(history):
    """Plot the training history."""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def predict_future(model, last_sequence, n_steps, scaler, sequence_length):
    """Predict future values using the trained model."""
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_steps):
        next_pred = model.predict(current_sequence.reshape(1, sequence_length, 1))[0]
        future_predictions.append(next_pred)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = next_pred
    
    return np.array(future_predictions)

def plot_predictions(df, y_test_inv, test_predict, future_dates=None, future_pred=None):
    """Plot the predictions against actual values."""
    plt.figure(figsize=(15, 6))
    plt.plot(df.index[-len(y_test_inv):], y_test_inv, label='Actual Price')
    plt.plot(df.index[-len(test_predict):], test_predict, label='Predicted Price')
    
    if future_dates is not None and future_pred is not None:
        plt.plot(future_dates, future_pred, label='Future Predictions', color='red')
    
    plt.title('Google Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Parameters
    sequence_length = 60  # Number of time steps to look back
    split_ratio = 0.8    # Training/testing split ratio
    
    # Get data
    print("Downloading Google stock data...")
    df = get_stock_data()
    print(f"Data shape: {df.shape}")
    
    # Plot original data
    plt.figure(figsize=(15, 6))
    plt.plot(df.index, df['Close'])
    plt.title('Google Stock Price History')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.show()
    
    # Prepare data
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = create_sequences(data_scaled, sequence_length)
    
    # Split into training and testing sets
    train_size = int(len(X) * split_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Create and train model
    print("\nCreating and training model...")
    model = create_model(sequence_length)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5
        )
    ]
    
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # Inverse transform predictions
    train_predict = scaler.inverse_transform(train_predict)
    y_train_inv = scaler.inverse_transform(y_train)
    test_predict = scaler.inverse_transform(test_predict)
    y_test_inv = scaler.inverse_transform(y_test)
    
    # Calculate RMSE
    train_rmse = np.sqrt(np.mean((train_predict - y_train_inv) ** 2))
    test_rmse = np.sqrt(np.mean((test_predict - y_test_inv) ** 2))
    print(f'\nTrain RMSE: ${train_rmse:.2f}')
    print(f'Test RMSE: ${test_rmse:.2f}')
    
    # Plot predictions
    plot_predictions(df, y_test_inv, test_predict)
    
    # Predict future prices
    print("\nPredicting future prices...")
    last_sequence = data_scaled[-sequence_length:]
    future_pred_scaled = predict_future(model, last_sequence, 30, scaler, sequence_length)
    future_pred = scaler.inverse_transform(future_pred_scaled)
    
    # Create future dates
    last_date = df.index[-1]
    future_dates = [last_date + timedelta(days=x) for x in range(1, 31)]
    
    # Plot predictions including future predictions
    plot_predictions(
        df,
        y_test_inv,
        test_predict,
        future_dates,
        future_pred
    )
    
    # Print future predictions
    print("\nPredicted prices for the next 5 days:")
    for date, price in zip(future_dates[:5], future_pred[:5]):
        print(f"{date.strftime('%Y-%m-%d')}: ${price[0]:.2f}")

if __name__ == "__main__":
    main()