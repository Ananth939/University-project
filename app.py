import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
import matplotlib.pyplot as plt
import yfinance as yf
import streamlit as st

def load_stock_data(stock_ticker):
    end = pd.Timestamp.now()
    start = end - pd.DateOffset(years=1)
    df = yf.download(stock_ticker, start, end)
    return df

def preprocess_data(df):
    df.reset_index(inplace=True)
    df.rename(columns={'Date': 'date', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}, inplace=True)
    df.set_index('date', inplace=True)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df

def create_dataset(data, time_step=1):
    dataX, dataY = [], []
    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(data[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def login():
    st.title('Stock Future Predictor with sentimenta analysis')
    stock_ticker = st.text_input('Enter stock Ticker', 'AAPL')
    predict_button = st.button('Predict')

    if predict_button:
        st.write(f"Fetching data for {stock_ticker}...")
        df = load_stock_data(stock_ticker)
        df = preprocess_data(df)
        st.write(df.head())
        st.write(df['close'][-1])
        closedf = df[['close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        closedf = scaler.fit_transform(np.array(closedf).reshape(-1, 1))
        st.subheader('closing Price VS Time Chart ')
        fig = plt.figure(figsize=(10,5))
        plt.style.use('dark_background')
        plt.plot(df.close , color = 'yellow')
        plt.legend()
        st.pyplot(fig)

        training_size = int(len(closedf) * 0.65)
        test_size = len(closedf) - training_size

        train_data, test_data = closedf[0:training_size,:], closedf[training_size:len(closedf),:1]

        time_step = 15
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # model = Sequential()
        # model.add(LSTM(32, return_sequences=True, input_shape=(time_step, 1)))
        # model.add(LSTM(32, return_sequences=True))
        # model.add(LSTM(32))
        # model.add(Dense(1))
        # model.compile(loss='mean_squared_error', optimizer='adam')
        # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=200, batch_size=5, verbose=1)
        model = load_model('my_stock.h5', compile=False)

        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)

        # Inverse transform predictions
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

        # Calculating evaluation metrics
        train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
        test_rmse = math.sqrt(mean_squared_error(y_test, test_predict))

        st.write(f"positive Ratio With actual Price: {train_rmse}")
        st.write(f"Negative Ratio With actual Price: {test_rmse}")

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(y_test, label='positive Ratio With actual Price:')
        ax.plot(test_predict, label='Negative Ratio With actual Price')
        ax.set_title('Stock Price Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Stock Price')
        ax.legend()
        st.pyplot(fig)

        

def main():
    login()

if __name__ == "__main__":
    main()
