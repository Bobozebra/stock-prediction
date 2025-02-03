import yfinance as yf
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import datetime

# Title of the app
st.title("📈 Simple Stock Price Prediction")

# Input: Stock Symbol
stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA):", "AAPL")

if stock_symbol:
    # Get stock data using yfinance
    data = yf.download(stock_symbol, period="1y", interval="1d")
    
    # Feature: Previous closing price
    data['Prev Close'] = data['Close'].shift(1)
    
    # Drop the first row with NaN
    data.dropna(inplace=True)
    
    # X is the previous day's closing price, Y is today's closing price
    X = data[['Prev Close']]
    Y = data['Close']
    
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(X, Y)
    
    # Predict next day's price
    next_day = pd.DataFrame({'Prev Close': [data['Close'].iloc[-1]]})
    prediction = model.predict(next_day)
    
    # Display result
    st.write(f"Predicted price for the next day: ${prediction[0]:.2f}")
    
    # Show the stock data as a chart
    st.line_chart(data['Close'])
# Minor update for redeployment

