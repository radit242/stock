import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
import datetime as date  

from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import load_model
import pickle

#slider 
#n_years = st.slider('Years of prediction:', 1, 3)

# User input for stock symbol and date range
st.title('Stock Prediction App')
st.write("This app predicts the closing price of a stock for four months based on the data from the past 10 years.")
stocks = ('GME','GOOG', 'AAPL', 'MSFT')
stock = st.selectbox('Select dataset for prediction', stocks)
start = '2020-01-01'
end = '2021-01-01'

#table of data 

def read_data():
    data = yf.download(stock, start, end)
    return data
data = read_data()
st.subheader('Raw data')
st.write(data.tail())

df = data.reset_index()
df = df.drop(['Adj Close'], axis=1)
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()

st.subheader('Closing Price Vs Time chart with 100MA and 200MA')
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='stock_close', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=df['Date'], y=ma100, name='MA100', line=dict(color='red', width=3)))
fig.add_trace(go.Scatter(x=df['Date'], y=ma200, name='MA200', line=dict(color='green', width=3)))
fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
st.plotly_chart(fig, use_container_width=False)

#loading models 

model1_path = "prophet_model.pkl"

model = load_model('stock_prediction2.h5')
with open(model1_path, "rb") as file:
    prophet = pickle.load(file)

scaler = MinMaxScaler(feature_range=(0,1))

#slider
st.subheader('Predictions for four months')
n_years = 4/12

@st.cache_data(show_spinner=False)
def forcast(n_years):
    n = n_years+4
    period = int(n*365)
    future = prophet.make_future_dataframe(periods=period)
    forcast = prophet.predict(future)
    return forcast, future

forcast, future = forcast(n_years)

@st.cache_data(show_spinner=False)
def coping_df(df, future, _model):
    _scaler = MinMaxScaler(feature_range=(0,1))
    temp_df = df.copy()
    temp_df.drop(['Open', 'High', 'Low', 'Date', 'Volume'], axis=1, inplace=True)
    temp_df = _scaler.fit_transform(temp_df)
    
    x_df= []
    y_df = []

    for i in range(100, temp_df.shape[0]):
        x_df.append(temp_df[i-100 : i])
        y_df.append(temp_df[i,0])

    x_df, y_df = np.array(x_df), np.array(y_df)
    y_df_pred = model.predict(x_df)
    original_y_df_pred = _scaler.inverse_transform(y_df_pred)
    original_y_df = _scaler.inverse_transform(y_df.reshape(-1,1))

    #convert to dataframe
    temp_df = _scaler.inverse_transform(temp_df)
    temp_df = pd.DataFrame(temp_df, columns=['Close'])

    for i, row in temp_df.iterrows():
        if i >= 100:
            temp_df.loc[i, 'y_pred'] = original_y_df_pred[i-100]
        else:
            temp_df.loc[i, 'y_pred'] = 0

    for i, row in temp_df.iterrows():
        if i >= 100:
            temp_df.loc[i, 'y_org'] = original_y_df[i-100]
        else:
            temp_df.loc[i, 'y_org'] = temp_df.loc[i, 'Close']
    
    temp_df['Date'] = df['Date']

    index = 0
    for i in range(len(future)):
        if future['ds'][i] == temp_df['Date'][len(temp_df)-1]:
            index = i 
            break

    temp_df = pd.concat([temp_df, future.iloc[index+1:]], ignore_index=True)
    
    def update_date(temp_df):
        temp_df_copy = temp_df.copy()
        for i in range(len(temp_df_copy)):
            if pd.isna(temp_df_copy['Date'][i]):
                temp_df_copy.loc[i, 'Date'] = temp_df_copy['ds'][i]
            elif pd.isna(temp_df_copy['ds'][i]):
                temp_df_copy.loc[i, 'ds'] = temp_df_copy['Date'][i]
        return temp_df_copy

    temp_df = update_date(temp_df)

    temp_df.drop(['ds'], axis=1, inplace=True)

    

    temp_df_copy = temp_df.copy()
    temp_df_copy_Close = temp_df["Close"].copy()
    
    
    _scaler = MinMaxScaler(feature_range=(0,1))

    temp_df_copy.drop(['Date','Close','y_org'], axis=1, inplace=True)
    temp_df_copy = _scaler.fit_transform(temp_df_copy)
    temp_df_copy_Close = _scaler.fit_transform(temp_df_copy_Close.values.reshape(-1,1))

    index = 0
    for i in range(len(temp_df_copy)):
        if pd.isna(temp_df_copy[i][0]):
            index = i
            break

    x_future = []
    for i in range(index-1, len(temp_df_copy)-1):
        x_future.append(temp_df_copy_Close[i-300 : i])
        x_future = np.array(x_future)
        y_prediction = _model.predict(x_future)
        temp_df_copy[i+1][0] = y_prediction
        temp_df_copy_Close[i+1][0] = y_prediction
        x_future = []
 
    
    

    temp_df_copy = _scaler.inverse_transform(temp_df_copy)
    temp_df_copy = pd.DataFrame(temp_df_copy, columns=['y_pred'])

    for i in range(index, len(temp_df)):
        temp_df.loc[i, 'y_pred'] = temp_df_copy.loc[i, 'y_pred']

    temp_df.drop_duplicates(subset='Date', keep='first', inplace=True)

    return temp_df, index

temp_df, index = coping_df(df,future, model)




st.subheader('Predictions Vs Actual Price Vs Time chart')
fig = go.Figure()
fig.add_trace(go.Scatter(x=forcast['ds'], y= forcast['yhat_upper'], name='Prophet Predictions', line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=temp_df['Date'], y=temp_df['y_pred'], name='LSTM Predictions', line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=temp_df['Date'][:index], y= temp_df['Close'], name='Actual Price', line=dict(color='green', width=2)))
fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
#increase the width and height of fig 
fig.update_layout(
    autosize=False,
    width=1000,
    height=500,
)
st.plotly_chart(fig, use_container_width=False)

