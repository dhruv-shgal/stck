import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

st.title('Stock Price Prediction App')

# Sidebar
st.sidebar.title('Settings')
stock_symbol = st.sidebar.text_input('Enter Stock Symbol (e.g., AAPL for Apple)', 'AAPL')
period = st.sidebar.selectbox('Select Time Period', ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'])

def load_data(symbol, period):
    data = yf.download(symbol, period=period)
    return data

data_load_state = st.text('Loading data...')
data = load_data(stock_symbol, period)
data_load_state.text('Data loaded successfully!')

st.subheader('Raw Data')
st.write(data)

# Plotting
st.subheader('Interactive Stock Price Plot')
trace1 = go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close')
layout = go.Layout(title='Interactive Stock Price Plot', xaxis_title='Date', yaxis_title='Close Price')
fig = go.Figure(data=[trace1], layout=layout)
st.plotly_chart(fig, use_container_width=True)

# Debugging: Check Data
st.subheader('Debugging: Check Data')
st.write(data.head())

# Debugging: Check Missing Values
st.subheader('Debugging: Check Missing Values')
st.write(data.isnull().sum())

# Prophet Forecasting
st.subheader('Prophet Forecasting')

# Preprocess data for Prophet
df_prophet = data.reset_index()[['Date', 'Close']].rename(columns={'Date':'ds', 'Close':'y'})

# Debugging: Check Data for Prophet
st.write(df_prophet.head())

# Check if data has enough rows
if len(df_prophet) < 2:
    st.error('Dataframe has less than 2 non-NaN rows.')
else:
    m = Prophet()
    m.fit(df_prophet)
    future = m.make_future_dataframe(periods=365)
    forecast = m.predict(future)
    fig = plot_plotly(m, forecast)
    st.plotly_chart(fig)
