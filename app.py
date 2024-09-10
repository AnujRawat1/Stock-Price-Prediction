import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

# Load the pre-trained model
model = load_model('D:\Python\Projects\Stock Price Prediction\Stock Prediction Model.keras')

st.header('Stock Market Predictor')

# Input for stock symbol
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2013-01-01'
end = '2024-09-09'  # Updated end date to current date for completeness

# Download stock data using yfinance
data = yf.download(stock, start, end)

st.subheader('Stock Data')
st.write(data)

# Prepare data for prediction
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Scale data using MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plotting moving averages
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

# Prepare data for prediction
x = []
y_true = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y_true.append(data_test_scale[i, 0])

x, y_true = np.array(x), np.array(y_true)

# Predict using the loaded model
predictions = model.predict(x)
predictions = predictions * scaler.scale_
y_true = y_true * scaler.scale_


# Plotting original vs predicted prices
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predictions, 'r', label='Predicted Price')
plt.plot(y_true, 'g', label='Original Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)


# Reverse scaling for predictions and original prices
y_true_original = scaler.inverse_transform(y_true.reshape(-1, 1))  # Reshape for inverse transform
predictions_original = scaler.inverse_transform(predictions)

# Ensure data index starts from the correct point (match the length of predictions)
results = pd.DataFrame({
    'Date': data.index[-len(predictions):],  # Adjusted to match the length of predictions
    'Original Price': y_true_original.flatten()[-len(predictions):],  # Match length of predictions
    'Predicted Price': predictions_original.flatten()
})

st.subheader('Original Price vs Predicted Price')
st.table(results)
