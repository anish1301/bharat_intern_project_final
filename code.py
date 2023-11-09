# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Download data from 2020
url = 'https://query1.finance.yahoo.com/v7/finance/download/GOOGL?period1=1577836800&period2=9999999999&interval=1d&events=history'
data = pd.read_csv(url, index_col='Date', parse_dates=True)
data.head()

# Data preparation
data = data[['Close']]
data.head()

# Data normalization
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Splitting data into train and test sets
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]

# Creating sequences
def create_sequences(data, seq_len=30):
  sequences = []
  for i in range(len(data) - seq_len):
    sequences.append(data[i:(i + seq_len)])
  return np.array(sequences)

seq_len = 30
train_sequences = create_sequences(train_data, seq_len)
test_sequences = create_sequences(test_data, seq_len)

# Building LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(seq_len, 1)))
model.add(LSTM(units=100))
model.add(Dense(units=1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Training the model
model.fit(train_sequences, train_data[seq_len:], epochs=100, batch_size=16)

# Predictions
predictions = model.predict(test_sequences)
predictions = scaler.inverse_transform(predictions)

# Plotting actual and predicted values
#plt.plot(data.index[train_size + seq_len:], test_data[seq_len:], label='Actual')
plt.plot(data.index[train_size + seq_len:], predictions, label='Predicted')
plt.legend()
plt.show()
