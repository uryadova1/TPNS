import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def create_sequences(data, window_size=10):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)


def preprocessing(split_ratio=0.8):
    df = pd.read_csv(
        "household_power_consumption.csv",
        sep=';',
        na_values='?',
        low_memory=False
    )

    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

    df.drop(columns=['Date', 'Time'], inplace=True)
    df.set_index('Datetime', inplace=True)

    df.dropna(inplace=True)

    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    df[numeric_cols] = df[numeric_cols].astype(float)

    series = df['Global_active_power'].values

    window_size = 20
    X, y = create_sequences(series, window_size)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = y.reshape(-1, 1)

    split = int(len(X) * split_ratio)
    return X[:split], y[:split], X[split:], y[split:]
