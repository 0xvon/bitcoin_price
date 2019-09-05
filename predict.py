# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import warnings
import glob
import re
import itertools
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Masking
from keras.layers.recurrent import SimpleRNN, LSTM
from keras.layers.normalization import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# %%
datasets = glob.glob("*.csv")


# %%
df = pd.read_csv(datasets[1])
df = df[["Timestamp", "Weighted_Price"]]
df = df.dropna()


# %%
# 直前のn_prev 日分のデータと、その次の営業日のデータを生成
def _load_data(data, n_prev=50):

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        t = data.iloc[i:i+n_prev].values
        t = t.reshape(-1, 1)
        docX.append(t)
        docY.append(data.iloc[i+n_prev].values)
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY
# 学習用とテスト用データを分割、ただし分割する際に_load_data()を適用


def train_test_split(df, test_size=0.1, n_prev=50):
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn], n_prev)
    X_test, y_test = _load_data(df.iloc[ntrn:], n_prev)

    return (X_train, y_train), (X_test, y_test)


# %%
prices = df.Weighted_Price.values.reshape(-1, 1)

# 対数をとる
prices_log = np.log(prices)

# MinMaxScalerで正規化
scaler = MinMaxScaler(feature_range=(0, 1))
prices_ex = scaler.fit_transform(prices_log)


# %%
df_ex = pd.DataFrame({"price": prices_ex.flatten()}).set_index(df.Timestamp)
df_ex = df_ex[df_ex.price != 0]
# df_ex.plot()


# %%
length_of_sequences = 4
test_size = 0.2

(X_train, y_train), (X_test, y_test) = train_test_split(
    df_ex, test_size=test_size, n_prev=length_of_sequences)
# %%
in_out_neurons = 1
hidden_neurons = 100

model = Sequential()
model.add(LSTM(hidden_neurons, input_shape=(
    X_train.shape[1], X_train.shape[2]), return_sequences=False))
# model.add(LSTM(hidden_neurons, return_sequences=False))
model.add(Dense(in_out_neurons))

es_cb = EarlyStopping(monitor='val_loss', patience=0.01,
                      verbose=1, mode='auto')
model.compile(loss="mean_squared_error", optimizer="adam")
history = model.fit(
    X_train, y_train,
    batch_size=256,
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[es_cb],
)


# %%
pred = model.predict(X_test)
pred_inv = np.exp(scaler.inverse_transform(pred.reshape(-1, 1)))
true_inv = np.exp(scaler.inverse_transform(y_test.reshape(-1, 1)))
rmse = np.sqrt(mean_squared_error(pred_inv, true_inv))

print("RMSE: {}".format(rmse))
plt.figure(figsize=(20, 10))
plt.plot(df.Timestamp.values[- len(pred):], pred_inv, label='pred')
plt.plot(df.Timestamp.values[- len(pred):], true_inv, label='true')
plt.legend()
plt.title('prediction of Bitcoin price with LSTM')
plt.xlabel('date')
plt.ylabel('price')
plt.show()
