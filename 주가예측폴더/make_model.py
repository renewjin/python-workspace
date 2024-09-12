import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import pickle
import os

df = pd.read_csv('./005930.KS.csv')

df = df[['Close']]

data = df.to_numpy()

# 전체 데이터를 학습에 사용
transformer = MinMaxScaler()
data = transformer.fit_transform(data)

sequence_length = 7
window_length = sequence_length + 1

x_data = []
y_data = []
for i in range(0, len(data) - window_length + 1):
    window = data[i:i + window_length, :]
    x_data.append(window[:-1])
    y_data.append(window[-1])

x_data = np.array(x_data)
y_data = np.array(y_data)
x_data = np.expand_dims(x_data, -1)

# 데이터 크기 확인
print(f"x_data 크기: {x_data.shape}")
print(f"y_data 크기: {y_data.shape}")

x_data, y_data = shuffle(x_data, y_data)

if not os.path.exists('./samsung_electronics_stock_close_price_time_series_regression_model'):
    os.makedirs('./samsung_electronics_stock_close_price_time_series_regression_model')

with open('./transformer.pkl', 'wb') as f:
    pickle.dump(transformer, f)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=10, input_shape=(sequence_length, 1), return_sequences=False),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, name='output')
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

history = model.fit(x_data, y_data, epochs=50, validation_split=0.2)

model.save('samsung_electronics_stock_close_price_time_series_regression_model/my_model.keras')

# 모델 예측 및 검증
y_data = y_data.reshape(-1, 1)
y_data_inverse = transformer.inverse_transform(y_data)

y_predict = model.predict(x_data)
y_predict = y_predict.reshape(-1, 1)
y_predict_inverse = transformer.inverse_transform(y_predict)

import matplotlib.pyplot as plt
plt.plot(y_data_inverse, label='Actual')
plt.plot(y_predict_inverse, label='Predicted')
plt.xlabel('Time Period')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# 샘플 데이터 예측
x_test_sample = np.array([[30000.000000, 29300.000000, 30000.000000, 29980.000000, 29700.000000, 29020.000000, 28740.000000]])
x_test_sample = transformer.transform(x_test_sample)
x_test_sample = np.expand_dims(x_test_sample, -1)

y_predict_sample = model.predict(x_test_sample)
y_predict_sample_inverse = transformer.inverse_transform(y_predict_sample)
print(y_predict_sample_inverse.flatten()[0])