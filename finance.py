from yahoo_fin.stock_info import get_data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.svm import SVR
from matplotlib import pyplot as plt

aapl = get_data("QQQ", start_date="03/28/2021", end_date="03/28/2022", index_as_date = True, interval="1d")

aapl.dropna(inplace=True)
print(aapl)

aapl = aapl.to_numpy()[:, :-1]

# scaler = MinMaxScaler(feature_range=(0,1))
# closedf = scaler.fit_transform(aapl)

# train_size = int(closedf.shape[0] * 0.85)
# train_size = int(aapl.shape[0] * 0.85)
train_size = aapl.shape[0] - 10
test_size = aapl.shape[0] - train_size

train_data = aapl[:train_size, :]
test_data = aapl[train_size:, :]

train_label = train_data[:, [3]]
test_label = test_data[:, [3]]

train_data = np.delete(train_data, [3, 4], 1)
test_data = np.delete(test_data, [3, 4], 1)

scaler = MinMaxScaler(feature_range=(0,1))
train_data = scaler.fit_transform(train_data)
train_label = scaler.fit_transform(train_label)
test_data = scaler.fit_transform(test_data)
test_label = scaler.fit_transform(test_label)

svr = SVR().fit(train_data, train_label)
print(svr.score(test_data, test_label))

'''
new_test_data = [
                    open = closed from previous day, 
                    high = avg difference between open and high?,
                    low = avg difference between open and low?,
                    volume = avg volume of past few days, maybe 10?
                  ]

'''

# prev_close = train_label[-1, 0]
# open = []
# range_high = [train_data[train_size - test_size :, 1] - train_data[train_size - test_size, 0]]
# range_low = [train_data[train_size - test_size:, 0] - train_data[train_size - test_size, 2]]
# volume = np.mean(train_data[train_size - test_size:, 3])
# high = np.mean(range_high)
# low = np.mean(range_low)
# pred1 = []
# for i in range(test_size):
#     new = np.zeros((1, 4))
#     open = prev_close
#     new[0, 0] = open
#     new[0, 1] = open + high
#     new[0, 2] = open - low
#     new[0, 3] = volume + np.random.randint(-1000000, 1000000)
    
#     prev_close = svr.predict(new)[0]
#     pred1.append(prev_close)

pred2 = svr.predict(test_data)

x = [i for i in range(test_data.shape[0])]

# plt.plot(x, pred1, 'r')
plt.plot(x, pred2, 'g')
plt.plot(x, test_label)
plt.legend(["Model Prediction", "Actual Data"])
# plt.legend(["Model Pred with New Data", "Model Pred with Actual Data", "Actual Data"])
plt.show()
