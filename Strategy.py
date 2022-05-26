import numpy as np
from math import sqrt
from matplotlib import pyplot
from numpy import concatenate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from Data_process import DataHandler
import pickle5 as pickle
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import adam_v2
adam = adam_v2.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-06)


class StrategyHandler:

    def __init__(self):
        self.signal = {}
        self.factor_num = 5
        self.history = None
        self.y_hat = pd.DataFrame()

        # 全连接神经网络
        self.model = Sequential()  # 初始化
        self.model.add(Dense(16, input_dim=self.factor_num))
        self.model.add(Activation('tanh'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(4))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(1))
        # 最小均方误差损失函数
        self.model.compile(loss='mean_squared_error', optimizer=adam)

    def train_model(self, train_x, train_y, set_epochs, set_batch_size):
        self.history = self.model.fit(train_x, train_y, epochs=set_epochs, batch_size=set_batch_size)

        return self.history

    def plot_loss(self):
        pyplot.plot(self.history.history['loss'], label='train')
        pyplot.title('Model loss')
        pyplot.ylabel('Loss')
        pyplot.xlabel('Epoch')
        pyplot.legend()
        pyplot.show()

    def predict(self, test_x):
        self.y_hat = self.model.predict(np.array(test_x.dropna()))

        return self.y_hat

    def create_signal(self,  test_x):
        line = np.percentile(np.array(self.y_hat), 0.2)
        j = 0    # 特征存在缺失值时，预测数据与原数据指针有差异，需要两个指针
        for i in range(len(test_x.index)):
            a = np.array(test_x.iloc[i, :])
            if np.isnan(a).any():
                self.signal[test_x.index[i]] = 0
            elif self.y_hat[j] > line:
                self.signal[test_x.index[i]] = 1
                j += 1
            else:
                self.signal[test_x.index[i]] = 0
                j += 1
        print(self.signal)
        return self.signal

    # def evaluate_model(self):
    #     # 预测y 逆标准化
    #     inv_yhat0 = concatenate((test_x, yhat), axis=1)
    #     inv_yhat1 = scaler.inverse_transform(inv_yhat0)
    #     inv_yhat = inv_yhat1[:, -1]
    #
    #     # 原始y逆标准化
    #     test_y = test_y.reshape(len(test_y), 1)
    #     inv_y0 = concatenate((test_x, test_y), axis=1)
    #     inv_y1 = scaler.inverse_transform(inv_y0)
    #     inv_y = inv_y1[:, -1]
    #
    #     # 计算 R2
    #     r_2 = r2_score(inv_y, inv_yhat)
    #     print('Test r_2: %.3f' % r_2)
    #     # 计算MAE
    #     mae = mean_absolute_error(inv_y, inv_yhat)
    #     print('Test MAE: %.3f' % mae)
    #     # 计算RMSE
    #     rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    #     print('Test RMSE: %.3f' % rmse)
    #     plt.plot(inv_y)
    #     plt.plot(inv_yhat)
    #     plt.show()


