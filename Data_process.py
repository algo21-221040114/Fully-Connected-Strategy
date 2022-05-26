import pickle5 as pickle
import pandas as pd
import numpy as np

# 读取数据
data_input = open('/Users/wyb/PycharmProjects/Fully-Connected-Strategy/Fully-Connected-Strategy/Database/stock_valuation_standard', 'rb')
data_factor = pickle.load(data_input)
data_factor = pd.DataFrame(data_factor)
# 选取所需特征因子值
a = []
for i in range(1410):
    for num in [4, 5, 6, 8, 9]:
        a.append(i*10+num)
data_factor = pd.DataFrame(data_factor.iloc[a])

data_input = open('/Users/wyb/PycharmProjects/Fully-Connected-Strategy/Fully-Connected-Strategy/Database/stock_price_standard', 'rb')
data_price = pickle.load(data_input)
data_price = pd.DataFrame(data_price)
# 选取所需数据：收盘价
a = []
for i in range(1410):
    a.append(i*11)
data_price = pd.DataFrame(data_price.iloc[a])
data_rate = data_price.copy()
for i in range(data_rate.shape[1]-1):
    data_rate.iloc[:, i] = data_price.iloc[:, i+1]/data_price.iloc[:, i]-1
# print(data_rate)

data_input = open('/Users/wyb/PycharmProjects/Fully-Connected-Strategy/Fully-Connected-Strategy/Database/stock_st_standard', 'rb')
data_st = pickle.load(data_input)
data_st = pd.DataFrame(data_st)
# print(data_st.shape)


# 数据处理
class DataHandler:
    def __init__(self):
        self.symbol_list = []
        self.time_list = []
        self.train = pd.DataFrame()
        self.train_target = pd.DataFrame()
        self.test = pd.DataFrame()
        self.roll = 10   # 可调参数：训练数据集长度
        self.factor_num = 5  # 可调参数：所用因子数量

    def create_time(self, dataset):
        for i in range(len(dataset.columns)):
            self.time_list.append(str(dataset.columns[i]))

        return self.time_list

    def create_symbol(self, dataset):
        for i in range(len(dataset.index)):
            if i % 5 == 0:
                self.symbol_list.append(dataset.index[i][0])
        return self.symbol_list

    def create_train(self, dataset, time):
        """

        :param dataset: 因子数据表
        :param time: 预测日期
        :return: 所需训练数据特征， DataFrame (len(self.symbol_list) * self.roll, self.factor_num)
        """
        t = self.time_list.index(time)
        # 每个月更新训练数据，判断时间标签中月份位置是否变化
        if self.time_list[t - 1][5:7] is not self.time_list[t][5:7]:
            print('Update train set ' + time)
            self.train = np.zeros((len(self.symbol_list) * self.roll, self.factor_num))
            for k in range(len(self.symbol_list)):
                a = np.array(dataset.iloc[self.factor_num*k: self.factor_num*(k+1), t-self.roll: t])
                self.train[self.roll * k: self.roll * (k + 1), :] = a.T
        self.train = pd.DataFrame(self.train).dropna()

        return self.train

    def create_train_target(self, dataset, time):
        """

        :param dataset: 收益率数据集
        :param time: 预测日期
        :return: 所需训练数据标签， DataFrame (len(self.symbol_list) * self.roll, 1)
        """
        t = self.time_list.index(time)
        # 每个月更新训练数据，判断时间标签中月份的位置是否变化
        if self.time_list[t - 1][5:7] is not self.time_list[t][5:7]:
            print('Update train set ' + time)
            self.train_target = np.zeros((len(self.symbol_list) * self.roll, 1))
            for k in range(len(self.symbol_list)):
                a = np.array(dataset.iloc[k, t - self.roll: t]).reshape((1, self.roll))
                self.train_target[self.roll * k: self.roll * (k + 1), :] = a.T
        self.train_target = pd.DataFrame(self.train_target).dropna()

        return self.train_target

    def create_test(self, dataset, time):
        """

        :param dataset: 因子数据表
        :param time: 预测日期
        :return: 所需预测数据， DataFrame (len(self.symbol_list), self.factor_num)
        """
        t = self.time_list.index(time)
        self.test = np.zeros((len(self.symbol_list), self.factor_num))
        for k in range(len(self.symbol_list)):
            a = np.array(dataset.iloc[self.factor_num * k: self.factor_num * (k + 1), t])
            self.test[k, :] = a.T
        self.test = pd.DataFrame(self.test)
        self.test.index = self.symbol_list

        return self.test


data = DataHandler()
t_list = data.create_time(data_factor)
print(t_list)






