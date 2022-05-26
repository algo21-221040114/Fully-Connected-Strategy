import pickle5 as pickle
import numpy as np
import pandas as pd
from Data_process import DataHandler
from Strategy import StrategyHandler

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
data_price = data_price.fillna(0)


data_rate = data_price.copy()
for i in range(data_rate.shape[1]-1):
    data_rate.iloc[:, i] = data_price.iloc[:, i+1]/data_price.iloc[:, i]-1
# print(data_rate)

data_input = open('/Users/wyb/PycharmProjects/Fully-Connected-Strategy/Fully-Connected-Strategy/Database/stock_st_standard', 'rb')
data_st = pickle.load(data_input)
data_st = pd.DataFrame(data_st)
# print(data_st.shape)


class PortfolioHandler:

    def __init__(self):
        self.net_value = 0
        self.single_net_value = {}
        self.cash = 0
        self.time = ''
        self.position = {}
        self.signal = {}
        self.cnt = 0
        self.commission = 5e-04
        self.close = {}

    def cnt_buy(self, signal):
        self.cnt = 0
        self.signal = signal
        for key in self.signal:
            self.cnt += int(self.signal[key])

        return self.cnt

    def trade(self, position, close, cash, time):
        self.cash = cash
        self.position = position
        self.close = close
        self.time = time
        # sell
        for key in self.signal:
            if self.signal[key] == 0 and self.position[key] > 0:
                self.cash += self.position[key] * self.close[key]
                self.position[key] = 0
                print(self.time + ' sell ' + str(key) + ' at ' + str(self.close[key]))
        # buy
        for key in self.signal:
            if self.signal[key] == 1 and self.position[key] == 0:
                self.position[key] = self.cash/self.cnt/self.close[key]
                print(self.time + ' buy ' + str(key) + ' at ' + str(self.close[key]))

        return self.position

    def show_position(self):

        return self.position

    def create_net_value(self):
        self.net_value = 0
        for key in self.position:
            self.net_value += self.position[key]*self.close[key]

        return self.net_value

    def create_single_net_value(self):
        for key in self.position:
            self.single_net_value[key] = self.position[key]*self.close[key]

        return self.single_net_value


test_time_list = ['2016-02-29 00:00:00', '2016-03-01 00:00:00', '2016-03-02 00:00:00', '2016-03-03 00:00:00',
                  '2016-03-04 00:00:00', '2016-03-07 00:00:00', '2016-03-08 00:00:00', '2016-03-09 00:00:00',
                  '2016-03-10 00:00:00', '2016-03-11 00:00:00', '2016-03-14 00:00:00', '2016-03-15 00:00:00',
                  '2016-03-16 00:00:00', '2016-03-17 00:00:00', '2016-03-18 00:00:00', '2016-03-21 00:00:00',
                  '2016-03-22 00:00:00', '2016-03-23 00:00:00', '2016-03-24 00:00:00', '2016-03-25 00:00:00',
                  '2016-03-28 00:00:00', '2016-03-29 00:00:00', '2016-03-30 00:00:00', '2016-03-31 00:00:00',
                  '2016-04-01 00:00:00', '2016-04-05 00:00:00', '2016-04-06 00:00:00', '2016-04-07 00:00:00',
                  '2016-04-08 00:00:00', '2016-04-11 00:00:00', '2016-04-12 00:00:00']
data = DataHandler()
strategy = StrategyHandler()
portfolio = PortfolioHandler()
init_wealth = 1000000000
init_position = {}

t_list = data.create_time(data_factor)
s_list = data.create_symbol(data_factor)
for s in s_list:
    init_position[s] = 0

simple_ret = []
wealth = []

for i in range(len(test_time_list)):
    time = test_time_list[i]
    test_x1 = data.create_test(data_factor, time)
    train_x1 = np.array(data.create_train(data_factor, time))
    train_y1 = np.array(data.create_train(data_factor, time))
    close = data_price[time]
    close_dirt = {}
    for j in range(len(data_price.index)):
        close_dirt[data_price.index[j][0]] = close[j]

    # 若未发生滚动，则无需重新训练数据
    t = t_list.index(time)
    if i == 0 or t_list[t - 1][5:7] is not t_list[t][5:7]:
        strategy.train_model(train_x1, train_y1, set_epochs=5, set_batch_size=705)
    strategy.predict(test_x1)
    signal_dirt = strategy.create_signal(test_x1)
    portfolio.cnt_buy(signal_dirt)
    if i == 0:
        portfolio.trade(init_position, close_dirt, init_wealth, time)
        last_position = portfolio.show_position()
    else:
        portfolio.trade(last_position, close_dirt, 0, time)
    wealth.append(portfolio.create_net_value())

print(wealth)










