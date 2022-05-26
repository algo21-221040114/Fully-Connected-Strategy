import numpy as np
import pandas as pd
from Data_process import DataHandler
from Strategy import StrategyHandler
from Portfolio import PortfolioHandler
import datetime

test_time_list = ['2016-02-29 00:00:00', '2016-03-01 00:00:00', '2016-03-02 00:00:00', '2016-03-03 00:00:00',
                  '2016-03-04 00:00:00', '2016-03-07 00:00:00', '2016-03-08 00:00:00', '2016-03-09 00:00:00',
                  '2016-03-10 00:00:00', '2016-03-11 00:00:00', '2016-03-14 00:00:00', '2016-03-15 00:00:00',
                  '2016-03-16 00:00:00', '2016-03-17 00:00:00', '2016-03-18 00:00:00', '2016-03-21 00:00:00',
                  '2016-03-22 00:00:00', '2016-03-23 00:00:00', '2016-03-24 00:00:00', '2016-03-25 00:00:00',
                  '2016-03-28 00:00:00', '2016-03-29 00:00:00', '2016-03-30 00:00:00', '2016-03-31 00:00:00',
                  '2016-04-01 00:00:00', '2016-04-05 00:00:00', '2016-04-06 00:00:00', '2016-04-07 00:00:00',
                  '2016-04-08 00:00:00', '2016-04-11 00:00:00', '2016-04-12 00:00:00']
wealth = [1000000000.0000132, 1022839051.6808819, 1075341449.78583, 1079268650.358933, 1046838605.8867071, 1071622018.642652, 1049307542.9926988, 1020688607.8060882, 1002038306.3912252, 1000132338.6186258, 1034428237.706901, 999410337.75486, 980648632.3257513, 1009544445.941632, 1045618402.7338512, 1057439554.5631155, 1051904985.2830572, 1061569761.8408483, 1044202561.9357213, 1053041466.8032745, 1045486535.6583232, 1024599477.1023475, 1059216223.1122378, 1058266681.824532, 1054195350.6875567, 1081312183.3866758, 1084770145.647988, 1067496349.5181464, 1055181622.9259629, 1077917655.0216942, 1069133042.6894827]
simple_ret = []


class Performance:

    def __init__(self, wealth1, period1):
        self.ret = []
        self.cumulative_ret = [1]
        self.wealth = wealth1
        self.period = period1


    def annulized(self):
        a = self.cumulative_ret[-1]
        print(a)
        annualized_ret = a**(252/self.test_length)
        print('Annualized return is ' + str(annualized_ret))

    def sharpe(self):
        a = self.cumulative_ret[-1]
        mu = a ** (252 / self.test_length)-1 - 0.035
        sigma = np.std(self.ret)*np.sqrt(252/self.hold_length)
        print('Sharpe ratio is ' + str(mu/sigma))

    def maxdrawdown(self):
        mdd = -1
        for i in range(len(self.cumulative_ret)-1):
            for j in range(i+1, len(self.cumulative_ret)):
                if self.cumulative_ret[j] < self.cumulative_ret[i]:
                    mdd = max(mdd, 1-self.cumulative_ret[j]/self.cumulative_ret[i])
                else:
                    break
        print('Max Drawdown is ' + str(mdd))

    def run(self):
        ExecutionHandler.first_day_trade(self)
        ExecutionHandler.trade(self)
        ExecutionHandler.annulized(self)
        ExecutionHandler.sharpe(self)
        ExecutionHandler.maxdrawdown(self)
