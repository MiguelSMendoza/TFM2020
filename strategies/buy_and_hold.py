from zipline.api import order, symbol, record, get_datetime
from matplotlib import pyplot as plt
import pandas as pd
from joblib import load
import numpy as np
import csv

def write_to_csv(row):
    with open("BuyNHoldX.csv", "a") as outfile:
        writer = csv.writer(outfile, delimiter = ";")
        writer.writerow(row)

class BuyAndHold:

    stocks = ['AAPL', 'MSFT']

    def initialize(self, context):
        context.has_ordered = False
        context.stocks = self.stocks
        context.asset = symbol('AAPL')
        context.regressor = load('./strategies/models/rf_regressor.joblib')

    def handle_data(self, context, data):
        if context.portfolio.cash > 0:
            for stock in context.stocks:
                price = data.current(symbol(stock), 'price')
                timeseries = data.history(
                    symbol(stock),
                    'price',
                    bar_count = 33,
                    frequency = '1d')
                historical_mean = np.mean(timeseries)

                if price < historical_mean:
                    order(symbol(stock), 10)
                    write_to_csv([get_datetime(), stock, data.current(symbol(stock), 'price'), context.portfolio.cash])

    def _test_args(self):
        return {
            'start': pd.Timestamp('2008', tz='utc'),
            'end': pd.Timestamp('2018', tz='utc'),
            'capital_base': 1e4
        }

    def analyze(self, context, perf):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        perf.portfolio_value.plot(ax=ax1)
        ax1.set_ylabel('Valor del portafolio en $')
        plt.legend(loc=0)
        plt.show()
