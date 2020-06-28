from zipline.api import order, symbol, record,order_target, get_datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import csv
#,STOCK,KIND,OPEN,STOP,LIMIT,BHI,BLW,MA1,MA2,LOW,VALUEOPEN,VALUECLOSE,RETURNS,RESULT
def write_to_csv(row):
        with open("DAX.csv", "a") as outfile:
           writer = csv.writer(outfile, delimiter = ",")
           writer.writerow(row)

class ScalpBollingerBand:

    stock = 'DAX'
    ma1 = 100
    ma2 = 200
    steps = 200
    stop_loss = 0.001
    stdv = 2

    def initialize(self, context):
        context.stock = self.stock
        context.burndown = 0
        context.number_shorts = 0
        context.number_longs = 0
        context.position = None
        context.open = 0
        context.limit = 0
        context.kind = None
        context.stop = 0
        context.value = 0
        context.time = None
        context.orders = {}
        context.current_order = 0

    def handle_data(self, context, data):

        # Contamos cada paso y esperamos a llegar al mínimo para comenzar
        context.burndown += 1

        if context.burndown > self.steps:

            # Obtenemos el precio actual (puede que para el minuto actual no esté disponible)
            try:
                current_price = data.current(symbol(context.stock), 'price')
            except:
                return
            record(PRICE = current_price)

            # Historial de valores
            hist = data.history(
                symbol(context.stock),
                'close',
                bar_count=20,
                frequency='1m')

            ewm = hist.ewm(span=20, ignore_na=True ).mean()[1]

            if ewm != ewm:
                return

            # Bollinger Bands
            blw = ewm - self.stdv * hist.std()
            bhi = ewm + self.stdv * hist.std()

            if ewm != ewm or bhi != bhi or blw != blw:
                return

            # Medias Móviles
            short_term = data.history(
                symbol(context.stock),
                'price',
                bar_count=self.ma1,
                frequency='1m').mean()

            long_term = data.history(
                symbol(context.stock),
                'price',
                bar_count=self.ma2,
                frequency='1m').mean()

            # Comprobamos indicadores
            if short_term > long_term and current_price > bhi and context.position != 'trade':
                context.position = 'short'
            elif short_term < long_term and current_price < blw and context.position != 'trade':
                context.position = 'long'

            high_price = data.current(symbol(context.stock), 'high')
            low_price = data.current(symbol(context.stock), 'low')
            open_price = data.current(symbol(context.stock), 'open')
            close_price = data.current(symbol(context.stock), 'close')
            stop_loss =  current_price * self.stop_loss

            # Comprobamos las Bollinger Bands
            if current_price <= bhi and context.position == 'short' and current_price - ewm >= 3:
                # Cálculamos la cantidad de acciones que nos podemos permitir
                num_shares = (context.portfolio.cash // current_price) // 3
                # Cálculamos el valor de stop loss
                stop = current_price + (current_price * self.stop_loss)
                # Abrimos una operación en CORTO
                order(asset=symbol(context.stock), amount=num_shares * -1)
                context.current_order += 1
                context.orders[context.current_order] = [get_datetime(), 'DAX', 'SHORT', current_price, stop_loss, ewm, bhi, blw, short_term, long_term, high_price, low_price, open_price, close_price]
                # Establecemos los valores contextuales de la operación necesarios para cerrarla
                context.position = 'trade'
                context.kind = 'short'
                context.number_shorts += 1
                context.stop = stop
                context.open = current_price
                context.limit = ewm
                context.time = context.burndown
                context.number_shorts = context.number_shorts + 1
                print('Enter Short: ', current_price, 'limit', context.limit, 'stop', stop)
            elif current_price >= blw and context.position == 'long' and ewm - current_price >= 3:
                # Cálculamos la cantidad de acciones que nos podemos permitir
                num_shares = (context.portfolio.cash // current_price) // 3
                # Cálculamos el valor de stop loss
                stop = current_price - (current_price * self.stop_loss)
                # Abrimos una operación en LARGO
                order(asset=symbol(context.stock), amount=num_shares)   # order_value
                context.current_order += 1
                context.orders[context.current_order] = [get_datetime(), 'DAX', 'LONG', current_price, stop_loss, ewm, bhi, blw, short_term, long_term, high_price, low_price, open_price, close_price]
                # Establecemos los valores contextuales de la operación necesarios para cerrarla
                context.position = 'trade'
                context.kind = 'long'
                context.number_longs += 1
                context.stop = stop
                context.open = current_price
                context.limit = ewm
                context.time = context.burndown
                context.number_longs = context.number_longs + 1
                print('Enter Long: ', current_price, 'limit', context.limit, 'stop', stop)

            if (current_price >= context.limit and context.position == 'trade' and context.kind == 'long') \
            or (current_price <= context.limit and context.position == 'trade' and context.kind == 'short') \
            or (context.position == 'trade' and (context.burndown - context.time) > 600):
                order(symbol(context.stock), 0)
                result = 0
                if context.kind == 'long':
                    result = current_price - context.open
                elif context.kind == 'short':
                    result = context.open - current_price
                context.value += result
                context.position = context.kind
                context.limit = 0
                context.kind = None
                context.stop = 0
                context.open = 0
                context.time = 0
                record(RETURNS= context.value )
                context.orders[context.current_order].append(current_price)
                context.orders[context.current_order].append(result)
                context.orders[context.current_order].append(get_datetime())
                context.orders[context.current_order].append('OK')
                write_to_csv(context.orders[context.current_order])
                print('Exit Trade: ', current_price, context.value, context.portfolio.cash )
            if (current_price < context.stop and context.position == 'trade' and context.kind == 'long') \
            or (current_price > context.stop and context.position == 'trade' and context.kind == 'short'):
                order(symbol(context.stock), 0)
                result = abs(context.open-context.stop)
                context.value -= result
                context.position = context.kind
                context.limit = 0
                context.kind = None
                context.stop = 0
                context.open = 0
                context.time = 0
                context.orders[context.current_order].append(current_price)
                context.orders[context.current_order].append(-result)
                context.orders[context.current_order].append(get_datetime())
                context.orders[context.current_order].append('STOP')
                write_to_csv(context.orders[context.current_order])
                record(RETURNS= context.value )
                print('Stop: ', current_price, context.value, context.portfolio.cash )


    def _test_args(self):
        return {
#            'start': pd.Timestamp('2017-09-18', tz='utc'),
#            'end': pd.Timestamp('2018-03-12', tz='utc'),
#             'capital_base': 1e7,
#             'data_frequency': 'minute'
        }

    def analyze(self, context, perf):
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        perf.plot(y=[
            'PRICE'
            ], ax=ax1)
        ax1.set_ylabel('Precio')
        perf.plot(y=['RETURNS'],ax=ax2)
        ax2.set_ylabel('Valor Acumulado')

        plt.legend(loc=0)
        plt.show()
