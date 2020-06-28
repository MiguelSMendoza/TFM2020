from zipline.api import order, symbol, record,order_target, get_datetime
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import norm
import csv
from joblib import load
from Cython.Shadow import returns
from statsmodels.tsa.tests.results.results_arima import forecast_results

def write_to_csv(row):
        with open("DAX-SVR.csv", "a") as outfile:
           writer = csv.writer(outfile, delimiter = ";")
           writer.writerow(row)

class ScalpBollingerBandSVR:

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
        context.last = 0

        # VaR
        context.historical_changes = []

    def handle_data(self, context, data):

        context.burndown += 1
        value_at_risk = None

        # Obtenemos el precio actual (puede que para el minuto actual no esté disponible)
        try:
            current_price = data.current(symbol(context.stock), 'price')
            if context.last == 0:
                context.last = current_price
        except:
            return
        record(PRICE = current_price)

        if len(context.historical_changes) > 1440 and context.position == 'trade':
            # Obtenemos los cambios porcentuales del último día
            historical_changes = np.array(context.historical_changes[−1440:])
            # Predecimos valores futuros tomando los últimos 15 minutos del histórico
            forecast_changes = context.regressor.predict(
                historical_changes[−15:].reshape(1, −1)
            )
            # Añadimos los nuevos valores al conjunto de datos
            historical_changes = np.concatenate (
                (
                    historical_changes,
                    forecast_changes [0]
                ), axis=0 
            )
            # Calculamos la media y la desviación estándar de los datos obtenidos
            mean = np.mean(historical_changes) std_dev = np.std(historical_changes)
            # Obtenemos el cuantil con un alfa de 0.95
            normal_returns = np . random . normal (mean , std_dev , 1440)
            lowest_percentile = np.percentile( normal_returns, 1−0.95 ) # Calculamos el ES y lo utilizamos como VaR
            expected_shortfall = np.mean(historical_changes[historical_changes<=lowest_percentile]) value_at_risk = expected_shortfall

            # Calculamos la variación porcentual entre el precio actual y el de apertura
            change = (( current_price - context.open ) / context.open ) * 100

            if  ( context.kind == 'long' and change <= value_at_risk) or ( context.kind == 'short' and (change-value_at_risk) > 0 ) :
                # Cerramos la operación
                order(symbol(context.stock), 0)
                # Calculamos el resultado de la operación
                returns = abs( context.open-current_price )
                result = returns
                if (current_price < context.open and context.kind == 'long') \
                or (current_price > context.open and context.kind == 'short'):
                    result = - returns
                context.value = context.value + result
                # Reiniciamos valores contextuales
                context.position = None
                context.limit = 0
                context.kind = None
                context.stop = 0
                context.open = 0
                context.time = 0
                context.orders[context.current_order].append(current_price)
                context.orders[context.current_order].append(result)
                context.orders[context.current_order].append(get_datetime())
                context.orders[context.current_order].append('STOP')
                write_to_csv(context.orders[context.current_order])
                # Imprimimos la información correspondiente
                print('Stop Loss: {} price {} @ VaR {} {} for {} at {}'.format(
                    context.value,
                    current_price,
                    value_at_risk,
                    change,
                    context.stock,
                    get_datetime()
                    )
                )

        if context.burndown > self.steps:

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
            if context.burndown % 1000 == 0:
                print(context.burndown, current_price, ewm, bhi, blw, short_term, long_term)

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
                # Calculamos la cantidad de acciones que nos podemos permitir
                num_shares = (context.portfolio.cash // current_price) // 3
                # Calculamos el valor de stop loss
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

            sw = False
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
                context.orders[context.current_order].append(current_price)
                context.orders[context.current_order].append(result)
                context.orders[context.current_order].append(get_datetime())
                context.orders[context.current_order].append('OK')
                write_to_csv(context.orders[context.current_order])
                print('Exit Trade: ', current_price, context.value, context.portfolio.cash )
            record(RETURNS= context.value )
            
            change = ( current_price - context.last ) / context.last * 100
            if change == change:
                context.historical_changes.append( change )
            context.last = current_price

    def _test_args(self):
        return {
#             'start': pd.Timestamp('2017', tz='utc'),
#             'end': pd.Timestamp('2018', tz='utc'),
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
        ax1.set_ylabel('Valor')
        perf.plot(y=['RETURNS'],ax=ax2)
        ax2.set_ylabel('Beneficios')
        plt.legend(loc=0)
        plt.show()
