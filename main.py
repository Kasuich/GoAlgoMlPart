from TrainModel import TrainModel
from Backtest import Backtest

tick = 'SBER'
period = '10m'
train_candles = 10_000
backtest_candles = 1_000

features = {'lags': {'features': ['open', 'close', 'target'],
                     'period': [1, 2, 3]},

            'cma': {'features': ['open', 'close', 'volume']},

            'sma': {'features': ['open', 'close', 'volume'],
                    'period': [2, 3, 4]},

            'ema': {'features': ['open', 'close', 'volume'],
                    'period': [2, 3, 4]},

            'green_candles_ratio': {'period': [2]},

            'red_candles_ratio': {'period': [2]},

            'rsi': False,

            'macd': False, # только (12, 26)

            'bollinger': False,

            'time_features': {'month':True,
                              'week':True,
                              'day_of_month':True,
                              'day_of_week':True,
                              'hour':True,
                              'minute': True},
            'model': 'catboost'} # выбор один из 'lightgbm', 'tabular_learner'



train_model = TrainModel(features, tick, period, candles=train_candles)
train_model.train()

management_features = {"balance": 100_000,
                       "max_balance_for_trading": 200_000,
                       "min_balance_for_trading": 0,
                       "part_of_balance_for_buy": 0.3,
                       "sum_for_buy_rur": None,
                       "sum_for_buy_num": None,
                       "part_of_balance_for_sell": None,
                       "sum_for_sell_rur": None,
                       "sum_for_sell_num": None,
                       "sell_all": True}

backtest = Backtest(features,
                    balance=management_features['balance'],
                    max_balance_for_trading=management_features['max_balance_for_trading'],
                    min_balance_for_trading=management_features['min_balance_for_trading'],
                    part_of_balance_for_buy=management_features['part_of_balance_for_buy'],
                    sum_for_buy_rur=management_features['sum_for_buy_rur'],
                    sum_for_buy_num=management_features['sum_for_buy_num'],
                    part_of_balance_for_sell=management_features['part_of_balance_for_sell'],
                    sum_for_sell_rur=management_features['sum_for_sell_rur'],
                    sum_for_sell_num=management_features['sum_for_sell_num'],
                    sell_all=management_features['sell_all'])

balance = backtest.do_backtest(tick, period, backtest_candles)
print(balance)