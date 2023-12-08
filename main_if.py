import pandas as pd
import numpy as np

from IfInference import IfInference

ticker = "YNDX"
timestamp = "1m"

IF_features_example = [
    {
        "type": "and", # tp.Literal["and", "if"]
        "blocks": [
            {
                "type": "if",
                "feature": "anomaly",
                "condition": "high", # tp.Literal["high", "low"]
                "param": "value", # tp.Literal["value", "price_changing"]
            },
            {
                "type": "if",
                "feature": "anomal_rsi",
                "param": {
                    "period": 10, # tp.Literal[2, 5, 10, 15, 20, 30, 50]
                    "value": 80 # tp.Literal[50, 55, 60, 65, 70, 75, 80, 85, 90]
                }
            },
            {
                "type": "if",
                "feature": "out_of_limits",
                "condition": "high", # tp.Literal["high", "low"]
                "param": {
                    "feature_name": "close", # tp.Literal["close", "high", "low", "open", "value", "volume", "green_candles_ratio", "red_candles_ratio", "price_changing"]
                    "limit": 277.7, # float
                    "period": None, # int | None
                }
            },
            {
                "type": "if",
                "feature": "average_cross",
                "param": {
                    "average_type": "ema", # tp.Literal["ema", "sma", "cma"],
                    "feature_name": "close", # tp.Literal["close", "high", "low", "open", "value", "volume", "green_candles_ratio", "red_candles_ratio"]
                    "n_fast": 10, # tp.Literal[2, 5, 10, 15, 50, 100]
                    "n_slow": 100, # tp.Literal[2, 5, 10, 15, 50, 100]
                }
            },
            {
                "type": "if",
                "feature": "macd_cross",
                "param": {
                    "feature_name": "close", # tp.Literal["close", "high", "low", "open", "value", "volume", "green_candles_ratio", "red_candles_ratio"]
                    "n_fast": 5, # tp.Literal[2, 5, 10, 15, 50, 100]
                    "n_slow": 50, # tp.Literal[2, 5, 10, 15, 50, 100]
                }
            }
        ]
    },
    {
        "type": "and",
        "blocks": [
            {
                "type": "if",
                "feature": "macd_cross",
                "param": {
                    "feature_name": "close", # tp.Literal["close", "high", "low", "open", "value", "volume", "green_candles_ratio", "red_candles_ratio"]
                    "n_fast": 50,
                    "n_slow": 100,
                }
            }
        ]
    }
]

if_model = IfInference(
    IF_features=IF_features_example,
    ticker=ticker,
    timestamp=timestamp,
)

last_100_candles, signals = if_model.predict_n_last_candles(
    candles=100
)

print(last_100_candles.date.max(), len(signals), signals.sum())

last_1_candles, signals = if_model.predict_one_last_candle()

print(last_1_candles.date.max(), signals)

