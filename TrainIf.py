from SimpleDataset import SimpleDataset
import pandas as pd

class TrainIf():

    def __init__(self, features, ticker, timeframe, candles=10_000, notebook=False):

        self.features = features
        self.ticker = ticker
        self.timeframe = timeframe
        self.candles = candles
        self.notebook = notebook

        self.raw_dataset = SimpleDataset.create_dataset(
            self.features,
            self.ticker,
            self.timeframe,
            self.candles,
            self.notebook
        )

    def train(self, value_col: str = 'value', pct_col: str = 'target'):

        if self.features['anomal_value']:
            sigma_val = self.raw_dataset[value_col].std()
        if self.features['anomal_price_changing']:
            sigma_changing = self.raw_dataset[pct_col].std()
        
        self.features['anomal_value'] = 3 * sigma_val
        self.features['anomal_price_changing'] = 3 * sigma_changing

        return self.features