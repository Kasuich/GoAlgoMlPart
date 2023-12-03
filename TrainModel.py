from __future__ import annotations
from typing import Dict, List, Iterable

import pandas as pd
import numpy as np

from SimpleDataset import SimpleDataset

import dill

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import catboost
import lightgbm
from sklearn.metrics import mean_squared_error
from fastai.tabular.all import *
import warnings
import lightgbm as lgb
warnings.filterwarnings('ignore')


CATBOOST = catboost.core.CatBoostRegressor
LGBM = lightgbm.sklearn.LGBMRegressor

class TrainModel():

  def __init__(self, features, ticker, timeframe, random_seed=42, candles=10_000):

    self.seed = random_seed
    self.timeframe = timeframe
    self.ticker = ticker
    self.features = features
    self.model = self.features['model']
    self.save_path = f'{12345678}_{self.ticker}_{self.timeframe}_{self.model}.bin' # Путь для сохранения модели, 12345678 - хэш юзера
    self.order = []
    self.candles = candles
    self.raw_dataset = SimpleDataset.create_dataset(features=features, ticker=self.ticker, timeframe=self.timeframe, candles=self.candles)

  def train(self,
            test_size: float = 0.1,
            date_col: str = 'date',
            target_col: str = 'target'):

    train_df = self.raw_dataset.copy()
    Xtrain, Xtest, ytrain, ytest = train_test_split(train_df.drop(columns=[target_col, date_col]),
                                                    train_df[target_col].fillna(train_df[target_col].mean()),
                                                    test_size=test_size,
                                                    shuffle=False)
    self.order = Xtrain.columns.values

    if self.features['model'] == 'catboost':
      self.model = CatBoostRegressor(eval_metric = 'RMSE', random_seed = self.seed)
      self.model.fit(Xtrain, ytrain, eval_set = (Xtest, ytest), plot = False, verbose = False)
      print(f'CatBoost RMSE score on validation set: {mean_squared_error(ytest, self.model.predict(Xtest), squared = False)}')
      self.model.save_model(self.save_path)

    if self.features['model'] == 'lightgbm':
      self.model = LGBMRegressor(random_state = self.seed)
      self.model.fit(Xtrain, ytrain, eval_set = (Xtest, ytest), eval_metric = 'RMSE')
      print(f'LGBM RMSE score on validation set: {mean_squared_error(ytest, self.model.predict(Xtest), squared = False)}')
      self.model.booster_.save_model(self.save_path)

    if self.features['model'] == 'tabular_learner':

      splits = (L(range(int(len(train_df) * test_size), len(train_df))), L(range(int(len(train_df) * test_size))))

      self.to = TabularPandas(train_df.drop(columns = [date_col]),
                              cat_names = [],
                              cont_names = [i for i in train_df.drop(columns = [date_col, target_col]).columns],
                              y_names = target_col,
                              splits = splits,
                              y_block = RegressionBlock())

      self.dls = self.to.dataloaders(bs = 64)
      self.model = tabular_learner(self.dls, metrics = rmse)
      self.model.fit_one_cycle(10)

      test_dl = self.dls.test_dl(Xtest)
      test_preds, _ = self.model.get_preds(dl = test_dl)
      print(f'Table Loader RMSE score on validation set: {mean_squared_error(ytest, test_preds, squared = False)}')
      self.model.export(self.save_path, pickle_module=dill)

    return "successful"