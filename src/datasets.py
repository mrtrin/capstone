from pandas_ta import Imports

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import yfinance as yf


def load_train_test(lookback_period=10, rebalance_period=5):
    tickers = ['AAPL', 'MSFT', 'GOOG', 'META', 'TSLA', 'SPY']
    dataset = Dataset(tickers, '2010-01-01', '2023-12-31')
    features, closes = dataset.features(scale=True)
    targets = dataset.targets(rebalance_period)
    X, y, y_price = dataset.create_training_set(features, targets, closes, lookback_period)
    
    return dataset.data, features, targets, X.astype(np.float32), y.astype(np.float32), y_price.astype(np.float32)

class Dataset:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.download(tickers, start=start_date, end=end_date)
        print('Downloaded Data Size', self.data.shape)
        self.data = self._clean(self.data)
        print('Cleaned Data Size', self.data.shape)

    def _clean(self, df):
        return df.fillna(method='ffill').dropna()

    def features(self, scale=True, stoch_lookback=15):
        # TODO: Add Features based on technical signals
        
        closes = self.data['Adj Close']
        volumes = self.data['Volume']

        features = []
        for ticker in closes.columns:
            close = pd.DataFrame(closes[ticker])
            close.columns = ['close']
            rsi = close.ta.rsi()
            macd = close.ta.macd()
            bbands = close.ta.bbands()

            stoch = close.copy()
            stoch['high'] = stoch['close'].rolling(stoch_lookback).max()
            stoch['low'] = stoch['close'].rolling(stoch_lookback).min()
            stoch.ta.stoch(append=True)
            del stoch['close']
            del stoch['high']
            del stoch['low']

            vol = pd.DataFrame(volumes[ticker], columns=['volume'])
            print('111', vol.columns)

            df = pd.concat([close, vol, rsi], axis=1)
            df.columns = [f'{ticker}_{col}' for col in df.columns]
            features.append(df)

        df = pd.concat(features, axis=1)
        non_price_volumes = [c for c in df.columns if not ('_close' in c or '_vol' in c)]
        df[non_price_volumes] = MinMaxScaler(feature_range=(0, 1)).fit_transform(df[non_price_volumes])
        return df.fillna(method='ffill'), closes
    
    def targets(self, prediction_period):
        period_returns = (self.data['Adj Close'] / self.data['Adj Close'].shift(prediction_period)) - 1
        period_returns[period_returns < 0] = 0
        weights = period_returns.div(period_returns.sum(axis=1), axis=0)
        weights = weights.shift(-1*prediction_period)
        return weights

    def create_training_set(self, features, targets, closes, lookback, dayofweek=4):
        # This is coded to create weekly rebalance on Friday
        print('0000000000000')
        print(features.shape)
        print(targets.shape)
        print(closes.shape)
        X, y, y_price = [], [], []
        for i in range(len(features)-lookback):
            if features.index[i].dayofweek == dayofweek: # Friday
                X.append(features[i:i+lookback])
                y.append(targets[i+lookback:i+lookback+1])
                y_price.append(closes[i+lookback:i+lookback+1])
        return np.array(X), np.array(y).reshape(len(y), targets.shape[1]), np.array(y_price).reshape(len(y_price), targets.shape[1])
