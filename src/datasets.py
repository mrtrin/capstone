import numpy as np
import pandas as pd
import yfinance as yf

def load_train_test(lookback_period=10, rebalance_period=5):
    tickers = ['AAPL', 'MSFT', 'GOOG', 'META', 'TSLA', 'SPY']
    dataset = Dataset(tickers, '2010-01-01', '2023-12-31')
    features = dataset.features()
    targets = dataset.targets(rebalance_period)
    X, y = dataset.create_training_set(features, targets, lookback_period)
    return X, y

class Dataset:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.download(tickers, start=start_date, end=end_date)
        self.data = self._clean(self.data)

    def _clean(self, df):
        return df.fillna(method='ffill')

    def features(self, stoch_lookback=15):
        # TODO: Add Features based on technical signals
        
        close = self.data['Adj Close']
        close.columns = [f'{col}_close' for col in close.columns]
        volume = self.data['Volume']
        volume.columns = [f'{col}_volume' for col in volume.columns]

        features = []
        for ticker in close.columns:
            df = pd.DataFrame(close[ticker])
            df.columns = ['close']
            rsi = df.ta.rsi()
            macd = df.ta.macd()
            bbands = df.ta.bbands()

            stoch = df.copy()
            stoch['high'] = stoch['close'].rolling(stoch_lookback).max()
            stoch['low'] = stoch['close'].rolling(stoch_lookback).min()
            stoch.ta.stoch(append=True)
            del stoch['close']
            del stoch['high']
            del stoch['low']

            df = pd.concat([rsi, macd, bbands, stoch], axis=1)
            df.columns = [f'{ticker}_{col}' for col in df.columns]
            features.append(df)

        return pd.concat(features, axis=1)
    
    def targets(self, prediction_period):
        period_returns = (self.data['Adj Close'] / self.data['Adj Close'].shift(prediction_period)) - 1
        period_returns[period_returns < 0] = 0
        weights = period_returns.div(period_returns.sum(axis=1), axis=0)
        weights = weights.shift(-1*prediction_period)
        return weights

    def create_training_set(self, features, targets, lookback, dayofweek=4):
        # This is coded to create weekly rebalance on Friday
        X, y = [], []
        for i in range(len(features)-lookback):
            if features.index[i].dayofweek == dayofweek: # Friday
                X.append(features[i:i+lookback])
                y.append(targets[i+lookback:i+lookback+1])
        return np.array(X), np.array(y).reshape(len(y), targets.shape[1])
