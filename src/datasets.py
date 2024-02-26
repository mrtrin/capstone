from pandas_ta import Imports

from sklearn.preprocessing import MinMaxScaler

import numpy as np
import pandas as pd
import yfinance as yf


def load_train_test(tickers, start, end, lookback_period=10, rebalance_period=5):
    dataset = Dataset(tickers, start, end)
    features = dataset.features(scale=True)
    targets = dataset.targets(features, rebalance_period)
    X, y, y_price = dataset.create_training_set(features, targets, lookback_period)
    
    return dataset.data, features, targets, X.astype(np.float32), y.astype(np.float32), y_price

class Dataset:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._download(tickers, start_date, end_date)
        self.data = self._clean(self.data)
        self.data = self._add_cash(self.data)
    
    def _download(self, tickers, start_date, end_date):
        return yf.download(tickers, start=start_date, end=end_date)

    def _clean(self, df):
        return df.fillna(method='ffill').dropna()
    
    def _add_cash(self, df, daily_rates=.01/365):
        cash = (pd.Series(1, index=df.index) * (1+daily_rates)).cumprod()
        columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        df_cash = pd.DataFrame({(c,'CASH'): cash for c in columns})
        df_cash.columns = pd.MultiIndex.from_tuples(df_cash.columns)
        return pd.concat([df, df_cash], axis=1)

    def features(self, stoch_lookback=15, scale=True):
        df = self.data.swaplevel(0, 1, 1)

        scale_min = 0.01
        scale_max = 1.00

        for ticker in df.columns.levels[0]:
            rsi = df[ticker].ta.rsi()
            rsi = pd.DataFrame(rsi.values, index=rsi.index, columns=['RSI'])
            rsi = pd.DataFrame(MinMaxScaler((scale_min, scale_max)).fit_transform(rsi), index=rsi.index, columns=rsi.columns)
            df[(ticker,'RSI')] = rsi

            macd = df[ticker].ta.macd()
            macd = pd.DataFrame(MinMaxScaler((scale_min, scale_max)).fit_transform(macd), index=macd.index, columns=macd.columns)
            for c in macd.columns:
                df[(ticker, c)] = macd[c]

            bbands = df[ticker].ta.bbands()
            bbands = pd.DataFrame(MinMaxScaler((scale_min, scale_max)).fit_transform(bbands), index=bbands.index, columns=bbands.columns)
            for c in bbands.columns:
                df[(ticker, c)] = bbands[c]

            stoch = df[ticker].ta.stoch(append=True)
            stoch = pd.DataFrame(MinMaxScaler((scale_min, scale_max)).fit_transform(stoch), index=stoch.index, columns=stoch.columns)
            for c in stoch.columns:
                df[(ticker, c)] = stoch[c]

            obv = df[ticker].ta.obv()
            obv = pd.DataFrame(obv.values, index=obv.index, columns=['OBV'])
            obv = pd.DataFrame(MinMaxScaler((scale_min, scale_max)).fit_transform(obv), index=obv.index, columns=obv.columns)
            df[(ticker,'OBV')] = obv
        return df.ffill().dropna(axis=0).swaplevel(0, 1, 1)
    
    def targets(self, df, prediction_period):
        period_returns = (df['Adj Close'] / df['Adj Close'].shift(prediction_period)) - 1
        period_returns[period_returns < 0] = 0
        weights = period_returns.div(period_returns.sum(axis=1), axis=0)
        weights = weights.fillna(0)
        weights = weights.shift(-1*prediction_period)
        return weights.ffill().round(5)

    def create_training_set(self, features, targets, lookback, dayofweek=4):
        # This is coded to create weekly rebalance on Friday
        closes = features['Adj Close']
        print('---- Creating Training Set')
        print(features.shape)
        print(targets.shape)
        print(closes.shape)
        
        X, y, y_price, y_date = [], [], [], []
        for i in range(len(features)-lookback):
            if features.index[i].dayofweek == dayofweek: # Friday
                X.append(features[i:i+lookback])
                y.append(targets[i+lookback:i+lookback+1])
                y_price.append(closes[i+lookback:i+lookback+1])
                y_date.append(closes.index[i+lookback])
        
        y_price = pd.DataFrame(index=np.array(y_date), data=np.array(y_price).reshape(len(y_price), targets.shape[1]), columns=closes.columns)
        return np.array(X), np.array(y).reshape(len(y), targets.shape[1]), y_price
