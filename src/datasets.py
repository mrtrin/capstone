import numpy as np
import yfinance as yf

def load_train_test(lookback_period=10, rebalance_period=5, split=0.8):
    tickers = ['AAPL', 'MSFT', 'GOOG', 'META', 'TSLA', 'SPY']
    dataset = Dataset(tickers, '2010-01-01', '2023-12-31')
    features = dataset.features()
    targets = dataset.targets(rebalance_period)
    X, y = [], []
    for i in range(len(features)-lookback_period):
        if features.index[i].dayofweek == 4: # Friday
            X.append(features[i:i+lookback_period])
            y.append(targets[i+lookback_period:i+lookback_period+1])
    return np.array(X), np.array(y).reshape(len(y), targets.shape[1])

class Dataset:

    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.download(tickers, start=start_date, end=end_date)
        self.data = self._clean(self.data)

    def _clean(self, df):
        return df.fillna(method='ffill')

    def features(self):
        # TODO: Add Features based on technical signals
        
        close = self.data['Adj Close']
        close.columns = [f'{col}_close' for col in close.columns]
        volume = self.data['Volume']
        volume.columns = [f'{col}_volume' for col in volume.columns]

        return close.join(volume)
    
    def targets(self, period):
        period_returns = (self.data['Adj Close'] / self.data['Adj Close'].shift(period)) - 1
        period_returns[period_returns < 0] = 0
        weights = period_returns.div(period_returns.sum(axis=1), axis=0)
        weights = weights.shift(-1*period)
        return weights
