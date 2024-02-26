import numpy as np
import pandas as pd

class Portfolio:
    @classmethod
    def portfolio_returns(cls, name, weights, prices):
        weights = pd.DataFrame(weights)
        returns = pd.DataFrame(prices).pct_change().shift(-1)
        port = pd.DataFrame(weights.multiply(returns).sum(axis=1), columns=[f'{name}_returns'])
        port[f'{name}_cumrets'] = (port[f'{name}_returns']+1).cumprod()
        return port
