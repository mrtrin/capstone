import numpy as np
import pandas as pd

class Portfolio:
    @classmethod
    def portfolio_returns(cls, weights, prices):
        weights = pd.DataFrame(weights)
        returns = pd.DataFrame(prices).pct_change().shift(-1)
        return weights.multiply(returns).sum(axis=1)