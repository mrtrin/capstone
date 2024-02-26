import numpy as np
import pandas as pd

class Portfolio:
    @classmethod
    def portfolio_returns(cls, name, weights, prices):
        returns = prices.pct_change().shift(-1)
        port = pd.DataFrame((returns*weights).sum(axis=1), columns=[f'{name}_returns'])
        port[f'{name}_cumrets'] = (port[f'{name}_returns']+1).cumprod()
        return port
