from src import datasets, portfolio

import pandas as pd
import unittest

class MockDatasets(datasets.Dataset):
    def __init__(self, tickers, start_date, end_date, mock_data):
        self.mock_data = mock_data
        super().__init__(tickers, start_date, end_date)

    def _download(self, tickers, start_date, end_date):
        return self.mock_data

class TestPortfolio(unittest.TestCase):

    def test_target(self):
        mock_data = pd.DataFrame([
            [1, 1],
            [1, 1.1],
            [1.2, 1],
            [0.8, 1]
        ], columns=[('Adj Close', 'AAPL'), ('Adj Close', 'TSLA')])
        mock_data.columns=pd.MultiIndex.from_tuples(mock_data.columns)

        ds = MockDatasets(['AAPL', 'TSLA'], '2022-01-01', '2022-01-03', mock_data)
        ds.data['Adj Close']

        weights = pd.DataFrame([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1]
        ], columns=['AAPL','TSLA', 'CASH'], dtype=float)

        returns = portfolio.Portfolio.portfolio_returns('unittest', weights, ds.data['Adj Close'])

        expected = pd.DataFrame([
            [0.1, 1.1],
            [0.2, 1.32],
            [0.000027, 1.320036],
            [0.0, 1.320036]
        ], columns=['unittest_returns', 'unittest_cumrets'])

        pd.testing.assert_frame_equal(expected.round(5), returns.round(5))