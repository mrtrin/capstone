from src import datasets

import pandas as pd
import unittest

class MockDatasets(datasets.Dataset):
    def __init__(self, tickers, start_date, end_date, mock_data):
        self.mock_data = mock_data
        super().__init__(tickers, start_date, end_date)

    def _download(self, tickers, start_date, end_date):
        return self.mock_data

class TestDatasets(unittest.TestCase):

    def test_target(self):
        mock_data = pd.DataFrame([
            [1, 1],
            [1, 2],
            [2, 1],
            [0, 1]
        ], columns=[('Adj Close', 'AAPL'), ('Adj Close', 'TSLA')])
        mock_data.columns=pd.MultiIndex.from_tuples(mock_data.columns)

        ds = MockDatasets(['AAPL', 'TSLA'], '2022-01-01', '2022-01-03', mock_data)
        targets = ds.targets(ds.data, 1)
        expected = pd.DataFrame([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 1]
        ], columns=['AAPL','TSLA', 'CASH'], dtype=float)
        
        pd.testing.assert_frame_equal(expected, targets.round(decimals=2))
