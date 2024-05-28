import unittest
from unittest.mock import mock_open, patch

import pandas as pd

from load_project.prep_data import (
    dump_pickle,
    preprocess_labels_financial_data,
    preprocess_stock_data,
    read_csv_stock_data,
    read_financial_data,
    save_preprocessed_data,
)


class TestPrepDataTasks(unittest.TestCase):
    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="date,v,vw,o,c,h,l,t,n\n2023-01-01,1000,150.0,150.0,155.0,160.0,145.0,1234567890,100",
    )
    def test_read_financial_data(self, mock_open):
        result = read_financial_data.fn(
            "dummy.csv"
        )  # Use .fn to call the function directly
        mock_open.assert_called_once_with("dummy.csv", "r")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["v"], "1000")

    def test_preprocess_labels_financial_data(self):
        data = [
            {
                "v": "1000",
                "vw": "150.0",
                "o": "150.0",
                "c": "155.0",
                "h": "160.0",
                "l": "145.0",
                "t": "1234567890",
                "n": "100",
                "date": "2023-01-01",
            }
        ]
        result = preprocess_labels_financial_data.fn(
            data
        )  # Use .fn to call the function directly
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["Volume"], 1000)

    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.dump")
    @patch("prefect.context")
    def test_dump_pickle(self, mock_context, mock_pickle_dump, mock_open):
        mock_context.__enter__ = lambda s: None
        mock_context.__exit__ = lambda s, t, v, tb: None
        obj = {"key": "value"}
        dump_pickle.fn(obj, "dummy.pkl")  # Use .fn to call the function directly
        mock_open.assert_called_once_with("dummy.pkl", "wb")
        mock_pickle_dump.assert_called_once_with(obj, mock_open())

    @patch("pandas.read_csv")
    def test_read_csv_stock_data(self, mock_read_csv):
        mock_read_csv.return_value = "dummy_data"
        result = read_csv_stock_data.fn(
            "dummy.csv"
        )  # Use .fn to call the function directly
        mock_read_csv.assert_called_once_with("dummy.csv")
        self.assertEqual(result, "dummy_data")

    def test_preprocess_stock_data(self):

        data = {
            "Date": ["2023-01-01", "2023-01-02"],
            "Close": [155.0, 160.0],
            "Volume": [1000, 2000],
            "VWAP": [150.0, 155.0],
            "Open": [150.0, 158.0],
            "High": [160.0, 165.0],
            "Low": [145.0, 150.0],
            "Number_of_Trades": [100, 200],
        }
        df = pd.DataFrame(data)
        result = preprocess_stock_data.fn(df)  # Use .fn to call the function directly
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1].iloc[0], 155.0)

    @patch("builtins.open", new_callable=mock_open)
    @patch("csv.DictWriter.writerows")
    def test_save_preprocessed_data(self, mock_writerows, mock_open):
        data = [
            {
                "Volume": 1000,
                "VWAP": 150.0,
                "Open": 150.0,
                "Close": 155.0,
                "High": 160.0,
                "Low": 145.0,
                "Timestamp": 1234567890,
                "Number_of_Trades": 100,
                "Date": "2023-01-01",
            }
        ]
        save_preprocessed_data.fn(
            data, "dummy.csv"
        )  # Use .fn to call the function directly
        mock_open.assert_called_once_with("dummy.csv", "w", newline="")
        mock_writerows.assert_called_once()


if __name__ == "__main__":
    unittest.main()
