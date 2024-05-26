import datetime
import unittest
from unittest.mock import mock_open, patch

from load_project import collect_data


class TestCollectDataTasks(unittest.TestCase):

    @patch("polygon.RESTClient.get_aggs")
    def test_fetch_polygon_data(self, mock_get_aggs):
        mock_get_aggs.return_value.data = '{"results": [{"t": 1234567890, "v": 1000, "vw": 150.0, "o": 150.0, "c": 155.0, "h": 160.0, "l": 145.0}]}'
        start_date = "2022-01-01"
        end_date = "2022-01-31"
        result = collect_data.fetch_polygon_data(start_date, end_date)
        mock_get_aggs.assert_called_once()
        self.assertEqual(result["results"][0]["v"], 1000)

    @patch("builtins.open", new_callable=mock_open)
    @patch("csv.DictWriter.writerow")
    def test_save_data(self, mock_writerow, mock_open):
        data = {
            "results": [
                {
                    "t": 1234567890,
                    "v": 1000,
                    "vw": 150.0,
                    "o": 150.0,
                    "c": 155.0,
                    "h": 160.0,
                    "l": 145.0,
                }
            ]
        }
        collect_data.save_data.fn(
            data, "dummy.csv"
        )  # Use .fn to call the function directly
        mock_open.assert_called_once_with("dummy.csv", "w", newline="")
        mock_writerow.assert_called()

    def test_generate_query_params(self):
        start_date, end_date = collect_data.generate_query_params()
        self.assertIsInstance(start_date, datetime.datetime)
        self.assertIsInstance(end_date, datetime.datetime)
        self.assertTrue((end_date - start_date).days <= 600)


if __name__ == "__main__":
    unittest.main()
