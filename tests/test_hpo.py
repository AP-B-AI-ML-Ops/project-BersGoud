import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from train_project.hpo import load_pickle, optimize, split_data


class TestHPOTasks(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("pickle.load")
    def test_load_pickle(self, mock_pickle_load, mock_open):
        mock_pickle_load.return_value = {"key": "value"}
        result = load_pickle.fn("dummy.pkl")  # Use .fn to call the function directly
        mock_open.assert_called_once_with("dummy.pkl", "rb")
        mock_pickle_load.assert_called_once()
        self.assertEqual(result, {"key": "value"})

    def test_split_data(self):
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, 2])
        result = split_data.fn(X, y)
        np.testing.assert_array_equal(result[0], X[:1])
        np.testing.assert_array_equal(result[1], y[:1])
        np.testing.assert_array_equal(result[2], X[1:])
        np.testing.assert_array_equal(result[3], y[1:])

    @patch("optuna.create_study")
    @patch("mlflow.start_run")
    def test_optimize(self, mock_start_run, mock_create_study):
        mock_study = MagicMock()
        mock_create_study.return_value = mock_study
        mock_study.optimize.return_value = None

        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([1, 2])
        X_val = np.array([[5, 6]])
        y_val = np.array([3])

        result = optimize.fn(X_train, y_train, X_val, y_val, 10)
        print(result)
        mock_create_study.assert_called_once()
        mock_study.optimize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
