import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from train_project.train import load_pickle, start_ml_experiment


class TestTrainTasks(unittest.TestCase):
    @patch("builtins.open", new_callable=unittest.mock.mock_open, read_data="data")
    @patch("pickle.load")
    def test_load_pickle(self, mock_pickle_load, mock_open):
        mock_pickle_load.return_value = {"key": "value"}
        result = load_pickle.fn("dummy.pkl")  # Use .fn to call the function directly
        mock_open.assert_called_once_with("dummy.pkl", "rb")
        mock_pickle_load.assert_called_once()
        self.assertEqual(result, {"key": "value"})

    @patch("mlflow.start_run")
    @patch("keras.models.Sequential.fit")
    @patch("mlflow.log_metrics")
    def test_start_ml_experiment(self, mock_log_metrics, mock_fit, mock_start_run):
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run

        X_train_scaled = np.array(
            [[[0.1]]] * 100
        )  # Adjusted to 3D array for LSTM input
        y_train = np.array([0.1] * 100)  # Convert to NumPy array

        start_ml_experiment.fn(
            X_train_scaled, y_train
        )  # Use .fn to call the function directly

        mock_start_run.assert_called_once()
        mock_fit.assert_called_once()
        mock_log_metrics.assert_called_once()


if __name__ == "__main__":
    unittest.main()
