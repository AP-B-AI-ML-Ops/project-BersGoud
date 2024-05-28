import unittest
from unittest.mock import MagicMock, mock_open, patch

import numpy as np

from train_project.register import (
    get_experiment_runs,
    load_pickle,
    select_best_model,
    train_and_log_model,
)


class TestRegisterTasks(unittest.TestCase):
    @patch("builtins.open", new_callable=mock_open, read_data="data")
    @patch("pickle.load")
    @patch("prefect.client.get_client")
    async def test_load_pickle(self, mock_get_client, mock_pickle_load, mock_open):
        mock_client = MagicMock()
        mock_get_client.return_value.__aenter__.return_value = mock_client
        mock_pickle_load.return_value = {"key": "value"}

        result = await load_pickle("dummy.pkl")

        mock_open.assert_called_once_with("dummy.pkl", "rb")
        mock_pickle_load.assert_called_once()
        self.assertEqual(result, {"key": "value"})

    @patch("mlflow.start_run")
    @patch("keras.models.Sequential.fit")
    def test_train_and_log_model(self, mock_fit, mock_start_run):
        X_train_scaled = np.array([[0.1]] * 100)  # Convert to NumPy array
        y_train = np.array([0.1] * 100)  # Convert to NumPy array
        params = {"units": 50}

        with patch("mlflow.log_metrics"), patch("mlflow.log_params"), patch(
            "mlflow.keras.log_model"
        ):
            result = train_and_log_model(X_train_scaled, y_train, params)

        mock_start_run.assert_called_once()
        mock_fit.assert_called_once()
        self.assertIsNone(result)

    @patch("mlflow.tracking.MlflowClient.get_experiment_by_name")
    @patch("mlflow.tracking.MlflowClient.search_runs")
    def test_get_experiment_runs(self, mock_search_runs, mock_get_experiment_by_name):
        mock_get_experiment_by_name.return_value = MagicMock(experiment_id="1")
        mock_search_runs.return_value = ["run1", "run2"]

        result = get_experiment_runs(2, "dummy_experiment")

        mock_get_experiment_by_name.assert_called_once_with("dummy_experiment")
        mock_search_runs.assert_called_once()
        self.assertEqual(result, ["run1", "run2"])

    @patch("mlflow.tracking.MlflowClient.get_experiment_by_name")
    @patch("mlflow.tracking.MlflowClient.search_runs")
    def test_select_best_model(self, mock_search_runs, mock_get_experiment_by_name):
        mock_get_experiment_by_name.return_value = MagicMock(experiment_id="1")
        mock_search_runs.return_value = ["run1"]

        result = select_best_model(1, "dummy_experiment")

        mock_get_experiment_by_name.assert_called_once_with("dummy_experiment")
        mock_search_runs.assert_called_once()
        self.assertEqual(result, "run1")


if __name__ == "__main__":
    unittest.main()
