import os
import pickle

import mlflow
from keras.layers import LSTM, Dense
from keras.models import Sequential
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow, task
from tensorflow import keras


@task(name="load-pickle-register")
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task(name="train-and-log-model-register")
def train_and_log_model(X_train_scaled, y_train, params):
    # Cast 'units' parameter to an integer
    units = int(params["units"])

    with mlflow.start_run():
        model = Sequential(
            [LSTM(units, input_shape=(X_train_scaled.shape[1], 1)), Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train_scaled, y_train, epochs=100, batch_size=64)


@task(name="get-experiment-runs-register")
def get_experiment_runs(top_n, hpo_experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(hpo_experiment_name)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.val_loss ASC"],
    )
    return runs


@task(name="select-best-model-register")
def select_best_model(top_n, experiment_name):
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.test_loss ASC"],
    )[0]

    return best_run


@flow(name="register-flow-register")
def register_flow(
    model_path: str, top_n: int, experiment_name: str, hpo_experiment_name: str
):
    mlflow.set_experiment(experiment_name)
    mlflow.tensorflow.autolog()

    X_train_scaled = load_pickle(os.path.join(model_path, "X.pkl"))
    y_train = load_pickle(os.path.join(model_path, "y.pkl"))

    # Retrieve the top_n model runs and log the models
    runs = get_experiment_runs(top_n, hpo_experiment_name)
    for run in runs:
        train_and_log_model(X_train_scaled, y_train, params=run.data.params)

    # Select the model with the lowest test loss
    best_run = select_best_model(top_n, experiment_name)

    # Register the best model
    run_id = best_run.info.run_id
    mlflow.register_model(f"runs:/{run_id}/model", "lstm-best-model")
