import os
import pickle

import mlflow
import numpy as np
import optuna
from keras.layers import LSTM, Dense
from keras.models import Sequential
from prefect import flow, task
from tensorflow import keras


@task(name="load-pickle-hpo")
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task(name="split-data-hpo")
def split_data(X_train_scaled, y_train, validation_split=0.2):
    num_samples = len(X_train_scaled)
    split_index = int(num_samples * (1 - validation_split))
    X_train, X_val = X_train_scaled[:split_index], X_train_scaled[split_index:]
    y_train, y_val = y_train[:split_index], y_train[split_index:]
    return X_train, y_train, X_val, y_val


@task(name="optimize-hpo")
def optimize(X_train_scaled, y_train, X_val_scaled, y_val, num_trials):
    def objective(trial):
        units = trial.suggest_int("units", 10, 100, 10)
        dropout = trial.suggest_uniform("dropout", 0.0, 0.5)
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

        with mlflow.start_run():
            mlflow.log_params({"units": units, "dropout": dropout, "lr": lr})
            model = Sequential(
                [LSTM(units, input_shape=(X_train_scaled.shape[1], 1)), Dense(1)]
            )
            model.compile(optimizer="adam", loss="mse")
            history = model.fit(
                X_train_scaled,
                y_train,
                epochs=100,
                batch_size=64,
                validation_data=(X_val_scaled, y_val),
            )
            val_loss = np.min(history.history["val_loss"])
            mlflow.log_metric("val_loss", val_loss)

        return val_loss

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)


@flow(name="hpo-flow-hpo")
def hpo_flow(model_path: str, num_trials: int, experiment_name: str):
    mlflow.set_experiment(experiment_name)

    mlflow.tensorflow.autolog(disable=True)

    X_train_scaled = load_pickle(os.path.join(model_path, "X.pkl"))
    y_train = load_pickle(os.path.join(model_path, "y.pkl"))

    X_train_scaled, y_train, X_val_scaled, y_val = split_data(X_train_scaled, y_train)

    optimize(X_train_scaled, y_train, X_val_scaled, y_val, num_trials)
