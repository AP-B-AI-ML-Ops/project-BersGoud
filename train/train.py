import os
import pickle

import mlflow
import numpy as np
from keras.layers import LSTM, Dense
from keras.models import Sequential
from prefect import flow, task
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow import keras


@task(name="load-pickle-train")
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@task(name="start-ml-experiment-train")
def start_ml_experiment(X_train_scaled, y_train):
    print("Shape of X_train_scaled:", X_train_scaled.shape)
    with mlflow.start_run():
        model = Sequential(
            [LSTM(50, input_shape=(X_train_scaled.shape[1], 1)), Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse")
        # Train the model
        history = model.fit(
            X_train_scaled, y_train, epochs=100, batch_size=64, validation_split=0.2
        )

        # Log training metrics to MLflow
        mlflow.log_metrics(
            {
                "train_loss": history.history["loss"][-1],
                "val_loss": history.history["val_loss"][-1],
            }
        )


@flow(name="train-flow-train")
def train_flow(model_path: str):
    mlflow.set_experiment("lstm-train")
    mlflow.tensorflow.autolog()

    X_train_scaled = load_pickle(os.path.join(model_path, "X.pkl"))
    y_train = load_pickle(os.path.join(model_path, "y.pkl"))

    start_ml_experiment(X_train_scaled, y_train)
