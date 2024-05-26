import mlflow
from prefect import flow

from load_project.collect_data import collect_flow
from load_project.prep_data import prep_data_flow
from train_project.hpo import hpo_flow
from train_project.register import register_flow
from train_project.train import train_flow

HPO_EXPERIMENT_NAME = "LSTM-hyperopt"
REG_EXPERIMENT_NAME = "LSTM-best-models"


@flow
def main_flow():
    print("start main flow")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    collect_flow()
    prep_data_flow(
        data_path="data/apple_stock_data.csv",
        dest_path="data/apple_stock_data.csv",
        model_path="./models/",
    )

    train_flow(model_path="./models/")
    hpo_flow(model_path="./models/", num_trials=5, experiment_name=HPO_EXPERIMENT_NAME)
    register_flow(
        model_path="./models/",
        top_n=5,
        experiment_name=REG_EXPERIMENT_NAME,
        hpo_experiment_name=HPO_EXPERIMENT_NAME,
    )


if __name__ == "__main__":
    main_flow()
