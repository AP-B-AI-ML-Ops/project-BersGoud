import logging
import os
from urllib.parse import urlparse

import mlflow
import pandas as pd
import tensorflow as tf
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report
from sqlalchemy import Boolean, Column, Float, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres"
)
Base = declarative_base()


# Define the Metrics table
class Metrics(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    data_drift_score = Column(Float)
    rmse_value = Column(Float)
    mae_value = Column(Float)
    model_version = Column(String)
    drift_detected = Column(Boolean)


# Set MLflow tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")


# Verify TensorFlow installation
def verify_tensorflow_installation():
    try:
        logger.info(f"TensorFlow version: {tf.__version__}")
    except ImportError as e:
        logger.error(
            f"TensorFlow is not installed or not found in the environment. {e}"
        )
        raise


# Load data
def load_data():
    try:
        data = pd.read_csv("./data/apple_stock_data.csv")
        logger.info("Data loaded successfully")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


# Clean the file path
def clean_path(path):
    parsed = urlparse(path)
    if parsed.scheme == "file":
        return parsed.path.lstrip("/")
    return path


# List directory contents
def list_directory(path):
    try:
        logger.info(f"Listing contents of directory: {path}")
        for root, dirs, files in os.walk(path):
            logger.info(f"Directory: {root}")
            for name in files:
                logger.info(f"File: {os.path.join(root, name)}")
            for name in dirs:
                logger.info(f"Subdirectory: {os.path.join(root, name)}")
    except Exception as e:
        logger.error(f"Error listing directory contents: {e}")
        raise


# Load model from MLflow
def load_model():
    try:
        model_name = "lstm-best-model"
        client = mlflow.tracking.MlflowClient()
        model_versions = client.search_model_versions(f"name='{model_name}'")
        # Find the latest model version
        latest_model_version = max(
            model_versions, key=lambda x: int(x.version)
        )  # Ensure version is int for comparison
        logger.info(f"Latest model version: {latest_model_version.version}")
        logger.info(f"Model source: {latest_model_version.source}")

        # Clean the model path
        model_path = clean_path(latest_model_version.source)
        logger.info(f"Cleaned model path: {model_path}")

        # Adjust path to point to the 'data/model' subdirectory where the actual model is located
        model_path = os.path.join(model_path, "data", "model")
        logger.info(f"Adjusted model path: {model_path}")

        # List directory contents for debugging
        list_directory(model_path)

        # Load model using TensorFlow directly
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully from MLflow using TensorFlow")
        return model, latest_model_version.version
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


# Calculate metrics using Evidently
def calculate_metrics(data, model):
    try:
        # Prepare data for prediction
        X = data.drop(
            columns=["Date", "Close"]
        )  # Assuming 'Close' is the target variable and 'Date' is not used for prediction
        y = data["Close"]  # Assuming 'Close' is the target variable
        X = X.values.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM input
        predictions = model.predict(X)

        result_data = pd.DataFrame(
            {
                "target": y,
                "prediction": predictions.flatten(),  # Flatten the predictions array
            }
        )

        report = Report(metrics=[DataDriftPreset(), RegressionPreset()])
        report.run(reference_data=result_data, current_data=result_data)
        logger.info("Metrics calculated successfully")

        # Save report to HTML
        report.save_html("evidently_report.html")
        logger.info("Report saved as HTML successfully")
        return report
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise


# Store metrics in the database
def store_metrics(report, model_version):
    try:
        report_dict = report.as_dict()
        metrics = report_dict.get("metrics", [])
        if not metrics or not isinstance(metrics, list):
            raise ValueError(
                "Unexpected structure in report dictionary: 'metrics' key missing or not a list"
            )

        # Extract data drift score
        data_drift_result = metrics[0].get("result", {})
        if isinstance(data_drift_result, dict):
            data_drift_score = (
                float(data_drift_result.get("dataset_drift", 0.0))
                if isinstance(data_drift_result.get("dataset_drift"), bool)
                else data_drift_result.get("dataset_drift", 0.0)
            )
        else:
            data_drift_score = 0.0

        # Extract regression metrics
        regression_metrics = metrics[1].get("result", {})
        rmse_value = regression_metrics.get("rmse", None)
        mae_value = regression_metrics.get("mean_abs_error", None)

        # Connect to the database and store metrics
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()

        # Create the metrics table if it doesn't exist
        Base.metadata.create_all(engine)

        # Store the metrics
        metric_entry = Metrics(
            data_drift_score=data_drift_score,
            rmse_value=float(rmse_value) if rmse_value is not None else None,
            mae_value=float(mae_value) if mae_value is not None else None,
            model_version=str(model_version),
            drift_detected=data_drift_score > 0.5,
        )
        session.add(metric_entry)
        session.commit()
        session.close()

        logger.info("Metrics stored successfully in the database")
    except Exception as e:
        logger.error(f"Error storing metrics: {e}")
        raise


def conditional_workflows(report, model_version):
    try:
        report_dict = report.as_dict()
        logger.info(f"Report dictionary: {report_dict}")

        metrics = report_dict.get("metrics", [])
        if not metrics or not isinstance(metrics, list):
            raise ValueError(
                "Unexpected structure in report dictionary: 'metrics' key missing or not a list"
            )

        # Extract data drift score
        data_drift_result = metrics[0].get("result", {})
        if isinstance(data_drift_result, dict):
            data_drift_score = data_drift_result.get("dataset_drift")
            if isinstance(data_drift_score, dict):
                data_drift_score = data_drift_score.get(
                    "drift_score", 0
                )  # Default to 0 if not found
            elif isinstance(data_drift_score, bool):
                data_drift_score = 0  # If it's a boolean, assume no drift
            else:
                raise ValueError(
                    "Unexpected structure in report dictionary: 'dataset_drift' key has unexpected type"
                )
        else:
            raise ValueError(
                "Unexpected structure in report dictionary: 'result' key missing or not a dictionary"
            )

        # Extract ME or MAE if RMSE is not available
        regression_metrics = metrics[1].get("result", {})
        logger.info(
            f"Regression metrics result: {regression_metrics}"
        )  # Log the regression metrics part

        # Fallback to MAE if RMSE is not available
        rmse_value = regression_metrics.get("rmse")
        if rmse_value is None:
            mae_value = regression_metrics.get("mean_abs_error")
            if mae_value is not None and mae_value > 100:
                log_alert()
        else:
            if rmse_value > 100:
                log_alert()

        if data_drift_score > 0.5:
            retrain_model()

        logger.info("Conditional workflows executed successfully")
    except Exception as e:
        logger.error(f"Error in conditional workflows: {e}")
        raise


# Retrain the model
def retrain_model():
    try:
        os.system("python project_main.py")
        logger.info("Model retraining triggered")
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        raise


# Log an alert to a file
def log_alert():
    try:
        with open("alerts.log", "a") as alert_file:
            alert_file.write(
                "The model performance has degraded. Please check the dashboard for details.\n"
            )
        logger.info("Alert logged successfully")
    except Exception as e:
        logger.error(f"Error logging alert: {e}")
        raise


# Main monitoring function
def monitor():
    try:
        verify_tensorflow_installation()
        data = load_data()
        model, model_version = load_model()
        report = calculate_metrics(data, model)
        store_metrics(report, model_version)
        conditional_workflows(report, model_version)
    except Exception as e:
        logger.error(f"Error in monitor function: {e}")


if __name__ == "__main__":
    monitor()
