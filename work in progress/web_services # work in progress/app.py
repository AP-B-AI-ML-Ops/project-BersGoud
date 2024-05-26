"""import os

import mlflow.pyfunc
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

app = Flask(__name__)


def get_db_connection():
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=os.getenv("POSTGRES_PORT"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        dbname=os.getenv("POSTGRES_DB"),
    )
    return conn


def load_model():
    model_uri = os.getenv("MODEL_URI")
    model = mlflow.pyfunc.load_model(model_uri)
    return model


@app.route("/")
def home():
    return "Welcome to the Forecasting Web Service"


@app.route("/forecast", methods=["POST"])
def forecast():
    data = request.json
    df = pd.DataFrame(data)

    model = load_model()
    predictions = model.predict(df)
    df["prediction"] = predictions

    conn = get_db_connection()
    df.to_sql("predictions", conn, if_exists="append", index=False)

    return jsonify(predictions.tolist())


@app.route("/retrain", methods=["POST"])
def retrain():
    # Placeholder for retraining logic
    return "Model retraining initiated"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)"""
