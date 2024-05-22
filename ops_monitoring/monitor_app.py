import datetime
import time
import os
import mlflow
import pandas as pd
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently import ColumnMapping
import psycopg2
from dotenv import load_dotenv

load_dotenv()

NUMERICAL = [
    'Volume', 'VWAP', 'Open', 'Close', 'High', 'Low', 'Timestamp', 'Number_of_Trades'
]

CATEGORICAL = ['Date']

COL_MAPPING = ColumnMapping(
    prediction='prediction',
    numerical_features=NUMERICAL,
    categorical_features=CATEGORICAL,
    target=None
)

CONNECT_STRING = f"host={os.getenv('POSTGRES_HOST')} port={os.getenv('POSTGRES_PORT')} user={os.getenv('POSTGRES_USER')} password={os.getenv('POSTGRES_PASSWORD')} dbname=test"

def prep_db():
    create_table_query = """
    DROP TABLE IF EXISTS metrics;
    CREATE TABLE metrics(
        timestamp TIMESTAMP,
        prediction_drift FLOAT,
        num_drifted_columns INTEGER,
        share_missing_values FLOAT
    );
    """

    with psycopg2.connect(CONNECT_STRING) as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1 FROM pg_database WHERE datname='test'")
            if cursor.fetchone() is None:
                cursor.execute("CREATE DATABASE test")
            cursor.execute(create_table_query)

def prep_data():
    ref_data = pd.read_csv('/mnt/data/apple_stock_data.csv')  # Adjusted to your uploaded CSV
    model = load_model()
    raw_data = pd.read_csv('/mnt/data/apple_stock_data.csv')  # Assuming similar structure
    return ref_data, model, raw_data

def load_model():
    model_uri = "models:/lstm-best-model/production"
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def calculate_metrics(current_data, model, ref_data):
    current_data['prediction'] = model.predict(current_data[NUMERICAL].fillna(0))

    report = Report(metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric()
    ])

    report.run(reference_data=ref_data, current_data=current_data, column_mapping=COL_MAPPING)
    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_cols = result['metrics'][1]['result']['number_of_drifted_columns']
    share_missing_vals = result['metrics'][2]['result']['current']['share_of_missing_values']

    return prediction_drift, num_drifted_cols, share_missing_vals

def save_metrics_to_db(date, prediction_drift, num_drifted_cols, share_missing_vals):
    conn = psycopg2.connect(CONNECT_STRING)
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO metrics(
        timestamp, prediction_drift, num_drifted_columns, share_missing_values
    ) VALUES (%s, %s, %s, %s);
    """, (date, prediction_drift, num_drifted_cols, share_missing_vals))
    conn.commit()
    cursor.close()
    conn.close()

def monitor():
    startDate = datetime.datetime(2023, 2, 1, 0, 0)
    endDate = datetime.datetime(2023, 2, 2, 0, 0)

    prep_db()
    ref_data, model, raw_data = prep_data()

    while True:
        current_data = raw_data[(raw_data['Timestamp'] >= startDate.timestamp()) &
                                (raw_data['Timestamp'] < endDate.timestamp())]
        prediction_drift, num_drifted_cols, share_missing_vals = calculate_metrics(current_data, model, ref_data)
        save_metrics_to_db(startDate, prediction_drift, num_drifted_cols, share_missing_vals)

        startDate += datetime.timedelta(1)
        endDate += datetime.timedelta(1)

        time.sleep(86400)  # Run daily
        print("Metrics updated")

if __name__ == '__main__':
    monitor()
